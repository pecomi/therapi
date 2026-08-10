
import argparse
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

EMBED_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(EMBED_DIR, os.pardir))
DATA_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir, 'data'))
CSG2A_DIR = os.path.join(EMBED_DIR, 'CSG2A')
sys.path.insert(0, CSG2A_DIR)

from models.CSG2A_net import CSG2A_finetune
from utils.utils_data import featurize_mol, pad_array

SOURCE_DATASET = 'GDSC'   # cell-line domain; everything else is a target domain


def _load_therapi_model_module():
    path = os.path.join(os.path.abspath(SRC_DIR), 'model.py')
    spec = importlib.util.spec_from_file_location('therapi_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# SMILES featurisation
# ---------------------------------------------------------------------------
def _conformer_mol(smiles, seed):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, maxAttempts=100, randomSeed=seed) == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol_h)
        except Exception:
            pass  # unoptimised conformer is still usable
        return Chem.RemoveHs(mol_h)

    # 3D embedding failed -- fall back to 2D coordinates on the H-free molecule
    AllChem.Compute2DCoords(mol)
    return mol


def featurize_unique_smiles(smiles_list, seed, cache_path=None):

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if set(cached) >= set(smiles_list):
            print(f'[featurize] loaded cached features from {cache_path}')
            return cached

    RDLogger.DisableLog('rdApp.*')
    feats = {}
    failed = []
    for smi in tqdm(smiles_list, desc='featurize SMILES'):
        mol = _conformer_mol(smi, seed)
        if mol is None:
            failed.append(smi)
            continue
        try:
            feats[smi] = featurize_mol(mol, add_dummy_node=True, one_hot_formal_charge=True)
        except Exception as e:  # noqa: BLE001 -- report the offending SMILES, don't drop it
            failed.append(f'{smi} ({e})')
    RDLogger.EnableLog('rdApp.*')

    if failed:
        raise RuntimeError(
            f'{len(failed)} SMILES could not be featurised; refusing to emit a '
            f'misaligned embedding matrix. First few: {failed[:5]}'
        )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(feats, f)
        print(f'[featurize] cached {len(feats)} molecules to {cache_path}')
    return feats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PerturbationPairs(Dataset):

    def __init__(self, gex_np, sample_rows, mol_feats, feat_ids):
        self.gex_np = gex_np              # (n_samples, n_genes) float32
        self.sample_rows = sample_rows    # pair -> row index into gex_np
        self.mol_feats = mol_feats        # list of (afm, adj, dist)
        self.feat_ids = feat_ids          # pair -> index into mol_feats

    def __len__(self):
        return len(self.sample_rows)

    def __getitem__(self, i):
        afm, adj, dist = self.mol_feats[self.feat_ids[i]]
        return afm, adj, dist, self.gex_np[self.sample_rows[i]]


def collate_pairs(batch):
 
    max_atoms = max(item[0].shape[0] for item in batch)
    afm = np.stack([pad_array(b[0], (max_atoms, b[0].shape[1])) for b in batch])
    adj = np.stack([pad_array(b[1], (max_atoms, max_atoms)) for b in batch])
    dist = np.stack([pad_array(b[2], (max_atoms, max_atoms)) for b in batch])
    gex = np.stack([b[3] for b in batch])
    return (
        torch.from_numpy(afm).float(),
        torch.from_numpy(adj).float(),
        torch.from_numpy(dist).float(),
        torch.from_numpy(gex).float(),
    )


# ---------------------------------------------------------------------------
# PPI adjacency
# ---------------------------------------------------------------------------
def build_ppi_adj(genes, string_edges_path):
    genes = pd.Index(genes)
    string_df = pd.read_csv(string_edges_path)
    string_df = string_df[string_df['source'].isin(genes) & string_df['target'].isin(genes)]

    adj = torch.eye(len(genes))
    src = genes.get_indexer(string_df['source'].values)
    tgt = genes.get_indexer(string_df['target'].values)
    adj[tgt, src] = 1.0
    adj[src, tgt] = 1.0
    print(f'[ppi] {len(string_df)} STRING edges retained over {len(genes)} genes '
          f'(density {adj.mean().item():.4f})')
    return adj


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_csg2a_net(genes, args, device):
    torch.manual_seed(args.seed)
    model = CSG2A_finetune(
        gex_dim=len(genes),
        CSG2A_params=dict(
            gex_dim=len(genes),
            hdim=args.gene_hdim,
            dropout=args.dropout,
            ppi_adj=build_ppi_adj(genes, args.string_edges).to(device),
            bias=False,
        ),
        dropout=args.dropout,
    ).to(device)

    state_dict = torch.load(args.csg2a_ckpt, map_location=device)
    state_dict = state_dict.get('state_dict', state_dict)
    model.CSG2A.load_state_dict(state_dict, strict=True)


    return model


# ---------------------------------------------------------------------------
# Alignment (target domain only)
# ---------------------------------------------------------------------------
def align_expression(args, target_gex_df, device):
 
    therapi = _load_therapi_model_module()

    source_gex_df = pd.read_csv(
        os.path.join(args.data_dir, 'GDSC/GDSC_gex.csv'), index_col=0)
    source_info_df = pd.read_csv(
        os.path.join(args.data_dir, 'GDSC/GDSC_info.csv'))
    num_tissue = source_info_df['tissue_label'].nunique()

    source_AE = therapi.SOURCE_AE(
        n_genes=source_gex_df.shape[1], n_classes=num_tissue, n_latent=args.latent_dim
    ).to(device)
    target_we = therapi.TARGET_weightencoder(
        n_genes=target_gex_df.shape[1], n_latent=args.latent_dim,
        n_celines=source_gex_df.shape[0],
    ).to(device)

    ckpt = torch.load(args.aligner_ckpt, map_location=device)
    source_AE.load_state_dict(ckpt['source_AE'])
    target_we.load_state_dict(ckpt['target_weightencoder'])
    source_AE.eval()
    target_we.eval()

    source_gex = torch.tensor(source_gex_df.values, dtype=torch.float32, device=device)
    target_gex = torch.tensor(target_gex_df.values, dtype=torch.float32, device=device)

    with torch.no_grad():
        source_z, _ = source_AE(source_gex)
        weights, _, wgex, _ = target_we(target_gex, source_z, source_gex)

    aligned = pd.DataFrame(
        wgex.cpu().numpy().astype(np.float32),
        index=target_gex_df.index,
        columns=source_gex_df.columns,
    )
    top = weights.max(dim=1).values
    print(f'[align] {args.aligner_ckpt}')
    print(f'[align] x\'_t {aligned.shape} from {source_gex_df.shape[0]} cell lines; '
          f'max attention weight mean={top.mean():.4f} (uniform would be '
          f'{1 / source_gex_df.shape[0]:.4f})')
    return aligned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_embeddings(args):
    device = torch.device(args.device)

    resp_path = os.path.join(args.data_dir, args.dataset, f'{args.dataset}_Durg_SMILES_Response.csv')

    # target domains ship only the labelled subset
    gex_name = f'{args.dataset}_gex.csv' if args.dataset == SOURCE_DATASET \
        else f'{args.dataset}_labeled_gex.csv'
    gex_path = os.path.join(args.data_dir, args.dataset, gex_name)
    out_pert = args.out_pert or os.path.join(
        args.data_dir, args.dataset, f'{args.dataset}_perturbation_float16.npy')
    out_comp = args.out_comp or os.path.join(
        args.data_dir, args.dataset, f'{args.dataset}_perturbation_compound_float16.npy')

    for path in (out_pert, out_comp):
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f'{path} already exists; pass --overwrite to replace it')

    # --- response pairs ----------------------------------------------------
    resp_df = pd.read_csv(resp_path)
    print(f'[data] {len(resp_df)} response pairs from {resp_path}')

    # --- basal expression --------------------------------------------------
    gex_df = pd.read_csv(gex_path, index_col=0)
    print(f'[data] basal expression {gex_df.shape} from {gex_path}')

    if args.aligner_ckpt:
        gex_df = align_expression(args, gex_df, device)
    elif args.dataset != SOURCE_DATASET:
        print(f'[warn] --aligner_ckpt not given for target domain {args.dataset}; '
              f'feeding raw expression to CSG2A. The paper aligns it first '
              f'(Equation 11).')
    genes = list(gex_df.columns)

    # TCGA expression is keyed by full barcode (TCGA-A2-A0EP-01) while the
    # response table uses the trimmed patient ID (TCGA-A2-A0EP).
    gex_index = gex_df.index.astype(str)
    id_to_row = {sample_id: i for i, sample_id in enumerate(gex_index)}
    by_patient = {}
    for i, sample_id in enumerate(gex_index):
        by_patient.setdefault('-'.join(sample_id.split('-')[:3]), []).append(i)
    id_to_row.update({p: rows[0] for p, rows in by_patient.items() if p not in id_to_row})

    resp_ids = resp_df[args.id_column].astype(str).values
    unknown = sorted({s for s in resp_ids if s not in id_to_row})
    if unknown:
        raise KeyError(f'{len(unknown)} response IDs have no expression profile in '
                       f'{gex_path}. First few: {unknown[:5]}')
    ambiguous = sorted({s for s in resp_ids
                        if s not in set(gex_index) and len(by_patient.get(s, ())) > 1})
    if ambiguous:
        picked = ', '.join(f'{s} -> {gex_index[by_patient[s][0]]}' for s in ambiguous[:5])
        print(f'[warn] {len(ambiguous)} response IDs match more than one barcode; '
              f'using the first occurrence of each: {picked}')
    sample_rows = np.array([id_to_row[s] for s in resp_ids], dtype=np.int64)
    gex_np = np.ascontiguousarray(gex_df.values, dtype=np.float32)

    # --- molecules ---------------------------------------------------------
    smiles = resp_df[args.smiles_column].astype(str).values
    unique_smiles = sorted(set(smiles))
    print(f'[data] {len(unique_smiles)} unique compounds')
    cache_path = args.feature_cache or os.path.join(
        args.data_dir, args.dataset, f'{args.dataset}_molfeat_dn_ohfc.p')
    feat_dict = featurize_unique_smiles(unique_smiles, args.seed, cache_path)

    smiles_order = {smi: i for i, smi in enumerate(unique_smiles)}
    mol_feats = [feat_dict[smi] for smi in unique_smiles]
    feat_ids = np.array([smiles_order[s] for s in smiles], dtype=np.int64)

    # --- model -------------------------------------------------------------
    model = build_csg2a_net(genes, args, device)
    model.eval()

    # --- inference ---------------------------------------------------------
    loader = DataLoader(
        PerturbationPairs(gex_np, sample_rows, mol_feats, feat_ids),
        batch_size=args.batch_size,
        shuffle=False,          # output rows must stay aligned with the response table
        collate_fn=collate_pairs,
        num_workers=args.num_workers,
    )

    n_pairs, n_genes = len(resp_df), len(genes)
    pert_out = np.empty((n_pairs, n_genes), dtype=np.float16)
    comp_out = np.empty((n_pairs, n_genes), dtype=np.float16)

    cursor = 0
    with torch.no_grad():
        for afm, adj, dist, gex in tqdm(loader, desc='CSG2A inference'):
            afm, adj = afm.to(device), adj.to(device)
            dist, gex = dist.to(device), gex.to(device)

            bsz = gex.shape[0]
            dose = torch.full((bsz, 1), args.dose, device=device)
            time = torch.full((bsz, 1), args.time, device=device)
            mask = torch.sum(torch.abs(afm), dim=-1) != 0
            # the IC50 head is computed and discarded; only the two 978-d
            # representations feed the THERAPI predictor
            _, pred_gex, comp = model(gex, afm, mask, adj, dist, dose, time)

            pert_out[cursor:cursor + bsz] = pred_gex.cpu().numpy().astype(np.float16)
            comp_out[cursor:cursor + bsz] = comp.cpu().numpy().astype(np.float16)
            cursor += bsz

    assert cursor == n_pairs, f'wrote {cursor} rows, expected {n_pairs}'

    os.makedirs(os.path.dirname(out_pert), exist_ok=True)
    np.save(out_pert, pert_out)
    np.save(out_comp, comp_out)
    print(f'[done] {out_pert}  {pert_out.shape} {pert_out.dtype}')
    print(f'[done] {out_comp}  {comp_out.shape} {comp_out.dtype}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build CSG2A embeddings for THERAPI')

    # data
    parser.add_argument('--dataset', type=str, default='GDSC')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--id_column', type=str, default='ID')
    parser.add_argument('--smiles_column', type=str, default='Canonical_SMILES')

    # aligner (target domains only)
    parser.add_argument('--aligner_ckpt', type=str, default=None)
    parser.add_argument('--latent_dim', type=int, default=128)

    # CSG2A
    parser.add_argument('--csg2a_ckpt', type=str,
                        default='../CSG2A/ckpts/CSG2A_pretrain_landmark.pt')
    parser.add_argument('--string_edges', type=str,
                        default='../CSG2A/data/STRING_edges.csv')
    parser.add_argument('--gene_hdim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dose', type=float, default=0.1)   # 0.1 == 10 uM
    parser.add_argument('--time', type=float, default=1.0)   # 1.0 == 72 h

    # runtime
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)

    # output
    parser.add_argument('--feature_cache', type=str, default=None)
    parser.add_argument('--out_pert', type=str, default=None)
    parser.add_argument('--out_comp', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')

    build_embeddings(parser.parse_args())
