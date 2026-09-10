import json
import os
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from model import *
from unlearning.loss_history import (
    RunLogger,
    format_epoch_log,
    plot_history,
    plot_retain_history,
    split_metrics_row,
    write_history,
)
from unlearning.objective import EVALUATION_BATCH_SIZE, evaluate_loader
from unlearning.split import build_sample_table, load_manifest_indices
from utils import set_seed
from center_loss import CenterLoss


def _validate_args(args):
    positive = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f'arguments must be positive: {invalid}')
    loss_weights = (args.recon_weight, args.class_weight, args.center_weight)
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError(
            'loss weights must be non-negative and at least one must be positive'
        )


def train_aligner(args):
    _validate_args(args)
    model_name = f'THERAPI_aligner_{args.source}_{args.target}'
    output_dir = Path('ckpts')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / 'training.log')
    set_seed(args.seed, logger=lambda _: None)

    # parameters
    batch_size = args.batch_size
    dim_latent = args.latent_dim
    lr = args.lr
    loss_a = args.recon_weight
    loss_b = args.center_weight
    loss_c = args.class_weight

    # load data
    source_data_dir = os.path.join(args.data_dir, f'{args.source}/{args.source}_gex.csv')
    source_info_dir = os.path.join(args.data_dir, f'{args.source}/{args.source}_info.csv')

    target_unlabeled_data_dir = os.path.join(args.data_dir, f'{args.target}/{args.target}_unlabeled_gex.csv')
    target_unlabeled_info_dir = os.path.join(args.data_dir, f'{args.target}/{args.target}_unlabeled_info.csv')
    
    source_data_df = pd.read_csv(source_data_dir, index_col=0)
    source_info_df = pd.read_csv(source_info_dir)    
    num_tissue = len(source_info_df['tissue_label'].unique())

    target_unlabeled_data_df = pd.read_csv(target_unlabeled_data_dir, index_col=0)
    target_unlabeled_info_df = pd.read_csv(target_unlabeled_info_dir)

    source_dataset = AlignerDataset(source_data_df, args.source, source_info_df['tissue_label'])
    target_unlabeled_dataset = AlignerDataset(
        target_unlabeled_data_df,
        args.target,
        target_unlabeled_info_df[args.tissue_column],
    )
    target_unlabeled_dataloader = DataLoader(target_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator = torch.Generator().manual_seed(args.seed))

    # model
    source_AE = SOURCE_AE(n_genes=source_dataset.n_genes, n_classes=num_tissue, n_latent=dim_latent)
    target_weightencoder = TARGET_weightencoder(n_genes=target_unlabeled_dataset.n_genes, n_latent=dim_latent, n_celines=source_data_df.shape[0])
    emb_dis_classifier = Emb_Dis_classifier(n_latent=dim_latent, n_classes=num_tissue)
    exp_dis_classifier = Exp_Dis_classifier(n_genes=target_unlabeled_dataset.n_genes, n_latent=dim_latent, n_classes=num_tissue)
    source_AE.to(args.device)
    target_weightencoder.to(args.device)
    emb_dis_classifier.to(args.device)
    exp_dis_classifier.to(args.device)

    autoencoder_criterion = nn.MSELoss()
    center_criterion = CenterLoss(num_classes=num_tissue, feat_dim=dim_latent, device=args.device)
    classifier_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(source_AE.parameters())+list(target_weightencoder.parameters())+list(emb_dis_classifier.parameters())+list(exp_dis_classifier.parameters()), lr=lr)

    # Optional split tracking does not participate in optimization. It evaluates
    # the target loss on fixed, full forget/retain sets before training and after
    # every completed epoch, using the same evaluator as the unlearning scripts.
    history = []
    initial_forget = initial_retain = None
    forget_indices = retain_indices = None
    forget_eval_loader = retain_eval_loader = None
    models = (source_AE, target_weightencoder, emb_dis_classifier, exp_dis_classifier)
    source_gex_eval = source_dataset.data.to(args.device)
    if args.split_dir is not None:
        sample_table = build_sample_table(
            target_unlabeled_data_df,
            target_unlabeled_info_df,
            tissue_column=args.tissue_column,
            info_id_column=args.info_id_column,
        )
        forget_indices, retain_indices = load_manifest_indices(sample_table, args.split_dir)
        forget_eval_loader = DataLoader(
            Subset(target_unlabeled_dataset, forget_indices),
            batch_size=EVALUATION_BATCH_SIZE,
            shuffle=False,
        )
        retain_eval_loader = DataLoader(
            Subset(target_unlabeled_dataset, retain_indices),
            batch_size=EVALUATION_BATCH_SIZE,
            shuffle=False,
        )

        def evaluate(loader):
            return evaluate_loader(
                loader,
                models,
                center_criterion,
                source_gex_eval,
                loss_a,
                loss_c,
                loss_b,
            )

        initial_forget = evaluate(forget_eval_loader)
        initial_retain = evaluate(retain_eval_loader)
    history.append(
        split_metrics_row(
            0,
            initial_forget,
            initial_retain,
        )
    )
    logger(
        f'[setup] model={model_name} target_samples={len(target_unlabeled_dataset)} '
        f'batch_size={args.batch_size} '
        f'seed={args.seed}'
    )
    logger(format_epoch_log(history[-1], args.epochs))

    # training
    for epoch in range(args.epochs):
        source_AE.train()
        target_weightencoder.train()
        emb_dis_classifier.train()
        exp_dis_classifier.train()

        train_losses = 0
        g_losses = 0
        t_losses = 0
        for target_gex, _, target_dis_label in target_unlabeled_dataloader:
            target_gex = target_gex.to(args.device)
            target_dis_label = target_dis_label.to(args.device)

            source_gex = source_dataset.data.to(args.device)
            source_dis_label = source_dataset.dis_label.to(args.device)
            source_z, source_recon = source_AE(source_gex)

            source_emb_dis_pred = emb_dis_classifier(source_z)
            source_exp_dis_pred = exp_dis_classifier(source_recon)
            
            # source loss
            Grecon_loss = autoencoder_criterion(source_recon, source_gex)
            Gcenter_loss = center_criterion(source_z, source_dis_label)
            Gclass_loss_emb = classifier_criterion(source_emb_dis_pred, source_dis_label)
            Gclass_loss_exp = classifier_criterion(source_exp_dis_pred, source_dis_label)
            G_losses = loss_a*Grecon_loss + loss_b*Gcenter_loss + loss_c*(Gclass_loss_emb + Gclass_loss_exp) 

            # target loss
            target_weights, target_latent, target_wgex, target_recon = target_weightencoder(target_gex, source_z, source_gex)
            target_emb_dis_pred = emb_dis_classifier(target_latent)
            target_exp_dis_pred = exp_dis_classifier(target_wgex)

            Trecon_loss = autoencoder_criterion(target_recon, target_gex)        
            Tcenter_loss = center_criterion(target_latent, target_dis_label)
            Tclass_loss_emb = classifier_criterion(target_emb_dis_pred, target_dis_label)
            Tclass_loss_exp = classifier_criterion(target_exp_dis_pred, target_dis_label)
            T_losses = loss_a*Trecon_loss + loss_b*Tcenter_loss + loss_c*(Tclass_loss_emb + Tclass_loss_exp)

            # update
            optimizer.zero_grad()
            total_losses = G_losses + T_losses
            total_losses.backward()
            optimizer.step()

            train_losses += total_losses.item()
            g_losses += G_losses.item()
            t_losses += T_losses.item()

        train_losses /= len(target_unlabeled_dataloader)
        g_losses /= len(target_unlabeled_dataloader)
        t_losses /= len(target_unlabeled_dataloader)
        forget_metrics = retain_metrics = None
        if forget_eval_loader is not None:
            forget_metrics = evaluate(forget_eval_loader)
            retain_metrics = evaluate(retain_eval_loader)
        history.append(
            split_metrics_row(
                epoch + 1,
                forget_metrics,
                retain_metrics,
                train_objective=train_losses,
                train_source=g_losses,
                train_target=t_losses,
            )
        )
        logger(format_epoch_log(history[-1], args.epochs))

    # save model
    checkpoint_path = output_dir / f'{model_name}.pt'
    torch.save({'epoch': epoch,
                'method': 'baseline',
                'completed_epochs': args.epochs,
                'optimizer_steps': args.epochs * len(target_unlabeled_dataloader),
                'source_AE': source_AE.state_dict(),
                'target_weightencoder': target_weightencoder.state_dict(),
                'emb_dis_classifier': emb_dis_classifier.state_dict(),
                'exp_dis_classifier': exp_dis_classifier.state_dict(),
                'center_criterion': center_criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': vars(args),
                }, checkpoint_path)
    history_path = output_dir / 'history.csv'
    curve_path = output_dir / 'loss_curve.png'
    write_history(history, history_path)
    if initial_forget is not None:
        plot_history(
            history,
            curve_path,
            args.loss_scale,
            epoch_label='baseline training epoch',
        )
        retain_curve_path = output_dir / 'retain_loss_curve.png'
        plot_retain_history(
            history,
            retain_curve_path,
            args.loss_scale,
            epoch_label='baseline training epoch',
        )
    else:
        retain_curve_path = None
    summary = {
        "method": "baseline",
        "objective": "minimize_source_loss_plus_target_loss",
        "objective_coefficients": {"source": 1.0, "target": 1.0},
        "sampling": "one_shuffled_full_target_pass_per_epoch",
        "batch_size": args.batch_size,
        "completed_epochs": args.epochs,
        "optimizer_steps": args.epochs * len(target_unlabeled_dataloader),
        "forget_samples_seen": (
            args.epochs * len(forget_indices) if forget_indices is not None else None
        ),
        "retain_samples_seen": (
            args.epochs * len(retain_indices) if retain_indices is not None else None
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "training_log": str((output_dir / 'training.log').resolve()),
        "history": str(history_path.resolve()),
        "loss_curve": str(curve_path.resolve()) if initial_forget is not None else None,
        "retain_loss_curve": (
            str(retain_curve_path.resolve()) if retain_curve_path is not None else None
        ),
        "split_dir": (
            str(Path(args.split_dir).resolve()) if args.split_dir is not None else None
        ),
        "config": vars(args),
        "initial_forget": initial_forget,
        "initial_retain": initial_retain,
        "final_forget": forget_metrics,
        "final_retain": retain_metrics,
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')
    logger(f'[done] checkpoint={checkpoint_path.resolve()} history={history_path.resolve()}')
   
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', default='../data/')
    parser.add_argument('--source', type=str, default='GDSC')
    parser.add_argument('--target', type=str, default='TCGA')
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--tissue-column', default='tissue_label')
    parser.add_argument('--info-id-column', default=None)
    parser.add_argument('--epochs', type=int, default=199)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--recon-weight', type=float, default=0.2)
    parser.add_argument('--center-weight', type=float, default=0.8)
    parser.add_argument('--class-weight', type=float, default=0.4)
    parser.add_argument(
        '--loss-scale', choices=('linear', 'log', 'symlog'), default='log'
    )
    
    args = parser.parse_args()

    train_aligner(args)
