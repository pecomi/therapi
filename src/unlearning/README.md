# THERAPI TCGA patient-level unlearning

이 문서는 같은 TCGA patient split로 다음 세 실험을 실행하는 방법을
정리한다.

1. **Baseline**: 전체 TCGA로 학습한 원본 aligner
2. **Unlearned**: baseline에 forget-only gradient ascent를 한 번 적용
3. **Retrained**: forget 환자를 처음부터 제외하고 다시 학습한 기준 모델

모든 명령은 THERAPI 프로젝트 루트에서 실행한다.

```bash
cd /home/young/unlearning/therapi/theraphi_add_embedding
source unlearn/bin/activate
```

실제 서버의 프로젝트 경로나 venv 이름이 다르면 위 두 경로만 바꾼다.

## 0. 공통 설정과 주의점

아래 예시는 다음 이름을 사용한다.

```bash
BASELINE_RUN=baseline_seed0_3
SPLIT_NAME=random_patient_5pct_seed0
UNLEARN_RUN=unlearn_5pct_seed0
RETRAIN_RUN=retrain_retain_5pct_seed0
DEVICE=cuda:4
```

세 실험은 반드시 같은 `splits/$SPLIT_NAME/samples.csv`를 사용해야 한다.
split을 학습 스크립트 안에서 다시 추출하면 실험 간 forget 환자가 달라질
수 있으므로 허용하지 않는다.

현재 unlearning 설정은 다음과 같다.

- retain 데이터는 unlearning backpropagation에 사용하지 않는다.
- center loss는 사용하지만 center parameter 자체는 고정한다.
- source/target decoder parameter는 고정한다.
- source encoder, target Q/K, 두 tissue classifier는 업데이트한다.
- 모든 forget mini-batch의 평균 gradient를 누적한 뒤
  `optimizer.step()`을 정확히 한 번 실행한다.
- 초기 폭발 여부를 보기 위해 gradient clipping은 주석 처리되어 있다.

## 1. Patient-level forget/retain split 생성

TCGA participant의 5%를 tissue-stratified 방식으로 forget에 배정한다.
동일 participant의 여러 sample은 항상 같은 쪽에 들어간다.

```bash
python src/unlearning/make_forget_split.py \
  --data_dir data \
  --forget-ratio 0.05 \
  --split-seed 0 \
  --output-dir "splits/$SPLIT_NAME"
```

주요 출력은 다음과 같다.

```text
splits/random_patient_5pct_seed0/
├── patients.csv
├── samples.csv
├── forget_patients.csv
├── retain_patients.csv
├── forget_samples.csv
├── retain_samples.csv
└── metadata.json
```

`make_forget_split.py`는 실행용 CLI이고, 실제 barcode 처리·분할·검증 로직은
`split.py`에 있다.

## 2. Baseline 준비

이미 다음 checkpoint와 downstream 결과가 정상적으로 생성되었다면 baseline을
다시 돌릴 필요가 없다.

```text
run/baseline_seed0_2/ckpts/THERAPI_aligner_GDSC_TCGA.pt
```

처음부터 baseline 전체 pipeline을 실행해야 한다면:

```bash
RUN_NAME="$BASELINE_RUN" \
DEVICE="$DEVICE" \
SEED=0 \
STAGES="aligner embed predictor test" \
bash pipline.sh
```

기존 run을 이어서 누락된 stage만 실행할 때는 `RESUME=1`을 반드시 지정한다.

```bash
RUN_NAME="$BASELINE_RUN" \
RESUME=1 \
DEVICE="$DEVICE" \
SEED=0 \
STAGES="embed predictor test" \
bash pipline.sh
```

## 3. Forget-only gradient-ascent unlearning

원본 baseline checkpoint에서 시작한다.

```bash
python src/unlearning/train.py \
  --data_dir data \
  --checkpoint "run/$BASELINE_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt" \
  --split-dir "splits/$SPLIT_NAME" \
  --output-dir "run/$UNLEARN_RUN" \
  --device "$DEVICE" \
  --original-train-seed 0 \
  --unlearn-seed 0 \
  --lr 1e-5
```

출력:

```text
run/unlearn_5pct_seed0/ckpts/THERAPI_aligner_unlearned.pt
run/unlearn_5pct_seed0/ckpts/history.csv
run/unlearn_5pct_seed0/ckpts/summary.json
```

gradient clipping은 `src/unlearning/train.py`에서 다음 줄이 주석 처리된
상태다.

```python
# torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
```

폭발 여부를 확인한 뒤 clipping 실험을 추가할 때만 주석을 해제한다.

## 4. Retain-only deletion retraining

이 실험은 baseline checkpoint를 fine-tuning하지 않는다. 모델을 처음부터
초기화하고 GDSC 전체와 retain TCGA만 사용해 원본 aligner와 같은 방식으로
199 epochs 학습한다.

```bash
python src/unlearning/retrain.py \
  --data_dir data \
  --split-dir "splits/$SPLIT_NAME" \
  --output-dir "run/$RETRAIN_RUN" \
  --device "$DEVICE" \
  --seed 0
```

출력:

```text
run/retrain_retain_5pct_seed0/ckpts/THERAPI_aligner_retrained_retain_only.pt
run/retrain_retain_5pct_seed0/ckpts/history.csv
```

이 모델은 unlearning 결과가 근접해야 하는 deletion-retraining reference다.

두 학습 스크립트의 `--output-dir`에는 run 디렉터리 또는 그 아래의 `ckpts`
디렉터리를 줄 수 있다. `run/<RUN_NAME>`을 주면 스크립트가 `ckpts`를 자동으로
추가하며, `run/<RUN_NAME>/ckpts`를 직접 주어도 중복으로 추가하지 않는다.

## 5. Pipeline이 찾는 aligner 이름 연결

`pipline.sh`는 각 run에서 아래 고정 이름을 찾는다.

```text
run/<RUN_NAME>/ckpts/THERAPI_aligner_GDSC_TCGA.pt
```

따라서 unlearned와 retrained checkpoint를 각각 이 이름으로 복사한다. 원본
파일은 보존된다.

```bash
cp "run/$UNLEARN_RUN/ckpts/THERAPI_aligner_unlearned.pt" \
   "run/$UNLEARN_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt"

cp "run/$RETRAIN_RUN/ckpts/THERAPI_aligner_retrained_retain_only.pt" \
   "run/$RETRAIN_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt"
```

## 6. Predictor를 동일하게 유지

`train_predictor.py`는 GDSC로 학습한다. 이번 비교에서는 predictor를 실험마다
다시 학습하지 않고 baseline의 동일한 10-fold predictor checkpoint를 복사해
사용하는 것을 권장한다. 그래야 결과 차이를 TCGA aligner/embedding 변화로
해석할 수 있다.

```bash
cp run/$BASELINE_RUN/ckpts/THERAPI_predictor_CV*.pt \
   "run/$UNLEARN_RUN/ckpts/"

cp run/$BASELINE_RUN/ckpts/THERAPI_predictor_CV*.pt \
   "run/$RETRAIN_RUN/ckpts/"
```

복사 여부를 확인한다.

```bash
ls "run/$UNLEARN_RUN/ckpts/"THERAPI_predictor_CV*.pt
ls "run/$RETRAIN_RUN/ckpts/"THERAPI_predictor_CV*.pt
```

## 7. 실험별 TCGA embedding 생성 및 test

unlearning과 retraining은 TCGA basal expression alignment를 바꾸므로 TCGA
perturbation embedding은 반드시 실험별로 다시 생성한다. 반면 GDSC embedding과
predictor는 동일하게 유지한다.

### 7-1. Unlearned

```bash
RUN_NAME="$UNLEARN_RUN" \
RESUME=1 \
DEVICE="$DEVICE" \
SEED=0 \
STAGES="embed test" \
bash pipline.sh
```

### 7-2. Retain-only retrained

```bash
RUN_NAME="$RETRAIN_RUN" \
RESUME=1 \
DEVICE="$DEVICE" \
SEED=0 \
STAGES="embed test" \
bash pipline.sh
```

여기서 `STAGES`에 `aligner`를 넣으면 안 된다. 넣으면 준비한 unlearned 또는
retrained checkpoint 대신 baseline aligner 학습이 다시 실행된다. 동일 predictor를
사용하려면 `predictor`도 넣지 않는다.

embedding stage는 각 run에 다음 파일을 만든다.

```text
run/<RUN_NAME>/data/TCGA/TCGA_perturbation_float16.npy
run/<RUN_NAME>/data/TCGA/TCGA_perturbation_compound_float16.npy
```

test 결과는 다음 위치에 저장된다.

```text
run/<RUN_NAME>/output/THERAPI_test_TCGA.csv
```

## 8. 최종 비교 대상

```text
run/baseline_seed0_2/output/THERAPI_test_TCGA.csv
run/unlearn_5pct_seed0/output/THERAPI_test_TCGA.csv
run/retrain_retain_5pct_seed0/output/THERAPI_test_TCGA.csv
```

전체 TCGA metric뿐 아니라 `splits/$SPLIT_NAME/samples.csv`의 `assignment`를
이용해 forget과 retain을 분리해서 평가해야 한다.

기대하는 비교 방향은 다음과 같다.

```text
forget set:
  Unlearned ≈ Retain-only retrained
  Unlearned ≠ Baseline

retain set:
  Unlearned ≈ Baseline
```

권장 보고 항목:

- 전체/forget/retain 각각의 AUC와 AUPRC
- baseline 대비 forget 예측 변화량
- baseline 대비 retain 예측 변화량
- unlearned와 retain-only retrained 예측 사이의 거리
- unlearning 전후 forget/retain aligner loss
- gradient group norm 및 non-finite 여부

현재 `test_TCGA.py`의 기본 metric은 전체 TCGA를 대상으로 한다. forget/retain
분리 metric을 얻으려면 test prediction과 `samples.csv`를 patient ID 기준으로
연결하는 별도 평가 단계가 필요하다.

## 9. 실행 전 최종 체크리스트

```bash
test -f "run/$BASELINE_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt"
test -f "splits/$SPLIT_NAME/samples.csv"
test -f src/embedding/CSG2A_LINCSpretrained_Landmark.pt
test -f src/embedding/CSG2A/data/STRING_edges.csv
python -c 'import torch, pandas, rdkit, sklearn; print(torch.cuda.is_available())'
```

각 pipeline 로그의 `[align]` 줄에서 실제로 사용된 checkpoint가 해당 run의
`THERAPI_aligner_GDSC_TCGA.pt`인지 반드시 확인한다.

## 10. Forget/retain representation 변화 평가

같은 `TCGA_unlabeled` 환자를 baseline과 unlearned aligner에 통과시켜 환자별
latent, attention, reconstruction, center distance 변화를 계산한다.

```bash
python src/unlearning/evaluate_representations.py \
  --data_dir data \
  --baseline-checkpoint "run/$BASELINE_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt" \
  --unlearned-checkpoint "run/$UNLEARN_RUN/ckpts/THERAPI_aligner_unlearned.pt" \
  --split-dir "splits/$SPLIT_NAME" \
  --output-dir "run/$UNLEARN_RUN/evaluation/representations" \
  --device "$DEVICE" \
  --original-train-seed 0
```

출력 파일:

```text
representation_change_per_sample.csv
representation_change_per_patient.csv
representation_change_group_summary.csv
representation_change_summary.json
```

터미널에는 forget/retain의 환자 단위 평균과 다음 선택성 비율이 출력된다.

```text
forget 환자 평균 latent RMSE / retain 환자 평균 latent RMSE
```

이 비율이 1보다 클수록 forget 환자 representation이 상대적으로 더 많이
변했다. 단, 비율만 보지 말고 retain의 절대 변화량과 center distance,
attention, reconstruction 변화도 함께 확인한다.
