# THERAPI TCGA patient-level unlearning

이 문서는 같은 TCGA patient split로 다음 세 실험을 실행하는 방법을
정리한다.

1. **Baseline**: 전체 TCGA로 학습한 원본 aligner
2. **Unlearned**: baseline에 forget-only gradient ascent를 여러 epoch 적용
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
- source decoder와 center anchor는 고정한다. target loss 경로에 있는 source
  encoder, target Q/K/decoder, 두 tissue classifier는 업데이트한다.
- 기본 `--step-mode full`은 모든 forget micro-batch의 sample-weighted 평균
  gradient를 누적해 epoch당 한 번 update한다. `--step-mode mini`는 일반
  train loop처럼 mini-batch마다 update한다.
- gradient clipping은 기본적으로 끈다(`--max-grad-norm 0`). 필요할 때만 양수로 켠다.
- forget 전체의 평가 loss가 plateau에 도달하면 자동 종료한다.

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
python src/unlearning/gradient_ascent.py \
  --data_dir data \
  --checkpoint "run/$BASELINE_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt" \
  --split-dir "splits/$SPLIT_NAME" \
  --output-dir "run/$UNLEARN_RUN" \
  --device "$DEVICE" \
  --original-train-seed 0 \
  --unlearn-seed 0 \
  --step-mode full \
  --lr 1e-5 \
  --epochs 100 \
  --min-epochs 10 \
  --patience 5 \
  --plateau-rtol 1e-3
```

출력:

```text
run/unlearn_5pct_seed0/ckpts/THERAPI_aligner_unlearned.pt
run/unlearn_5pct_seed0/ckpts/history.csv
run/unlearn_5pct_seed0/ckpts/summary.json
run/unlearn_5pct_seed0/ckpts/loss_curve.png
```

### Batch 선택 근거

후보는 (1) forget sample마다 step, (2) 일반 mini-batch마다 step, (3) forget
전체 평균 gradient로 epoch마다 한 step이다. (1)은 잡음이 매우 크다. 기본값인
`--step-mode full`은 (3)이며, forget 전체를 GPU에 한 번에 올리지 않고
`--batch-size 64` micro-batch로 나눠 gradient를 누적한다. 이 모드에서는
`batch-size`가 주로 메모리/속도만 바꾸고 effective batch는 forget 전체다.

일반 train loop와 같은 stochastic ascent가 필요하면 (2)를 선택한다.

```bash
python src/unlearning/gradient_ascent.py \
  ... \
  --step-mode mini \
  --batch-size 128
```

이 모드에서는 각 mini-batch 직후 `optimizer.step()`을 실행한다. 따라서 batch
크기와 shuffle 순서가 Adam의 경로 및 epoch당 update 수를 바꾼다. 종료용
forget/retain loss는 두 모드 모두 epoch이 끝난 뒤 shuffle 없이 전체 set에서
다시 계산한다. mini 모드의 곡선이 진동하면 `--patience 10`처럼 확인 구간을
늘리는 것을 권장한다.

### Loss와 종료 epoch

학습과 평가에는 원본 aligner의 target objective를 그대로 쓴다.

```text
L = 0.2 * reconstruction MSE
  + 0.4 * (latent tissue CE + expression tissue CE)
  + 0.8 * center loss
```

gene expression은 연속값이고 원본 학습도 reconstruction MSE를 사용하므로
MSE를 유지한다. CE는 tissue 판별 정보, center loss는 같은 tissue latent의
응집도를 측정한다. 이 세 항의 가중합을 그대로 사용해야 baseline, retrained,
unlearned가 같은 기준으로 비교되고 gradient ascent가 실제 원본 학습 목적의
역방향이 된다.

`history.csv`와 `loss_curve.png`의 값은 noisy mini-batch loss가 아니라 매
epoch update 후 forget/retain 전체에서 다시 계산한 sample-weighted 평균이다.
epoch 0은 baseline이다. 다음 조건이 모두 만족되면 종료한다.

```text
epoch >= min_epochs
abs(L_t - L_{t-1}) / max(abs(L_{t-1}), 1e-12) <= plateau_rtol
위 조건이 patience epoch 연속 유지
```

즉 loss가 상승한 뒤 상대 변화가 충분히 작게 유지되는 첫 plateau의 끝을
선택한다. 점선이 선택 epoch이다. plateau가 없으면 `--epochs`가 선택된다.
실험 전에는 `--epochs 100 --patience 0`으로 곡선을 먼저 얻고, 곡선의 noise
크기에 맞춰 `plateau-rtol`을 정한 뒤 자동 종료를 켜는 방식도 권장한다.
retain loss가 급격히 함께 증가한다면 forget loss의 plateau만으로 좋은
unlearning이라고 해석하면 안 된다.

gradient가 실제로 발산해 non-finite가 되면 즉시 실패시킨다. clipping 비교가
필요한 경우에만 예를 들어 `--max-grad-norm 1.0`을 지정한다.

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

같은 `TCGA_unlabeled`의 forget/retain 환자를 baseline, unlearned,
retain-only retrained aligner 세 개에 모두 통과시킨다. 환자별 latent,
attention, reconstruction, center distance를 비교하고 unlearning이 deletion
retraining에 가까워졌는지 계산한다.

```bash
python src/unlearning/evaluate_representations.py \
  --data_dir data \
  --baseline-checkpoint "run/$BASELINE_RUN/ckpts/THERAPI_aligner_GDSC_TCGA.pt" \
  --unlearned-checkpoint "run/$UNLEARN_RUN/ckpts/THERAPI_aligner_unlearned.pt" \
  --retrained-checkpoint "run/$RETRAIN_RUN/ckpts/THERAPI_aligner_retrained_retain_only.pt" \
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
loss_metrics.csv
representation_similarity.csv
```

`loss_metrics.csv`는 baseline/unlearned/retrained 각각에 대해 forget/retain의
평균 `task`, reconstruction MSE, 두 CE, center loss를 같은 함수로 계산한다.
`unit=sample`은 모든 sample에 같은 가중치를 주고, `unit=patient`는 환자 내
sample을 먼저 평균내어 sample 수가 많은 환자의 영향이 커지지 않게 한다.

`representation_similarity.csv`에는 latent에 대한 다음 두 지표가 sample과
patient 수준으로 기록된다.

- **Linear CKA**: 동일 sample의 representation geometry 유사도다. 1에
  가까울수록 두 모델이 sample 사이 관계를 비슷하게 보존한다. 좌표의 직교
  회전과 isotropic scale에 비교적 강하므로 retraining으로 latent 좌표계가
  바뀌어도 직접 RMSE보다 안정적이다.
- **Fréchet latent distance**: latent를 Gaussian으로 근사해 평균과 공분산의
  차이를 측정한 FID/FCD 방식의 거리다. 0에 가까울수록 분포가 비슷하다.
  이미지 feature가 아니므로 엄밀히는 FID라 부르지 않고
  `frechet_latent_distance`로 기록한다. 특히 FCD(Fréchet ChemNet Distance)는
  생성 분자의 ChemNet feature용 지표인데 이 aligner는 분자를 생성하지 않으므로
  그대로 적용하는 것은 부적절하다. Gaussian 가정과 표본 수에 민감하므로 CKA
  및 task loss와 함께 해석한다.

핵심 비교는 forget에서 `unlearned_vs_retrained`의 CKA가 커지고 Fréchet
distance가 작아지는지, retain에서 `baseline_vs_unlearned`가 잘 보존되는지다.

터미널에는 forget/retain의 환자 단위 평균과 다음 선택성 비율이 출력된다.

```text
forget 환자 평균 latent RMSE / retain 환자 평균 latent RMSE
```

이 비율이 1보다 클수록 forget 환자 representation이 상대적으로 더 많이
변했다. 단, 비율만 보지 말고 retain의 절대 변화량과 center distance,
attention, reconstruction 변화도 함께 확인한다.

세 모델 비교에서 핵심 열은 다음과 같다.

```text
*_baseline_unlearned
  원본과 unlearned 사이 거리

*_baseline_retrained
  원본과 삭제 후 재학습 모델 사이 거리

*_unlearned_retrained
  unlearned와 삭제 후 재학습 모델 사이 거리

*_improvement_to_retrained
  baseline-retrained 거리 - unlearned-retrained 거리
```

`*_improvement_to_retrained`이 양수이면 unlearning이 해당 지표에서 원본보다
retain-only retrained 모델에 가까워졌다는 뜻이다. 이 값을 forget과 retain에서
각각 확인한다.

```text
forget:
  improvement_to_retrained > 0 이 주요 목표

retain:
  baseline_unlearned 거리가 작아야 함
  retain utility가 유지되는지도 함께 확인
```

Raw latent는 retraining 과정에서 좌표계가 달라질 수 있으므로 참고 지표로
사용한다. center distance, tissue 정답 확률, attention 분포,
reconstruction output처럼 기능적 출력의 `improvement_to_retrained`을 더
중요하게 해석한다.
