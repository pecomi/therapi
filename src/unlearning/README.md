# THERAPI patient-level unlearning

이 디렉터리는 THERAPI aligner에서 TCGA 환자 정보를 제거하기 위한 코드를
담고 있다. 현재 구현한 unlearning 방법은 두 가지다.

- **NegGrad**: `gradient_ascent.py`
- **NegGrad+**: `retain_finetune.py`

삭제 재학습 기준선은 `retrain.py`로 생성한다. 모든 명령은 프로젝트 루트에서
실행하며, 세 방법은 반드시 동일한 split manifest를 사용해야 한다.

## 파일 역할

| 파일 | 역할 | 기본 실행에 필요한가 |
| --- | --- | --- |
| `make_forget_split.py` | patient-level forget/retain manifest 생성 CLI | 필요 |
| `split.py` | TCGA barcode 처리, 층화 분할, manifest 검증 | 필요 |
| `objective.py` | aligner forward, 원본 target loss, 전체-set 평가 | 필요 |
| `loss_history.py` | 공통 history row, console log, CSV와 curve 출력 | 필요 |
| `gradient_ascent.py` | NegGrad 학습 | NegGrad에 필요 |
| `retain_finetune.py` | NegGrad+ 학습. 파일명은 호환성을 위해 유지 | NegGrad+에 필요 |
| `retrain.py` | forget sample을 제외한 scratch retraining | 기준선 생성 시 필요 |
| `evaluate_representations.py` | baseline/unlearned/retrained 표현 평가 | 평가 시 선택 |
| `plot_unlearning_results.py` | 평가 결과와 loss history 통합 plot | 평가 시 선택 |

## 1. Forget/retain split

```bash
python src/unlearning/make_forget_split.py \
  --data-dir data \
  --forget-ratio 0.05 \
  --split-seed 0 \
  --output-dir splits/random_patient_5pct_seed0
```

같은 TCGA participant에 속한 sample은 항상 같은 assignment를 갖는다. 생성된
`samples.csv`를 baseline loss tracking, NegGrad, NegGrad+, retraining에 재사용한다.
학습 스크립트 안에서는 split을 다시 추출하지 않는다.

## 2. 공통 target loss

세 스크립트의 target loss는 원본 aligner와 동일하다.

```text
L_target = recon_weight * reconstruction_MSE
         + class_weight * (latent_tissue_CE + expression_tissue_CE)
         + center_weight * center_loss
```

기본값은 `recon_weight=0.2`, `class_weight=0.4`, `center_weight=0.8`이다.
`history.csv`의 `forget_task`와 `retain_task`는 매 epoch이 끝난 후 고정된 전체
forget/retain set에서 계산한 sample mean이다. 학습 mini-batch loss와 구분한다.

두 방법 모두 환자 정보가 흐르는 target-loss 경로를 갱신한다. NegGrad+는
삭제 대상이 아닌 GDSC도 retained source domain으로 매 step 함께 사용한다.

- 갱신: source encoder, target Q/K, latent tissue classifier, expression tissue classifier
- 고정: source decoder, target decoder, center anchor

이 범위는 현재 실험 의도에 따른 설정이다. GDSC source 성능 보존을 위한 추가
제약이나 갱신 범위 변경은 별도 실험으로 다룬다.

## 3. NegGrad

```bash
python src/unlearning/gradient_ascent.py \
  --data-dir data \
  --checkpoint run/baseline/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
  --split-dir splits/random_patient_5pct_seed0 \
  --output-dir run/neggrad_seed0 \
  --device cuda:0 \
  --original-train-seed 0 \
  --unlearn-seed 0 \
  --batch-size 64 \
  --lr 1e-5 \
  --epochs 30
```

최소화하는 목적함수는 다음과 같다.

```text
L_NegGrad = -L_forget
```

한 epoch은 shuffled forget loader 한 번이다. 따라서 forget sample이 `N_f`, batch
size가 `B`이면 epoch당 optimizer step 수는 `ceil(N_f / B)`이다. Retain set은
optimizer update에 사용하지 않고 full-set metric 계산에만 사용한다.

`--step-mode`와 `--forget-weight`는 제거했다. 구현은 mini-batch NegGrad 한
가지뿐이고, 목적 계수는 정의상 `-1`이므로 사용자 인자로 받을 필요가 없다.

## 4. NegGrad+

기존 파일명 `retain_finetune.py`는 유지하지만 실제 동작은 retain-only
fine-tuning이 아니라 NegGrad+다.

```bash
python src/unlearning/retain_finetune.py \
  --data-dir data \
  --checkpoint run/baseline/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
  --split-dir splits/random_patient_5pct_seed0 \
  --output-dir run/neggrad_plus_seed0 \
  --device cuda:0 \
  --original-train-seed 0 \
  --unlearn-seed 0 \
  --batch-size 128 \
  --lr 1e-3 \
  --beta 0.95 \
  --epochs 30
```

각 optimizer step에서 최소화하는 실제 목적함수는 다음과 같다.

```text
L_NegGrad+ = beta * (L_GDSC + L_retain) - (1 - beta) * L_forget
```

`beta`는 0 이상 1 이하이고 기본값은 `0.95`다. `history.csv`의
`evaluation_objective`에는 전체 GDSC와 고정 forget/retain set mean으로
재계산한 위 목적함수가 기록된다. `source_task`은 같은 GDSC full-set loss다.

### Sampling과 재현성

Retain loader가 epoch 길이를 정한다. 모든 retain batch를 한 번 사용하고, forget
loader가 먼저 끝나면 그 epoch에서 생성된 shuffled forget batch 순서를
처음부터 반복한다.

```text
for forget_batch, retain_batch in zip(cycle(forget_loader), retain_loader)
```

이는 매 epoch retain 크기만큼 새로운 forget sample을 replacement draw하는
방식이 아니다. 첫 forget traversal에서 만들어진 batch 순서를 같은 epoch 안에서
cycle한다. 다음 epoch에는 DataLoader의 generator state가 진행되어 새로운
shuffle 순서가 결정적으로 생성된다. 같은 데이터, PyTorch 환경, seed에서는
순서를 재현할 수 있다.

두 batch의 loss는 각각 batch mean으로 계산된 후 `beta`로 결합한다. 마지막
partial batch의 sample 수가 서로 달라도 forget/retain 항의 계수는 `1-beta`와
`beta`로 유지된다.

GDSC source 673개는 삭제 대상이 아니므로 매 paired forget/retain step에서
전체를 한 번 사용한다. 이는 원본 `train_aligner.py`가 매 TCGA batch마다 GDSC
전체 source loss를 계산한 방식과 같다. source loss는 source encoder와 shared
tissue classifier를 보존하는 gradient를 제공하며, target Q/K에는 직접 gradient를
보내지 않는다.

`summary.json`에는 다음 run 전체 처리량을 기록한다.

- 완료 epoch 수와 전체 optimizer step 수
- forget/retain/source sample 노출 수
- forget set 크기로 나눈 `effective_forget_passes`

NegGrad와 NegGrad+의 같은 epoch 수는 같은 연산량을 뜻하지 않는다. 예를 들어
forget 401개, retain 7,641개, batch size 128이면 NegGrad는 epoch당 4 step,
NegGrad+는 epoch당 60 step이다. 결과를 해석할 때 epoch뿐 아니라 step 수와 sample
노출량을 함께 확인해야 한다.

## 5. Deletion retraining

```bash
python src/unlearning/retrain.py \
  --data-dir data \
  --split-dir splits/random_patient_5pct_seed0 \
  --output-dir run/retrain_seed0 \
  --device cuda:0 \
  --seed 0
```

Baseline checkpoint에서 fine-tune하지 않는다. 무작위 초기화부터 시작하여 GDSC
전체와 retain TCGA만으로 원본 `source loss + target loss` 학습을 반복한다.

## 6. 공통 로그와 산출물

학습 방법의 `history.csv`는 가능한 경우 다음 공통 필드를 사용한다.

| 필드 | 의미 |
| --- | --- |
| `epoch` | 0은 update 전 상태, 1 이상은 완료된 epoch |
| `train_objective` | baseline/retrain의 mini-batch training loss 평균 |
| `evaluation_objective` | 결과 비교에 쓰는 full-set mean 방법별 목적함수 |
| `source_task` | NegGrad+에서 평가한 전체 GDSC source loss |
| `forget_*`, `retain_*` | 고정 split에서 계산한 full-set target metrics |
| `gradient_norm` | optimizer update 전 step gradient norm의 epoch 평균 |

`history.csv`의 paired metric column은 `forget_task`, `retain_task`,
`forget_recon`, `retain_recon`, `forget_emb_class`, `retain_emb_class` 순서처럼
동일 metric의 forget/retain 값을 나란히 저장한다.

Baseline을 split 없이 학습하면 forget/retain 관련 칼럼은 비어 있고 loss curve는
생성하지 않는다. Split을 주면 학습 데이터에는 영향을 주지 않고 metric만
추가한다.

## 7. 사후 평가: target unlearning과 GDSC 보존

`evaluate_representations.py`는 동일한 baseline, unlearned, retrained
checkpoint를 사용해 TCGA forget/retain 평가와 GDSC source 보존 평가를 함께
수행한다.

```bash
python src/unlearning/evaluate_representations.py \
  --data-dir data \
  --baseline-checkpoint run/baseline/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
  --unlearned-checkpoint run/neggrad_plus_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
  --retrained-checkpoint run/retrain_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
  --split-dir splits/random_patient_5pct_seed0 \
  --output-dir run/evaluation_seed0 \
  --device cuda:0
```

GDSC는 환자 split과 무관하므로 전체 cell line을 한 번 평가한다.
`source_metrics.csv`에는 원래 source objective의 항별 평균(`task`, `recon`,
`emb_class`, `exp_class`, `center`)과 두 tissue classifier accuracy를 기록한다.
`source_representation_similarity.csv`에는 같은 GDSC cell line이 checkpoint
사이에서 얼마나 바뀌었는지 기록한다.

`retraining_consistency_per_sample.csv`와
`retraining_consistency_per_patient.csv`는 baseline/retrained 및
unlearned/retrained의 직접 출력 차이를 각각 sample·patient 단위로 기록한다.
`retraining_consistency_summary.csv`에는 forget/retain 및 비교쌍별 mean/median을
정리한다.

- `weighted_expression_mse`: attention-weighted GDSC expression의 sample별 MSE

두 CSV의 비교쌍은 `baseline_vs_retrained`, `unlearned_vs_retrained`다.
같은 forget/retain group에서 후자의 거리가 전자보다 작으면 unlearned output이
baseline보다 deletion-retraining output에 가까워진 것으로 해석한다. latent는
좌표계 재배치 영향을 피하기 위해 기존 CKA/Frechet 결과로 보조 비교한다.

- `linear_cka`, `frechet_latent_distance`: 분포·기하 수준의 비교
- `mean_paired_cosine_similarity`, `normalized_representation_change`: 동일 cell
  line의 직접적인 paired 변화량

unlearning checkpoint가 배포용 새 aligner checkpoint다. GDSC 원본을 다시
학습하거나 GDSC drug-response predictor를 재학습할 필요는 없다. 다만 환자
expression을 새 checkpoint로 다시 align하고, 기존 파일을 덮어쓰지 않는 별도
출력 경로에 새 CSG2A embedding을 만들어야 한다.

프로젝트 루트의 `unlearning_pipeline.sh`는 이 연결 과정을 실행한다. 기본 실행은
split, 새 seed baseline, NegGrad+, deletion retraining, representation 평가이며,
`embed` stage를 추가하면 NegGrad+ checkpoint로 TCGA CSG2A embedding을 새 run
경로에 생성한다.
`predictor_test` stage는 `PREDICTOR_CKPT_DIR`로 기존 GDSC predictor의 10-fold
checkpoint 경로를 지정했을 때만 실행한다. predictor는 재학습하지 않는다.

```bash
PREDICTOR_CKPT_DIR=run/predictor_seed0/ckpts \
STAGES="split baseline neggrad_plus retrain evaluate embed predictor_test" \
RUN_NAME=seed1 ORIGINAL_TRAIN_SEED=1 UNLEARN_SEED=1 SPLIT_SEED=1 BETA=0.9 \
./unlearning_pipeline.sh
```

`DEPLOY_METHODS="baseline neggrad_plus retrain"`처럼 지정하면 세 checkpoint의
TCGA embedding과 predictor test 결과를 각각 생성해 비교할 수 있다. 모든 command
출력은 `run/<RUN_NAME>/pipeline.log`에 저장되고, 각 학습 단계의 `training.log`도
각자의 `ckpts/`에 별도로 남는다.

### Pipeline stage와 hyperparameter 실험

`STAGES`는 공백으로 구분한 실행 단계다.

| stage | 수행 내용 | 언제 쓰는가 |
| --- | --- | --- |
| `split` | manifest가 없을 때 patient-level split 생성, 있으면 재사용 | 새 삭제 집합 생성 시 |
| `baseline` | `ORIGINAL_TRAIN_SEED`으로 원본 aligner 학습 | 새 baseline seed 실험 시 |
| `neggrad` | NegGrad 실행 | NegGrad 비교 시 |
| `neggrad_plus` | NegGrad+ 실행 | 기본 unlearning 실험 |
| `retrain` | retain-only deletion retraining | 새 reference checkpoint가 필요할 때 |
| `evaluate` | baseline/unlearned/retrained representation 평가 | 후보 설정 평가 시 |
| `embed` | `DEPLOY_METHODS`별 TCGA CSG2A embedding 재생성 | 최종 checkpoint 선택 후 |
| `predictor_test` | 기존 predictor checkpoint로 TCGA 추론 | 최종 predictor 결과 산출 시 |

기본값은 `split baseline neggrad_plus retrain evaluate`다. 즉 새 `RUN_NAME`으로
실행하면 새 baseline seed부터 결과를 만든다. 출력은
`run/<RUN_NAME>/baseline`, `neggrad_plus`, `retrain`, `evaluation`에 분리된다.
같은 `RUN_NAME`이 이미 있으면 기본적으로 `_2` suffix를 붙인다. `RESUME=1`은
이미 생성한 run에 `embed`나 `predictor_test`처럼 뒤 단계만 추가할 때만 사용한다.

hyperparameter 후보 탐색에서는 baseline과 split을 고정하고 `neggrad_plus`만
실행한다. 다음 예시는 beta와 learning rate만 바꾼 후보 하나를 만든다.

```bash
BASELINE_CHECKPOINT=run/baseline_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
SPLIT_DIR=splits/random_patient_5pct_seed0 \
RUN_NAME=ngp_beta090_lr3e4_seed0 \
STAGES="neggrad_plus" \
ORIGINAL_TRAIN_SEED=0 UNLEARN_SEED=0 \
BETA=0.90 NEGGRAD_PLUS_LR=3e-4 \
NEGGRAD_PLUS_EPOCHS=30 NEGGRAD_PLUS_BATCH_SIZE=128 \
./unlearning_pipeline.sh
```

이 단계에서는 `history.csv`, loss curve, `training.log`를 보고 후보를 고른다.
representation 평가까지 반복하려면 `evaluate`는 retrained checkpoint를 요구한다.
같은 split의 retrained reference는 한 번만 만든 뒤 `RETRAIN_CHECKPOINT`로 재사용할
수 있다.

```bash
BASELINE_CHECKPOINT=run/baseline_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
RETRAIN_CHECKPOINT=run/retrain_retain_5pct_seed0/ckpts/THERAPI_aligner_GDSC_TCGA.pt \
SPLIT_DIR=splits/random_patient_5pct_seed0 \
RUN_NAME=ngp_beta090_lr3e4_seed0 \
STAGES="neggrad_plus evaluate" \
BETA=0.90 NEGGRAD_PLUS_LR=3e-4 \
./unlearning_pipeline.sh
```

조절 가능한 주요 변수는 다음과 같다.

| 목적 | 변수 |
| --- | --- |
| NegGrad+ | `BETA`, `NEGGRAD_PLUS_LR`, `NEGGRAD_PLUS_EPOCHS`, `NEGGRAD_PLUS_BATCH_SIZE` |
| NegGrad | `NEGGRAD_LR`, `NEGGRAD_EPOCHS`, `NEGGRAD_BATCH_SIZE` |
| baseline | `BASELINE_LR`, `BASELINE_EPOCHS`, `BASELINE_BATCH_SIZE` |
| retraining | `RETRAIN_LR`, `RETRAIN_EPOCHS`, `RETRAIN_BATCH_SIZE` |
| 공통 모델/목적함수 | `LATENT_DIM`, `RECON_WEIGHT`, `CLASS_WEIGHT`, `CENTER_WEIGHT` |
| split | `SPLIT_DIR`, 새 manifest 생성 시 `FORGET_RATIO`, `SPLIT_SEED` |
| deployment | `DEPLOY_METHODS`, `PREDICTOR_CKPT_DIR`, `CSG2A_CKPT`, `DOSE`, `TIME` |

`FORGET_RATIO`를 기본 0.05와 다르게 쓸 때는 ratio가 반영된 새 `SPLIT_DIR`을 직접
지정해야 한다. 기존 manifest 경로가 있으면 `split` stage는 seed나 ratio와 무관하게
그 파일을 그대로 재사용한다.

각 run의 `ckpts/`에는 다음 파일이 생성된다.

```text
ckpts/
├── THERAPI_aligner_GDSC_TCGA.pt
├── training.log
├── history.csv
├── loss_curve.png
├── retain_loss_curve.png
└── summary.json
```

`loss_curve.png`는 forget component, `retain_loss_curve.png`는 retain component를
각각 표시한다. 두 파일 모두 좌측에는 forget/retain 전체 task loss를 함께 둔다.
Baseline을 split 없이 실행한 경우에만 두 curve가 없다. `summary.json`은
공통적으로 method, objective, completed epochs, optimizer steps,
checkpoint/log/history 경로, config, 초기/최종 forget·retain metrics를 기록한다.
`training.log`에는 해당 실행의 setup, epoch, done 콘솔 행을 원문 그대로 기록한다.
