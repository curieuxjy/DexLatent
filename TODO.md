# TODO: Allegro Hand 추가

## 1. Combined URDF 생성 (`Assets/xarm7_allegro/`)
- [x] xarm7 arm + allegro hand를 합친 URDF 생성 (right)
- [x] xarm7 arm + allegro hand를 합친 URDF 생성 (left)
- [x] joint_eef origin (xyz, rpy) 마운팅 파라미터 결정 → xyz="0 0 0.1" rpy="0 0 0"
- [x] mesh 경로가 올바른지 확인
- [x] URDF 파싱 테스트 (urdf_parser_py로 로드 가능한지)
- [x] FK 동작 확인 (HandKinematicsModel로 forward kinematics 정상 동작)

## 2. HAND_CONFIGS 등록 (`HandLatent/kinematics.py`)
- [x] xarm7_allegro_right 설정 추가
- [x] xarm7_allegro_left 설정 추가
- [x] tip_links 순서 확인 (thumb first: link_15.0_tip, link_3.0_tip, link_7.0_tip, link_11.0_tip)
- [x] FK 동작 확인 (MultiHandDifferentiableFK 정상 로드, 23 DOF, 4 tips)

## 3. 학습 코드 수정 (`HandLatent/train.py`)
- [x] hand_names에 xarm7_allegro_right 추가

## 4. 추론 코드 수정 (`HandLatent/infer.py`)
- [x] trainer_hands에 xarm7_allegro 추가
- [x] target_hands에 xarm7_allegro 추가

## 5. 학습 및 검증
- [x] 5개 핸드 포함하여 재학습
- [x] 추론 시각화로 retargeting 품질 확인

## 6. 수치 평가 스크립트 (`HandLatent/evaluate.py`)
- [x] 평가 스크립트 작성 및 테스트
- [x] 세 가지 메트릭 구현:
  - **Self-reconstruction error**: encode→같은 핸드로 decode 시 hand qpos MSE
  - **Cross-embodiment pinch loss**: source→target 간 pinch 거리/방향 오차 (exponential weight 적용)
  - **Fingertip position error**: full pipeline (encode→decode+IK) 후 공유 fingertip L2 거리 (m)
- [x] 실행 방법:
  ```bash
  # demo 데이터로 평가 (source: inspire)
  uv run -m HandLatent.evaluate --ckpt <checkpoint_path> --side right

  # random sample로 평가
  uv run -m HandLatent.evaluate --ckpt <checkpoint_path> --side right --num_samples 500
  ```
- [x] 참고: self-reconstruction은 demo 데이터가 있는 source hand만 demo로 평가하고, 나머지 핸드는 random qpos로 평가함

## 7. 학습 로깅 (`HandLatent/train.py`)

학습 시작 시 wandb에 자동 연결되어 매 step마다 아래 메트릭을 로깅합니다.

### 기본 loss
| key | 설명 | 이상적 방향 | 비고 |
|-----|------|------------|------|
| `loss_total` | 전체 loss (rec + pinch + kl) | ↓ 낮을수록 좋음 | 모든 loss의 가중합. 단조 감소가 이상적이나 kl과 pinch 간 trade-off로 초반에 진동 가능 |
| `loss_rec_total` | reconstruction loss (weighted) | ↓ 낮을수록 좋음 | 학습 초반에 가장 빠르게 감소해야 함. 0에 수렴하면 autoencoder가 정상 동작 중 |
| `loss_rec_hand` | hand qpos reconstruction MSE (weighted) | ↓ 낮을수록 좋음 | ~0.01 이하면 양호. 특정 핸드만 높으면 해당 핸드의 DOF가 많거나 학습이 부족한 것 |
| `loss_pinch_dis` | cross-embodiment pinch distance loss | ↓ 낮을수록 좋음 | 서로 다른 핸드 간 pinch 위치 일치도. 0에 가까울수록 cross-embodiment 전이가 잘 됨 |
| `loss_pinch_dir` | cross-embodiment pinch direction loss | ↓ 낮을수록 좋음 | pinch 방향 일치도. distance보다 천천히 수렴하는 경향 |
| `loss_kl` | KL divergence loss | ↕ 적절한 범위 유지 | 너무 낮으면(→0) posterior collapse (latent를 활용 못함). 너무 높으면 latent가 정규화 안 됨. 안정적으로 수렴하는 것이 중요 |
| `exp_dis` | exponential distance weight 평균 | ↑ 높을수록 좋음 | fingertip이 가까울수록 높은 가중치 → 값이 올라가면 pinch 자세를 잘 만들고 있다는 의미 |

### 핸드별 메트릭
| key 패턴 | 설명 | 이상적 방향 | 비고 |
|----------|------|------------|------|
| `rec/{hand}` | 핸드별 reconstruction MSE | ↓ 낮을수록 좋음 | 핸드 간 비교 시: DOF가 높은 핸드(allegro 16, paxini 16)가 DOF가 낮은 핸드(ability 6, inspire 6)보다 높은 것이 정상. 특정 핸드만 안 줄면 해당 autoencoder에 문제 |
| `kl/{hand}` | 핸드별 KL divergence | ↕ 적절한 범위 유지 | 핸드 간 KL이 비슷한 수준이어야 shared latent space가 균형 잡힘. 한 핸드만 극단적으로 높거나 낮으면 latent 공유가 편향됨 |

### 학습 상태
| key | 설명 | 이상적 방향 | 비고 |
|-----|------|------------|------|
| `lr` | 현재 learning rate | — | scheduler 사용 시 감소 추이 확인용. 현재는 고정값 |
| `grad_norm` | 전체 파라미터 gradient L2 norm | ↕ 안정적 유지 | 급격히 튀면(spike) 학습 불안정 → lr 낮추기 고려. 0에 가까워지면 학습 정체. 일반적으로 1~100 범위에서 안정적이면 정상 |

### 학습 단계별 예상 패턴

```
초반 (0~20%):  rec loss 급감, pinch loss 천천히 감소, kl 상승
중반 (20~60%): rec loss 수렴, pinch loss 본격 감소, kl 안정화, exp_dis 상승
후반 (60~100%): 전체 loss 미세 감소, exp_dis 높은 수준 유지, grad_norm 안정
```

- **정상 학습**: `loss_total`이 단조 감소 추세, `exp_dis`가 점진적 상승, `grad_norm` 안정
- **학습 불안정 징후**: `grad_norm` spike, `loss_total` 급등, `kl/{hand}` 간 큰 편차
- **posterior collapse 징후**: `loss_kl` → 0, `rec` loss는 낮지만 cross-embodiment 성능 저하

### 설정
- **project**: `DexLatent`
- `train_cfg.json`의 파라미터가 wandb config로 함께 기록됨
- `WANDB_SILENT=true`로 콘솔 출력 최소화
