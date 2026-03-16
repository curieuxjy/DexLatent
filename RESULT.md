# Evaluation Results

## 설정

- **Checkpoint**: `Checkpoints/20260316_154432/checkpoint_epoch_1000.pt`
- **Source hand**: `xarm7_inspire_right`
- **평가 데이터**: `Dataset/demo.npz` (792 frames)
- **학습 핸드**: xhand, ability, inspire, paxini, allegro (5개)

---

## 1. Self-Reconstruction Error

encode → 같은 핸드로 decode 했을 때 hand qpos MSE.

| Hand | hand_qpos_mse | 데이터 |
|------|--------------|--------|
| xarm7_xhand_right | 0.009931 | random |
| xarm7_ability_right | 0.002815 | random |
| xarm7_inspire_right | 0.001609 | demo |
| xarm7_paxini_right | 0.011762 | random |
| xarm7_allegro_right | 0.013372 | random |

inspire가 가장 낮은 reconstruction error를 보임 (demo 데이터 기준).
allegro는 16 DOF로 가장 높은 hand DOF를 가져 reconstruction이 상대적으로 어려움.

| Hand | Total DOF | Arm DOF | Hand DOF | Fingers |
|------|-----------|---------|----------|---------|
| xarm7_xhand | 19 | 7 | 12 | 5|
| xarm7_ability | 13 | 7 | 6 | 5 |
| xarm7_inspire | 13 | 7 | 6 | 5 |
| xarm7_paxini | 23 | 7 | 16 | 4 |
| xarm7_allegro | 23 | 7 | 16 | 4 |

(4-Hardwares에 나와있음)

---

## 2. Cross-Embodiment Pinch Loss

inspire에서 encode한 latent를 각 target hand로 decode 후 pinch 거리/방향 오차.
exponential weight가 적용된 값 (pinch가 가까울수록 높은 가중치).

| Source → Target | pinch_distance | pinch_direction |
|----------------|---------------|----------------|
| inspire → xhand | 0.000005 | 0.001897 |
| inspire → ability | 0.000008 | 0.001277 |
| inspire → paxini | 0.000014 | 0.005006 |
| inspire → allegro | 0.000004 | 0.000544 |

allegro가 pinch distance, direction 모두 가장 낮은 오차를 기록.

---

## 3. Fingertip Position Error (Full Pipeline)

encode → decode + IK arm solving 후 source와 target의 fingertip L2 거리 (단위: m).
공유 가능한 fingertip만 비교 (allegro는 4개, 나머지는 5개).

| Source → Target | mean L2 | thumb | index | middle | ring | pinky |
|----------------|---------|-------|-------|--------|------|-------|
| inspire → xhand | 0.0070 | 0.0056 | 0.0051 | 0.0078 | 0.0065 | 0.0101 |
| inspire → ability | 0.0062 | 0.0035 | 0.0069 | 0.0080 | 0.0030 | 0.0096 |
| inspire → inspire | 0.0034 | 0.0021 | 0.0026 | 0.0058 | 0.0037 | 0.0028 |
| inspire → paxini | 0.0111 | 0.0053 | 0.0081 | 0.0087 | 0.0187 | 0.0147 |
| inspire → allegro | 0.0066 | 0.0064 | 0.0040 | 0.0056 | 0.0104 | - |

- inspire→inspire (self-retargeting): 3.4mm 평균 오차로 가장 낮음
- inspire→allegro: 6.6mm 평균 오차, ability(6.2mm)와 유사한 수준
- inspire→paxini: 11.1mm로 가장 높은 오차 (ring finger 18.7mm)

---

## 재현 방법

```bash
uv run -m HandLatent.evaluate \
  --ckpt Checkpoints/20260316_154432/checkpoint_epoch_1000.pt \
  --side right
```
