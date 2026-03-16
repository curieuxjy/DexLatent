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
