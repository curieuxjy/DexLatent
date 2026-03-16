# 새로운 핸드 Embodiment 추가 가이드

이 문서는 DexLatent에 새로운 핸드 로봇(embodiment)을 추가하는 과정을 Allegro Hand 사례를 기반으로 정리한 것입니다.

## 전체 흐름 요약

```
1. Combined URDF 생성 (xarm7 arm + new hand)
2. HAND_CONFIGS 등록 (kinematics.py)
3. train.py / infer.py에 hand name 추가
4. 재학습 및 검증
```

---

## 1. Combined URDF 생성

DexLatent의 모든 핸드는 **xarm7 arm + hand**가 하나의 URDF로 합쳐진 형태입니다. 새로운 핸드를 추가하려면 이 combined URDF를 만들어야 합니다.

### 구조

```
Assets/
├── xarm7/                  # 기본 arm URDF (link_base ~ link7)
├── allegro_hand/           # 핸드 단독 URDF + mesh
└── xarm7_allegro/          # 합쳐진 URDF (여기를 새로 만듦)
    ├── xarm7_allegro_right.urdf
    └── xarm7_allegro_left.urdf
```

### 핵심: `joint_eef`

arm의 마지막 링크(`link7`)와 핸드의 루트 링크(`base_link`)를 fixed joint로 연결합니다.

```xml
<joint name="joint_eef" type="fixed">
    <origin xyz="0 0 0.1" rpy="0 0 0" />
    <parent link="link7" />
    <child link="base_link" />
</joint>
```

- `xyz`: 핸드의 z축 방향 오프셋 (마운팅 높이)
- `rpy`: 핸드의 회전 방향 (마운팅 각도)
- 이 값은 실제 하드웨어 마운팅에 따라 조정이 필요합니다.

### mesh 경로

combined URDF에서 mesh는 상대경로로 참조합니다:
- arm mesh: `../xarm7/meshes/visual/link_base.glb`
- hand mesh: `../allegro_hand/meshes/allegro/base_link.obj`

---

## 2. HAND_CONFIGS 등록

`HandLatent/kinematics.py`의 `HAND_CONFIGS` dict에 새 핸드를 등록합니다.

```python
"xarm7_allegro_right": {
    "urdf_path": os.path.join(ASSETS_DIR, "xarm7_allegro", "xarm7_allegro_right.urdf"),
    "root_link": "link_base",
    "wrist_link": "link7",
    "tip_links": (
        "link_15.0_tip",   # thumb (반드시 첫 번째)
        "link_3.0_tip",    # index
        "link_7.0_tip",    # middle
        "link_11.0_tip",   # ring
    ),
},
```

### tip_links 순서가 중요한 이유

pinch pair 기본값이 `((0, 1), (0, 2), (0, 3), (0, 4))`이므로 **index 0 = thumb**이어야 합니다. Allegro는 4개 손가락이라 `(0, 4)` pair는 `pinch_pairs_for_hand()`에서 자동으로 필터링됩니다.

---

## 3. train.py / infer.py 수정

### train.py

`hand_names` 리스트에 추가:

```python
hand_names = [
    "xarm7_xhand_right",
    "xarm7_ability_right",
    "xarm7_inspire_right",
    "xarm7_paxini_right",
    "xarm7_allegro_right",
]
```

### infer.py

`trainer_hands`와 `target_hands` 두 곳에 추가:

```python
# trainer_hands
f"xarm7_allegro_{side}",

# target_hands
f"xarm7_allegro_{side}",
```

---

## 4. 학습 시 새로운 핸드가 적용되는 과정

`CrossEmbodimentTrainer`에 `hand_names`가 전달되면 다음이 자동으로 일어납니다:

### 초기화

1. `MultiHandDifferentiableFK`가 URDF를 파싱하여 각 핸드별 FK 모델 생성
2. 핸드별 DOF 파악 (allegro: 7 arm + 16 hand = 23 DOF)
3. 핸드별 `HandAutoencoder` 생성 — hand DOF에 맞는 encoder/decoder MLP 자동 구성
4. 모든 autoencoder의 파라미터가 하나의 AdamW optimizer에 등록

### 학습 루프 (`step()`)

```
모든 핸드에 대해:
  1. 랜덤 qpos 샘플링 (uniform + pinch template 혼합)
  2. autoencoder forward → hand latent 추출 + hand 복원
  3. reconstruction loss 계산 (hand 부분만)

모든 (source, target) 핸드 쌍에 대해:
  4. source의 hand latent → target의 decoder로 decode
  5. FK로 fingertip 위치 계산
  6. cross-embodiment pinch loss 계산 (거리 + 방향)

전체 loss = reconstruction + pinch_distance + pinch_direction + KL
```

**핵심**: 모든 핸드의 autoencoder가 **같은 latent space (dim=32)**를 공유하도록 학습됩니다. source 핸드에서 추출한 latent를 target 핸드의 decoder에 넣어도 유사한 pinch 동작이 나오도록 pinch loss가 강제합니다.

### 추론 시 적용 과정

```
1. source 핸드 trajectory → normalize → encode → hand latent 추출
2. 각 target 핸드에 대해:
   a. hand latent → target decoder → target hand qpos
   b. Pink IK로 arm 풀기 (wrist pose + alignment point 맞춤)
   c. arm + hand 합쳐서 최종 qpos 출력
3. Rerun으로 시각화
```

### Rerun 시각화 동작 과정

`infer.py`의 `main()`에서 디코딩된 궤적을 Rerun으로 시각화하는 과정입니다.

```
1. rr.init(spawn=True)로 Rerun Viewer GUI 프로세스를 자동 실행

2. visualize_hand_motion() 호출 (source 원본 + 각 target 디코딩 결과)
   │
   ├─ URDFLogger(urdf_path, prefix)
   │   └─ URDF 파일을 파싱하여 각 링크의 3D 메시(STL/DAE/OBJ)를 Rerun에 정적 로깅
   │      entity_path_prefix로 핸드별 구분 (예: "xarm7_xhand_right_decode/")
   │
   ├─ discover_revolute_joints() / discover_mimic_joints()
   │   └─ URDF에서 관절 정보 추출 (axis, limit, origin)
   │
   ├─ scale_joint_values()
   │   └─ normalized qpos [-1, 1] → 실제 라디안 각도로 변환
   │
   └─ 프레임 루프 (매 프레임마다):
       ├─ set_time(step * 0.02)  # 타임라인 위치 설정 (20ms 간격)
       ├─ per_frame_root_offsets 적용 (left/right side 오프셋)
       ├─ 각 revolute joint에 대해:
       │   rotation = origin_rotation × axis_rotation(angle)
       │   → rr.Transform3D 로깅 (translation + quaternion)
       └─ 각 mimic joint에 대해:
           angle = reference_angle × multiplier + offset
           → 동일하게 Transform3D 로깅
```

**핵심 원리**: Rerun은 데이터 로깅 기반 시각화입니다. 코드가 타임라인에 Transform3D 데이터를 push하면, Viewer가 타임라인 슬라이더로 재생합니다. URDF 메시는 한 번만 등록하고, 매 프레임에서는 각 관절의 Transform만 업데이트하여 애니메이션을 구현합니다.

**동시 표시**: source 원본과 5개 target 디코딩 결과가 각각 다른 `entity_path_prefix`로 로깅되므로, Rerun Viewer에서 6개 로봇이 나란히 표시됩니다.

---

## Allegro Hand 특이사항

| 항목 | Allegro | 기존 핸드 (xhand, inspire 등) |
|------|---------|-------------------------------|
| 손가락 수 | 4개 (pinky 없음) | 5개 |
| Hand DOF | 16 | 핸드마다 다름 |
| Total DOF | 23 (7 arm + 16 hand) | 핸드마다 다름 |
| Mimic joints | 없음 | 일부 핸드에 있음 |
| Pinch pairs | (0,1), (0,2), (0,3) | (0,1), (0,2), (0,3), (0,4) |

4개 손가락이라도 코드 변경 없이 동작합니다. `pinch_pairs_for_hand()`가 tip 개수에 따라 유효한 pair만 자동 선택하고, `HandAutoencoder`는 hand DOF를 인자로 받아 MLP 크기를 자동 조절합니다.
