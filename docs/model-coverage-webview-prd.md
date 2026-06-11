# PRD: 모델 기준 영상 촬영 Evidence 표시 웹뷰

## 1. 목적

`lingbot-map`에서 사용자가 영상을 업로드하면, 프로젝트의 기준 모델 파일인 `models/` 아래 GLB/PAG 데이터를 기준으로 업로드된 영상이 실제로 어느 위치와 어느 객체/구간을 촬영했는지 웹뷰에서 확인할 수 있게 한다.

이 기능의 판정 범위는 “시공 완료 여부”가 아니라 “해당 영상에서 모델의 어느 부분에 촬영 evidence가 있는가”이다. 자동 결과는 검토 가능한 근거로 제공하며, 확정 판정이 필요한 경우 사용자가 evidence frame을 확인해야 한다.

최종 사용자는 포인트클라우드나 재구성 메시의 품질만 보는 것이 아니라, 설계/기준 모델 위에서 다음 질문에 답할 수 있어야 한다.

- 이 영상은 기준 모델의 어느 구역을 촬영했는가?
- 영상 경로가 모델 공간에서 어떻게 이동했는가?
- 각 시점마다 어떤 장면을 보고 있었는가?
- 모델 객체 중 촬영 evidence가 있는 부분과 없는 부분은 무엇인가?
- 자동 정합/판정 결과가 불확실한 경우 어떤 프레임을 보면 검수할 수 있는가?

## 2. 배경

현재 `lingbot-map`은 영상 업로드 후 LBP4 점군, LBM1 메시, GLB 메시, 카메라 포즈, 썸네일을 생성할 수 있다. 그러나 재구성 결과를 그대로 보면 중복 표면, 포즈 드리프트, 낮은 텍스처 품질 때문에 실제 객체를 식별하기 어렵다.

이 기능은 재구성 결과를 최종 시각 객체로 쓰는 대신, `models/`의 기준 모델을 화면의 기준으로 사용한다. 영상 재구성 결과는 모델 위에 촬영 evidence layer로 표시한다. 재구성 점군/메시가 모델 근처에 있다는 사실만으로 해당 객체가 명확히 촬영됐다고 단정하지 않는다.

### 2.1 설계 원칙과 Why 검증

이 PRD는 다음 질문에 대한 답을 전제로 한다.

| Why 질문 | 판단 | 설계 반영 |
| --- | --- | --- |
| 왜 포인트클라우드/메시를 주 화면으로 쓰지 않는가? | 현재 재구성물은 중복 표면, 드리프트, 낮은 텍스처 품질 때문에 확대 검토용 기준 형상으로 부적합하다. | 기준 모델을 주 화면으로 두고, 재구성물은 evidence layer로만 사용한다. |
| 왜 확정형 상태명을 피하는가? | 모델 근처에 점이 있다는 사실만으로 객체가 명확히 촬영됐다고 볼 수 없다. | 상태명을 `observed`, `likely_observed`, `uncertain`, `not_observed`로 둔다. |
| 왜 manifest가 필요한가? | GLB와 PAG JSON이 같은 단위, 원점, 축, GUID를 공유한다는 보장이 없다. | coverage 분석 전에 `model_manifest` 검증을 필수로 둔다. |
| 왜 최근접 거리만 쓰지 않는가? | 반복 배관/덕트, 벽면, 재구성 중복점이 가까운 객체에 잘못 붙을 수 있다. | visibility, temporal support, geometry consistency를 함께 사용한다. |
| 왜 수동 정합을 허용하는가? | 영상만으로 BIM 좌표에 자동 정합하는 것은 현장 환경에서 신뢰도가 낮다. | MVP는 수동 대응점과 품질 게이트를 사용하고, 자동 정합은 확장으로 둔다. |
| 왜 원본 keyframe/crop이 필요한가? | 썸네일은 이동경로 확인에는 충분하지만 객체 판정 검수에는 해상도가 부족할 수 있다. | evidence 판정 프레임은 고해상도 keyframe 또는 crop을 저장한다. |
| 왜 정합 전 scan을 모델 중심에 임의로 맞추지 않는가? | 임의 중심 정렬은 화면을 그럴듯하게 만들지만 실제 위치 증거가 아니며 잘못된 coverage를 만든다. | 정합 전에는 raw scan local 상태를 명시하고, green/yellow 정합 전 coverage 분석을 막는다. |

### 2.2 보완 후 설계 판단

현재 재구성 메시/포인트클라우드는 원본 실사 객체처럼 확대 검토하기에는 품질 한계가 있다. 따라서 본 기능의 1차 목표는 “실사형 3D 복원”이 아니라 “기준 모델의 어느 부분을 촬영했는지 증거를 연결해 검토 가능하게 만드는 것”이다.

보완 설계는 다음 판단을 따른다.

- 스캔 GLB/점군은 기준 형상이 아니라 정합과 evidence 계산에 쓰는 보조 데이터다.
- 웹뷰 장면은 BIM 기준 `Z-up`으로 통일한다. PAG proxy, 모델 경로, 정합 후 scan은 같은 축으로 보여야 한다.
- 정합 전 scan/경로는 영상 재구성 local 좌표로만 표시한다. 모델 bounds에 맞춰 자동 이동/회전/스케일 보정해 보여주는 것은 금지한다.
- 사용자가 스캔 GLB와 모델 proxy에서 같은 물리 위치를 직접 찍어 대응점을 만들 수 있어야 한다.
- 정합 대응점은 JSON 직접 입력만으로 운영하기 어렵기 때문에 웹뷰에서 클릭 기반으로 생성/수정한다.
- 대응점이 4개 미만이거나 정합 품질이 `red`이면 coverage 결과를 자동 `observed`로 승격하지 않는다.
- 90% 품질 기준은 메시 텍스처 품질이 아니라 “모델 기준 위치/경로/evidence의 검토 정확도”로 판단한다.

## 3. 기준 입력

### 3.1 모델 파일

현재 프로젝트의 기준 모델 후보는 다음 파일이다.

| 파일 | 용도 | 비고 |
| --- | --- | --- |
| `models/gasan-7F.glb` | 전체 또는 현장 기준 3D 모델 | 약 62.8MB |
| `models/PXX.glb` | PXX 계통/부분 모델 | 약 1.06MB |
| `models/SXX.glb` | SXX 계통/부분 모델 | 약 0.83MB |
| `models/pxx_pag_export.json` | PXX 객체 메타데이터 | pipe, elbow, tee, valve, reducer 포함 |
| `models/pag_export.json` | 전체/PAG 메타데이터 | 프로젝트, 레벨, 시스템 정보 포함 |

### 3.1.1 모델 manifest

GLB와 PAG JSON은 서로 다른 export 경로에서 생성될 수 있으므로, 다음 정보를 명시하는 manifest가 필요하다.

필수 필드:

- `model_id`
- `glb_path`
- `metadata_path`
- `units`: 예: `mm`, `m`
- `up_axis`: 예: `Z_UP`, `Y_UP`
- `pag_to_glb_transform`: PAG 좌표를 GLB 좌표로 옮기는 4x4 matrix
- `guid_mapping_policy`: GLB node name, extras, external map 중 어떤 방식으로 GUID를 연결하는지
- `fallback_geometry`: GUID 매핑 실패 시 PAG 좌표 기반 proxy geometry 사용 여부

예시:

```json
{
  "model_id": "pxx",
  "glb_path": "models/PXX.glb",
  "metadata_path": "models/pxx_pag_export.json",
  "units": "mm",
  "up_axis": "Z_UP",
  "pag_to_glb_transform": [
    [0.001, 0, 0, 0],
    [0, 0.001, 0, 0],
    [0, 0, 0.001, 0],
    [0, 0, 0, 1]
  ],
  "guid_mapping_policy": "node_name_or_extras",
  "fallback_geometry": "pag_proxy"
}
```

수용 기준:

- GLB node와 PAG GUID 매핑률을 계산한다.
- 매핑률이 낮으면 객체 직접 색칠 대신 PAG proxy geometry overlay로 표시한다.
- 단위, 원점, 축 방향 변환이 불명확하면 coverage 분석을 실행하지 않고 `model_manifest_invalid` 상태를 반환한다.

### 3.2 영상 처리 산출물

업로드 영상 처리 후 다음 산출물이 필요하다.

| 산출물 | 현재 상태 | 목적 |
| --- | --- | --- |
| `*.lbp4.*` | 존재 | 점군, 색상, source frame id, component id |
| `*.lbm1.mesh` | 존재 | TSDF 메시, RGB vertex color, camera poses |
| `*.mesh.glb` | 존재 | 웹뷰에서 스캔 overlay로 사용 |
| `*.thumbs.json` | 존재 | 프레임별 썸네일 표시 |
| keyframe 원본/crop | API 추가됨 | 객체 evidence 검수 |
| camera poses | 존재 | 촬영 경로, frustum 표시 |
| frame timestamp | 보강 필요 | 촬영 시점별 탐색 |
| scan-to-model alignment | 신규 필요 | 영상 재구성 좌표를 모델 좌표에 정합 |
| model manifest | 신규 필요 | GLB/PAG 좌표계와 GUID 매핑 검증 |

## 4. 목표 사용자 경험

### 4.1 업로드 후 기본 화면

사용자가 영상을 업로드하고 처리가 끝나면 웹뷰는 다음 상태로 열린다.

1. `models/` 기준 모델이 먼저 표시된다.
2. 업로드 영상의 이동 경로가 모델 위에 라인으로 표시된다.
3. 일정 간격의 카메라 위치가 frustum 또는 방향 화살표로 표시된다.
4. 촬영 evidence가 있는 모델 구간은 색상 또는 반투명 overlay로 표시된다.
5. 우측 또는 하단 패널에서 시점별 썸네일을 타임라인처럼 확인할 수 있다.

### 4.2 모델 기준 촬영 evidence 표시

모델 객체 또는 표면은 촬영 evidence 상태에 따라 다음처럼 표시한다.

| 상태 | 의미 | 표시 |
| --- | --- | --- |
| `observed` | 촬영 evidence가 충분하고 정합 품질도 허용 범위임 | 녹색 또는 cyan highlight |
| `likely_observed` | 일부 evidence가 있으나 coverage 또는 support가 부족함 | 노란색 highlight |
| `uncertain` | 정합/거리/프레임 support가 애매함 | 주황색 또는 점선 overlay |
| `not_observed` | 해당 영상에서 촬영 evidence 없음 | 기본 모델 색상 또는 회색 |
| `out_of_scope` | 선택한 모델/구역 외부 | 흐리게 표시 |

초기 MVP에서는 객체 단위 evidence 판정을 우선한다. 이후 표면 heatmap으로 확장한다. `observed`는 “시공 완료”가 아니라 “이 영상에서 관측 근거가 충분함”을 뜻한다.

### 4.3 이동경로와 썸네일

웹뷰는 영상의 시간 흐름을 간단히 확인할 수 있어야 한다.

- 카메라 궤적 라인 표시
- keyframe 점 표시
- keyframe 점 클릭 시 해당 썸네일 표시
- 썸네일 클릭 시 3D 카메라 위치로 이동
- 프레임 번호와 timestamp 표시
- 현재 선택된 시점의 frustum 강조
- 해당 시점에서 보이는 모델 객체 후보 표시
- 프레임 후보 목록은 `visibility_frames`와 `support_frames`를 구분해 `visible`/`support` 근거를 표시한다.

## 5. 기능 요구사항

### FR-00. 모델 manifest 및 GUID 매핑 검증

coverage viewer는 모델을 로딩하기 전에 GLB/PAG 좌표계와 GUID 매핑 가능성을 검증해야 한다.

필수:

- `models/*.glb`와 `models/*_pag_export.json` 조합을 manifest로 관리
- PAG 단위와 GLB 표시 단위 변환
- PAG 좌표계와 GLB 좌표계의 원점/축 방향 변환
- GLB mesh/node와 PAG GUID 매핑률 계산
- GUID 매핑 실패 객체에 대한 PAG proxy geometry 생성

수용 기준:

- 모델 로딩 시 `model_manifest_valid`, `guid_mapping_ratio`, `fallback_geometry_count`를 표시한다.
- `/api/models` 조회 시 `models/*.model_manifest.json` 파일이 생성 또는 갱신된다.
- GUID 매핑률이 낮아도 PAG proxy geometry로 coverage 결과를 볼 수 있다.
- manifest 검증 실패 시 coverage 분석 버튼을 비활성화하고 원인을 표시한다.

### FR-01. 모델 로딩

시스템은 `models/` 아래 기준 모델을 웹뷰에 로딩할 수 있어야 한다.

필수:

- GLB 모델 선택: `gasan-7F.glb`, `PXX.glb`, `SXX.glb`
- PAG JSON 선택: `pag_export.json`, `pxx_pag_export.json`
- GLB 단독 표시
- PAG 객체 메타데이터를 GUID 기준으로 파싱
- 모델 단위 변환 처리: PAG JSON 단위는 `mm`, viewer 내부 단위는 `m` 또는 통일된 scene unit으로 변환
- manifest 기준 좌표 변환 적용

수용 기준:

- `/viewer.html` 또는 신규 coverage viewer에서 기준 모델이 5초 이내 표시된다.
- 모델 선택을 바꾸면 객체 메타데이터와 표시 범위가 함께 갱신된다.
- GLB GUID 매핑이 불완전해도 PAG proxy geometry로 객체 선택과 coverage 색상 표시가 가능하다.

### FR-02. 영상 업로드 처리 연동

영상 업로드 후 기존 inference pipeline 산출물을 coverage viewer에서 재사용한다.

필수:

- 업로드 ID 기준으로 LBP4/LBM1/GLB/thumbnail/pose를 조회
- 처리 완료 후 coverage viewer로 이동하는 버튼 제공
- 모바일 브라우저에서 카메라 촬영 업로드와 기존 영상 파일 선택을 구분해 제공
- 기존 `/audit` 결과에서 “이 결과 보기”와 “모델 기준 촬영 evidence 보기”를 구분
- `/upload`의 업로드 완료 상태와 이전 업로드 목록에서 `/coverage.html?model=pxx&upload={upload_id}` 링크 제공
- `/upload`의 업로드 완료 상태와 이전 업로드 목록에서 `/coverage-report.html?model=pxx&upload={upload_id}` 링크 제공
- `/audit` 결과 카드에서 기존 재생 버튼과 별도로 모델 기준 evidence 링크 제공

수용 기준:

- `upload_1779442357085.mp4` 같은 기존 업로드 결과를 다시 처리하지 않고 coverage viewer에서 불러올 수 있다.

### FR-03. scan-to-model 정합

영상 재구성 좌표계를 모델 좌표계로 변환하는 정합 결과를 생성/저장한다.

MVP:

- 웹뷰에서 스캔 GLB 점과 모델 proxy 점을 순서대로 클릭해 대응점 생성
- 스캔 GLB가 없는 업로드는 대응점 모드 진입 시 RGB 점군 샘플을 자동 로딩해 기준점 선택에 사용
- JSON 대응점 직접 입력/수정도 허용
- 수동 대응점 최소 4개 입력
- 운영 권장 대응점 6개 이상 입력
- 대응점은 한 구역에 몰리지 않고 촬영 경로의 시작/중간/끝 또는 모델 공간의 넓은 범위에 분포해야 함
- `lingbot_world -> model_world` Sim(3) 계산
- RMSE, per-point residual, scale, rotation, translation 표시
- leave-one-out residual 계산
- residual이 큰 대응점 후보 표시
- 정합 결과를 `realtime/_uploads/{upload_id}.{model_id}.alignment.json` 및 `realtime/_uploads/{upload_id}.alignment.json`으로 저장
- 저장된 대응점은 재접속 시 다시 로드되어 재분석에 사용할 수 있어야 함

확장:

- 모델 표면 기반 ICP 보정
- 반복 영상에서 이전 정합값 재사용
- 마커 또는 known point 기반 자동 정합
- 구간별 re-anchor 또는 window별 drift 보정

수용 기준:

- 대응점 4개 이상 입력 시 모델 위에 카메라 경로가 표시된다.
- 대응점 클릭 UI는 스캔 GLB 기준점과 모델 proxy 기준점을 서로 다른 색상 마커로 표시한다.
- 대응점 목록, 마지막 삭제, 전체 초기화를 제공한다.
- RMSE가 임계값을 넘으면 coverage 판정을 `uncertain`으로 낮춘다.
- 대응점 중 하나를 제외했을 때 변환이 크게 흔들리면 `alignment_unstable`로 표시한다.
- 정합 품질은 `green`, `yellow`, `red`로 표시한다. `red`에서는 자동 observed 판정을 생성하지 않는다.

### FR-03A. 반복 객체 환경의 자동 매핑 전략

배관, 덕트, 케이블 트레이처럼 비슷한 객체가 반복되는 구역에서는 영상 재구성만으로 전역 BIM 위치를 단정하지 않는다. 영상 기반 SLAM/재구성은 local 좌표와 상대 이동경로를 만들 뿐이며, 같은 형태의 객체가 여러 곳에 있으면 서로 다른 BIM 위치가 같은 수준의 기하 점수를 낼 수 있다.

원칙:

- 무제약 완전 자동 매핑은 MVP 판정의 신뢰 근거로 사용하지 않는다.
- 자동 매핑은 `single accepted alignment`가 아니라 `candidate hypotheses`를 만든 뒤 품질 게이트를 통과한 경우에만 적용한다.
- 1등 후보와 2등 후보의 점수 차이가 작으면 `ambiguous_alignment`로 표시하고 사용자 anchor를 요구한다.
- 자동 후보가 있더라도 coverage를 `observed`로 승격하려면 alignment 품질, temporal support, visibility support, negative evidence 검사를 모두 통과해야 한다.

필수 입력 또는 제약:

- 최소 1개 이상의 anchor: 사용자가 찍은 대응점, QR/AprilTag, known point, room/level/zone 선택, 이전 촬영분의 검증된 alignment 중 하나.
- 모델 후보 범위 제한: 전체 BIM이 아니라 선택 모델, 층, 구역, 시스템, 객체 category로 candidate search space를 줄인다.
- 스캔 품질 기준: 포즈 수, 이동거리, 충분한 시차, drift 지표가 부족하면 자동 매핑을 시도하지 않는다.

자동 후보 생성 흐름:

1. 영상 경로의 형태, 길이, 회전량, keyframe 간 상대 거리로 trajectory descriptor를 만든다.
2. 점군/메시에서 주 방향, 평면/선형 구조, pipe/duct-like 축을 추정해 local geometry descriptor를 만든다.
3. BIM proxy 객체에서 후보 subgraph를 만든다. 반복 객체가 많은 경우 여러 후보를 유지한다.
4. 후보별 Sim(3) 또는 SE(3)+scale을 RANSAC으로 계산하고, 필요 시 모델 표면 ICP로 보정한다.
5. 정합 점수는 RMSE만 보지 않고 visibility, frame support, geometry consistency, occlusion/negative evidence를 함께 반영한다.
6. 최고 후보가 margin 기준을 통과하면 `auto_alignment_candidate`로 저장하고, 통과하지 못하면 `ambiguous_alignment`로 저장한다.

UI 요구사항:

- 자동 후보는 모델 위에 후보 번호와 confidence를 표시한다.
- 후보별 예상 이동경로, 대표 썸네일, 매칭된 모델 객체 목록을 비교할 수 있어야 한다.
- 사용자는 후보를 승인하거나, 대응점 1-2개를 추가해 후보를 좁힐 수 있어야 한다.
- `can_save=true`인 후보만 명시적 “후보 저장” 버튼을 제공한다. 저장 불가 후보는 미리보기 전용으로 유지한다.
- 승인 전 후보는 coverage 결과를 `observed`로 만들 수 없고 `likely/uncertain`까지만 허용한다.

수용 기준:

- 유사 객체가 반복되는 모델에서 후보가 2개 이상이면 자동으로 하나를 확정하지 않고 `ambiguous_alignment`를 반환한다.
- 사용자 anchor 1-2개 추가 후 후보가 하나로 좁혀지면 기존 수동 정합과 같은 품질 게이트를 통과해야 한다.
- readiness/status API는 raw scan local 좌표, model 좌표, 정합 후 bounds를 분리해 표시한다.
- 정합 전에는 경로/scan overlay가 모델 위치가 아님을 명확히 표시한다.

### FR-04. 카메라 이동경로 표시

업로드 영상의 프레임별 pose를 모델 좌표계로 변환해 이동경로를 표시한다.

필수:

- 전체 카메라 경로 polyline
- keyframe 위치 점
- 선택 프레임 frustum
- 시작/끝 위치 표시
- 경로 표시 on/off

수용 기준:

- 80개 pose 기준으로 경로 표시가 브라우저에서 끊기지 않는다.
- keyframe 클릭 시 해당 썸네일과 timestamp가 표시된다.

### FR-05. 썸네일 타임라인

촬영 시점별로 영상을 빠르게 확인할 수 있는 썸네일 타임라인을 제공한다.

필수:

- 썸네일 목록 또는 필름스트립
- 프레임 번호, timestamp
- 현재 선택 프레임 highlight
- 썸네일 클릭 시 3D 경로 위치 선택
- “이 시점에서 보이는 모델 객체” 목록 표시
- evidence 검수용 keyframe 원본 또는 고해상도 crop 저장
- `/api/uploads/{upload_id}/keyframes`를 통해 evidence frame 원본 JPEG 생성/조회

수용 기준:

- `*.thumbs.json`에 있는 80개 썸네일을 모두 탐색할 수 있다.
- 썸네일 클릭과 3D 카메라 위치 선택이 서로 동기화된다.
- 선택 프레임에서 보이는 객체 후보 목록을 확인하고, 후보 클릭 시 객체 상세/evidence로 이동할 수 있다.
- 객체 판정의 근거로 쓰는 프레임은 썸네일만이 아니라 원본 keyframe 또는 crop으로 확인할 수 있다.

### FR-06. 모델 객체별 촬영 coverage 계산

PAG JSON의 객체를 기준으로 촬영 evidence를 계산한다.

MVP 알고리즘:

1. PAG 객체를 GUID 기준으로 변환한다.
2. pipe/duct/conduit/cableTray는 `start/end` 선형 객체로 샘플링한다.
3. elbow/tee/valve/reducer 등 point 객체는 `center + bbox`로 샘플링한다.
4. 각 프레임의 camera frustum 안에 들어오는 모델 객체 후보를 계산한다.
5. 정합된 LBP4 점군 또는 LBM1 mesh vertex와 모델 객체 샘플의 최근접 거리를 계산한다.
6. 선형 객체는 길이 방향 segment coverage를 계산한다.
7. pipe/duct/conduit처럼 방향성이 있는 객체는 길이 방향 segment bin coverage와 point 분포 span을 계산한다. normal/axis 일치도는 source normal 품질이 충분한 경우 추가한다.
8. 허용 거리, frustum visibility, temporal support, geometry consistency를 함께 사용해 evidence score를 계산한다.
9. source frame id 또는 근접 camera pose를 evidence frame으로 연결한다.

금지:

- 최근접 거리만으로 `observed`를 판정하지 않는다.
- 정합 품질이 `red`인 상태에서 자동 `observed`를 생성하지 않는다.
- 한 프레임 또는 한 점군 cluster만으로 긴 선형 객체 전체를 observed로 처리하지 않는다.

판정 초안:

| 상태 | 기준 |
| --- | --- |
| `observed` | coverage >= 0.70, frame support >= 3, alignment green/yellow, geometry consistency 통과 |
| `likely_observed` | coverage >= 0.30 and coverage < 0.70, 또는 support/consistency 일부 부족 |
| `uncertain` | 정합 RMSE 초과, 또는 frame support 부족 |
| `not_observed` | coverage < 0.30 and visible candidate였으나 evidence 부족 |
| `out_of_scope` | 해당 업로드의 촬영 frustum에 들어오지 않은 객체 |

수용 기준:

- GUID별 `coverage_ratio`, `segment_coverage_ratio`, `segment_hit_bins`, `linear_hit_span_ratio`, `nearest_distance_median`, `support_frames`, `visibility_frames`, `geometry_consistency`, `status`가 생성된다.
- 객체를 클릭하면 evidence frame 썸네일이 표시된다.
- `observed` 객체는 evidence frame에서 실제 촬영 흔적을 검수할 수 있어야 한다.

### FR-07. 촬영 evidence 시각화

모델 기준으로 촬영된 영역을 웹뷰에 표시한다.

필수:

- 객체 상태별 색상 표시
- observed/likely_observed/uncertain/not_observed/out_of_scope 필터
- 스캔 GLB overlay on/off
- RGB 점군 overlay on/off
- 모델 투명도 slider 조절
- 선택 객체 강조

수용 기준:

- 사용자는 스캔 메시를 보지 않아도 기준 모델만으로 촬영 evidence가 있는 부분을 식별할 수 있다.

### FR-08. 검토 패널

선택된 객체/프레임의 근거를 확인할 수 있는 패널을 제공한다.

필수 표시 항목:

- GUID
- category
- system
- level/zone
- coverage ratio
- segment coverage ratio
- median/max distance
- support frame count
- visibility frame count
- geometry consistency
- evidence thumbnails
- keyframe 원본 또는 crop
- alignment quality
- 판정 상태

수용 기준:

- 사용자는 왜 해당 객체가 observed/likely_observed/uncertain인지 화면에서 확인할 수 있다.

### FR-09. 결과 저장

coverage 분석 결과를 파일로 저장하고, 재접속 시 재사용한다.

필수 파일:

- `realtime/_uploads/{upload_id}.{model_id}.coverage.json`
- `realtime/_uploads/{upload_id}.coverage.json`
- `realtime/_uploads/{upload_id}.{model_id}.alignment.json`
- `realtime/_uploads/{upload_id}.alignment.json`
- 필요 시 `realtime/_uploads/{upload_id}.model_manifest.json`
- 필요 시 `realtime/_uploads/{upload_id}.keyframes/`
- 수동 검수 기록: `realtime/_uploads/{upload_id}.{model_id}.coverage_review.json`

예시:

```json
{
  "upload_id": "upload_1779442357085",
  "model": {
    "glb": "models/PXX.glb",
    "metadata": "models/pxx_pag_export.json",
    "units": "mm",
    "manifest_valid": true,
    "guid_mapping_ratio": 0.72,
    "fallback_geometry": "pag_proxy"
  },
  "alignment": {
    "quality": "yellow",
    "rmse_m": 0.18,
    "scale": 0.001,
    "stable": true,
    "max_leave_one_out_rmse_m": 0.24
  },
  "objects": [
    {
      "guid": "ca858adb-463f-40da-8d64-644fad7c8a41-000ec1a5",
      "category": "Pipe",
      "status": "observed",
      "coverage_ratio": 0.82,
      "segment_coverage_ratio": 0.78,
      "segment_hit_bins": 7,
      "segment_total_bins": 8,
      "linear_hit_span_ratio": 0.86,
      "nearest_distance_median_m": 0.045,
      "geometry_consistency": 0.81,
      "visibility_frames": [10, 11, 12, 13, 14, 15],
      "support_frames": [12, 13, 14, 15],
      "evidence_thumb_ids": [12, 14],
      "evidence_keyframes": ["keyframes/frame_000012.jpg"]
    }
  ]
}
```

## 6. 비기능 요구사항

### 성능

- 1000개 이하 객체 coverage 계산은 30초 이내를 목표로 한다.
- 브라우저에서는 100k개 이상의 점을 한 번에 표시하지 않는다. 필요 시 downsample한다.
- GLB 모델은 progressive loading 또는 선택 모델 로딩으로 처리한다.

### 신뢰도

- alignment quality가 낮으면 observed 상태를 생성하지 않는다.
- coverage 결과에는 항상 근거 frame과 confidence를 남긴다.
- alignment quality가 `red`이면 자동 observed 판정을 생성하지 않는다.
- `not_observed`는 “미시공”이 아니라 “이 영상에서 촬영 evidence 없음”으로 표현한다.
- `observed`는 “설치 완료”가 아니라 “촬영 evidence가 충분함”으로 표현한다.

### 사용성

- 기본 화면은 모델 기준으로 보여야 한다.
- 점군/스캔 메시 overlay는 보조 토글이어야 한다.
- 사용자가 가장 먼저 볼 정보는 “촬영된 모델 영역”과 “이동경로”다.

## 7. 제외 범위

MVP에서 제외한다.

- 무제약 완전 자동 BIM global localization
- 실사 수준 texture baking
- Gaussian Splatting production viewer
- 시공 완료/미시공 확정 판정
- 여러 날짜 촬영분 간 progress delta
- 모바일 실시간 coverage 표시

단, 위 항목은 후속 단계에서 확장 가능해야 한다.

## 8. API 요구사항

### POST `/api/upload-video`

영상 업로드와 처리 완료 후 업로드 ID 및 coverage viewer 진입 URL을 반환한다.

응답에는 최소 다음 필드가 포함된다.

```json
{
  "ok": true,
  "id": "upload_1779442357085",
  "file": "upload_1779442357085.mp4",
  "coverage_url": "/coverage.html?model=pxx&upload=upload_1779442357085",
  "report_url": "/coverage-report.html?model=pxx&upload=upload_1779442357085",
  "frames_used": 80,
  "elapsed_s": 12.3
}
```

### GET `/api/models`

`models/` 아래 사용 가능한 기준 모델 목록을 반환한다.

응답 예시:

```json
{
  "items": [
    {
      "id": "pxx",
      "name": "PXX",
      "glb": "/models/PXX.glb",
      "metadata": "/models/pxx_pag_export.json",
      "manifest": "/models/PXX.model_manifest.json",
      "units": "mm",
      "manifest_valid": true,
      "guid_mapping_ratio": 0.72,
      "guid_mapping_status": "direct",
      "fallback_geometry": "pag_proxy",
      "coverage_geometry_source": "glb_guid_mesh",
      "glb_role": "coverage_geometry"
    }
  ]
}
```

GUID 매핑률이 낮은 모델은 `coverage_geometry_source = pag_proxy`, `glb_role = visual_context_only`로 반환한다. 이 경우 coverage 색상과 객체 선택은 GLB mesh가 아니라 PAG proxy geometry를 기준으로 처리한다.

### GET `/api/uploads/{upload_id}/coverage`

기존 coverage 결과를 반환한다. 없으면 `404` 또는 `needs_analysis`를 반환한다.

### GET `/api/uploads/{upload_id}/alignment`

기존 정합 결과를 반환한다. `model_id`가 있으면 모델별 alignment를 우선 조회한다. 없으면 `404` 또는 `needs_alignment`를 반환한다.

### GET/POST `/api/uploads/{upload_id}/alignment/candidates`

반복 객체 환경에서 자동 정합 후보를 계산하거나 마지막 후보 결과를 조회한다.

요청 예시:

```json
{
  "model_id": "pxx",
  "alignment": {
    "pairs": [
      {"scan": [0.1, 0.0, -0.2], "model": [-3.2, 8.1, 26.4]}
    ]
  },
  "max_candidates": 5
}
```

응답 규칙:

- 대응점 4개 이상이면 `manual_pairs_preview` 후보를 생성하고 저장 가능 여부를 표시한다.
- 대응점 0-3개이면 이전 검증 정합을 prior 후보로만 반환한다.
- 반복 객체 위험이 높거나 후보 점수가 비슷하면 `ambiguous_alignment`를 반환한다.
- 후보는 미리보기용이며, 저장된 alignment가 아니므로 coverage observed 판정에는 사용하지 않는다.
- 저장 전에는 사용자가 대응점/anchor를 추가하거나 후보를 검증해야 한다.
- `manual_pairs_preview`처럼 `can_save=true`인 후보는 사용자가 명시적으로 저장해야만 `/alignment` 결과가 된다.

### GET `/api/uploads/{upload_id}/coverage/status`

coverage viewer가 분석 실행 전에 표시할 준비 상태를 반환한다.

필수 응답:

- `status`: `model_manifest_invalid`, `needs_scan`, `needs_alignment`, `needs_analysis`, `ready` 중 하나
- `next_action`: 사용자가 바로 수행해야 할 다음 조치
- `scan`: payload 종류, point/pose/thumb 수, scan GLB 존재 여부, source frame ID 존재 여부
- `alignment`: 품질, RMSE, leave-one-out residual, 대응점 분포 상태
- `coordinate`: raw scan local bounds, model bounds, 정합 후 scan bounds, `alignment_required`
- `checks`: manifest, GUID/기준 형상, scan payload, pose/thumb, scan overlay, source frame, alignment, 자동 매핑 위험, coverage 결과, keyframe 준비 상태를 `pass/warn/block/pending`으로 반환
- `warnings`: 현재 화면이 왜 모델 기준 evidence가 아닌지 설명하는 경고 목록

좌표 진단 규칙:

- 정합 전에는 `coordinate.state = raw_scan_local`이며, scan/경로 overlay는 BIM 모델 위치가 아니라고 표시한다.
- 정합 후에는 `coordinate.state = aligned_model_world`이며, `aligned_scan_bounds`가 모델 bounds와 비교 가능한 값이 된다.
- raw scan local bounds와 model bounds가 멀리 떨어져 보여도 이것만으로 모델 축 오류로 판정하지 않는다. 유효한 scan-to-model alignment가 없으면 정상적인 중간 상태다.

### GET `/api/uploads/{upload_id}/coverage/report`

coverage 결과를 기준으로 90% 품질 기준의 자동 게이트와 수동 검수 필요 항목을 반환한다.

필수 응답:

- `status`: `blocked`, `needs_evidence_review`, `ready_for_manual_review`
- `automated_gate_passed`: 정합, coverage, observed evidence traceability, support frame 기준을 자동으로 만족했는지 여부
- `manual_review_required`: 최종 90% 판정에 evidence keyframe/crop 육안 검수가 필요한지 여부. MVP에서는 항상 `true`
- `passed_90`: 자동 게이트만으로는 최종 통과 처리하지 않는다. 수동 검수 기록이 없으면 `false`
- `criteria`: keyframe 위치 90%, observed evidence 90%, support frame, 과대판정 방지, 수동 evidence 검수 상태
- `blockers`: 정합 없음, red/stale coverage, scan 없음 등 분석을 신뢰할 수 없는 이유

수용 기준:

- 정합이 없거나 red이면 `blocked`를 반환한다.
- observed 객체가 evidence frame에 90% 이상 연결되지 않으면 자동 게이트를 통과하지 않는다.
- 자동 게이트가 통과해도 최종 90% 통과는 사람이 keyframe/crop을 확인한 뒤 별도 확정해야 한다.

### GET/POST `/api/uploads/{upload_id}/coverage/review`

수동 evidence 검수 기록을 조회하거나 저장한다.

저장 항목:

- `object_reviews`: GUID별 evidence 확인 여부, 대표 frame, note
- `frame_reviews`: frame별 모델 위치 확인 여부, note
- `notes`: 전체 검수 메모

수용 기준:

- 객체 상세에서 evidence 확인/불일치를 저장할 수 있다.
- 선택 프레임에서 모델 위치 확인/불일치를 저장할 수 있다.
- 잘못 저장한 객체/프레임 검수 기록을 삭제할 수 있다.
- 전체 검수 메모를 저장하고, 필요 시 수동 검수 기록 전체를 초기화할 수 있다.
- `/coverage/report`는 이 검수 기록을 읽어 observed 객체 검수율과 keyframe 위치 검수율을 계산한다.
- 최종 `passed_90`은 자동 게이트와 수동 검수 게이트가 모두 90% 이상일 때만 `true`가 된다.

### GET `/api/uploads/{upload_id}/keyframes`

지원 프레임의 원본 영상 기반 JPEG keyframe을 생성하고 목록을 반환한다.

요청 예시:

```text
/api/uploads/upload_1779442357085/keyframes?frames=12,14,15&max_width=1280
```

응답 예시:

```json
{
  "ok": true,
  "id": "upload_1779442357085",
  "count": 3,
  "items": [
    {
      "frame": 12,
      "raw_frame": 184,
      "url": "/api/uploads/upload_1779442357085/keyframes/12.jpg",
      "width": 1280,
      "height": 720
    }
  ]
}
```

### GET `/api/uploads/{upload_id}/keyframes/{frame_idx}.jpg`

단일 keyframe JPEG를 반환한다. 파일이 없으면 원본 영상에서 on-demand로 생성한다.

### GET `/api/uploads/{upload_id}/scan`

업로드 산출물의 pose, thumbnail, 선택적 point sample을 반환한다.

점군 overlay용 요청 예시:

```text
/api/uploads/upload_1779442357085/scan?variant=detail&include_points=true&include_thumbs=false&max_points=120000
```

`include_points=true`일 때 응답은 `points`, `colors`, `source_ids`를 포함할 수 있다. `colors`는 원본 LBP/LBM payload의 RGB를 그대로 샘플링한 값이다.

### POST `/api/uploads/{upload_id}/alignment`

coverage 계산 없이 대응점만으로 scan-to-model 정합을 계산하고 저장한다.

요청 예시:

```json
{
  "model_id": "pxx",
  "alignment": {
    "pairs": [
      {"scan": [0, 0, 0], "model": [-2.8, -1.7, 26.8]},
      {"scan": [1, 0, 0], "model": [-1.8, -1.7, 26.8]},
      {"scan": [0, 1, 0], "model": [-2.8, -0.7, 26.8]},
      {"scan": [0, 0, 1], "model": [-2.8, -1.7, 27.8]}
    ]
  }
}
```

응답은 `quality`, `rmse_m`, `stable`, `scale`, `rotation`, `translation`, `per_point_residuals`, `pairs`를 포함한다.

### POST `/api/uploads/{upload_id}/coverage/analyze`

선택 모델과 정합 정보로 coverage를 계산한다.

저장된 `{upload_id}.{model_id}.alignment.json`이 있고 품질이 `green/yellow`이면 `alignment.pairs`를 생략할 수 있다. 요청에 `alignment.pairs`를 포함하는 경우에는 최소 4개가 필요하며, 분석 성공 시 새 정합으로 저장된다.

요청 예시:

```json
{
  "model_id": "pxx",
  "alignment": {
    "pairs": [
      {"scan": [0, 0, 0], "model": [-2.8, -1.7, 26.8]},
      {"scan": [1, 0, 0], "model": [-1.8, -1.7, 26.8]},
      {"scan": [0, 1, 0], "model": [-2.8, -0.7, 26.8]},
      {"scan": [0, 0, 1], "model": [-2.8, -1.7, 27.8]}
    ]
  },
  "coverage_options": {
    "distance_threshold_m": 0.10,
    "min_support_frames": 3,
    "min_geometry_consistency": 0.6,
    "require_visibility": true
  }
}
```

### GET `/coverage.html?upload={upload_id}&model={model_id}`

모델 기준 coverage viewer를 연다.

### GET `/coverage-report.html?upload={upload_id}&model={model_id}`

coverage/status, coverage/report, coverage/review, coverage 결과를 조합해 인쇄/PDF 가능한 검수 리포트 HTML을 연다.

수용 기준:

- blocker, readiness check, 90% criteria, 수동 검수 기록, 객체 evidence 목록을 한 페이지에서 확인할 수 있다.
- coverage가 없거나 stale red alignment인 경우에도 리포트는 열리고 차단 사유를 표시한다.
- 리포트에서 coverage viewer로 다시 이동할 수 있다.

## 9. UI 구성

### 상단 툴바

- 모델 선택
- 업로드 선택
- coverage 분석/재분석
- 정합 저장
- 경로 표시 toggle
- 썸네일 표시 toggle
- 스캔 overlay toggle
- 점군 overlay toggle
- 모델 투명도 slider
- 상태 필터

### 좌측/우측 패널

- 업로드 정보
- 모델 정보
- manifest 검증 상태
- GUID 매핑률
- alignment quality
- 대응점 생성/목록/삭제/초기화
- 상태별 객체 count
- 선택 객체 상세

### 하단 타임라인

- 썸네일 필름스트립
- frame number
- timestamp
- 선택 frame marker

### 3D 뷰

- 기준 모델
- 촬영 evidence 색상
- 이동경로 polyline
- keyframe frustum
- 선택 frame/object highlight

## 10. 데이터 처리 흐름

```text
Video Upload
  -> realtime inference
     -> LBP4 / LBM1 / GLB / poses / thumbnails
  -> model selection
     -> models/*.glb + models/*_pag_export.json
  -> model manifest validation
     -> units / axis / origin / GUID mapping / proxy geometry
  -> alignment
     -> scan_to_model Sim(3)
  -> coverage analysis
     -> object coverage + evidence frames
  -> coverage webview
     -> model colored by observed state
     -> camera path + thumbnails
```

## 11. 구현 단계

### Phase 0. 모델 manifest 검증

- `models/` 기준 모델 목록 정의
- GLB/PAG 조합별 manifest 작성
- 단위/축/원점 변환 검증
- GLB node와 PAG GUID 매핑률 산출
- GUID 매핑 실패 시 PAG proxy geometry 생성

완료 기준:

- `PXX.glb + pxx_pag_export.json` 조합에 대해 manifest 검증 결과가 생성된다.
- GUID 매핑 실패 객체도 proxy geometry로 웹뷰에 표시할 수 있다.

### Phase 1. 읽기 전용 coverage viewer

- `/coverage.html` 추가
- `/api/models` 추가
- manifest 검증 상태 표시
- 기존 업로드의 LBM1/LBP4 pose와 thumbs 로딩
- 모델 GLB 표시
- 이동경로와 썸네일 타임라인 표시

완료 기준:

- 기존 업로드를 선택하면 모델 위에 경로와 썸네일이 표시된다.

### Phase 2. 수동 정합

- 스캔 GLB -> 모델 proxy 순서의 대응점 클릭 UI
- 대응점 입력 JSON 편집 UI
- 대응점 마커와 pair line 표시
- 대응점 목록, 마지막 삭제, 전체 초기화
- Sim(3) 계산 API
- leave-one-out residual 검증
- unstable alignment 경고
- alignment 저장/로드
- 경로를 모델 좌표계에 표시

완료 기준:

- 사용자가 대응점 4개 이상을 입력하면 카메라 경로가 모델 위치에 맞춰진다.
- 새로고침 후에도 저장된 alignment와 대응점이 재사용된다.

### Phase 3. 객체 coverage 분석

- PAG JSON -> 내부 BIM metadata 변환
- 객체 샘플링
- LBP4/LBM1 기반 nearest distance, frustum visibility, temporal support, geometry consistency 계산
- GUID별 coverage 결과 저장

완료 기준:

- 모델 객체가 observed/likely_observed/uncertain/not_observed/out_of_scope로 색상 표시된다.

### Phase 4. 증거 프레임 연결

- 객체별 support frame 추출
- 상세 패널에 evidence thumbnails 표시
- 썸네일 클릭 시 3D camera 이동

완료 기준:

- 객체 클릭만으로 어떤 시점에 촬영됐는지 확인 가능하다.

### Phase 5. 품질 고도화

- ICP 보정
- frame visibility/occlusion 계산
- surface heatmap
- 모델-스캔 거리 histogram
- 여러 업로드 비교

완료 기준:

- coverage 결과가 단순 거리 기반에서 시야/가림/정합 품질까지 반영한다.

## 12. 품질 기준

### MVP pass 기준

- 기존 업로드 `upload_1779442357085.mp4`를 coverage viewer에서 열 수 있다.
- 선택 모델 `PXX.glb` 또는 `gasan-7F.glb`가 표시된다.
- 선택 모델의 manifest 검증 상태와 GUID 매핑률이 표시된다.
- 영상 이동경로와 keyframe 썸네일이 동기화된다.
- 웹뷰에서 스캔 GLB와 모델 proxy를 클릭해 대응점을 만들 수 있다.
- 수동 정합 후 경로와 스캔 overlay가 모델 좌표에 표시된다.
- 최소 1개 PAG 객체에 대해 observed/likely_observed/not_observed 판정 결과가 생성된다.

### 90% 품질 기준의 재정의

이 기능에서 90% 품질은 “실사형 메시 품질”이 아니라 다음 기준으로 판단한다.

- 선택한 테스트 구간의 keyframe 중 90% 이상이 올바른 모델 위치 근처에 표시된다.
- 사용자가 육안으로 촬영 경로와 촬영 영역을 모델 기준으로 이해할 수 있다.
- observed로 표시된 객체 중 90% 이상이 evidence keyframe 또는 crop에서 실제 촬영 흔적을 확인할 수 있다.
- uncertain 객체는 observed로 과대 판정하지 않는다.

## 13. 리스크와 대응

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| 영상 재구성 scale/pose drift | 경로와 coverage가 모델에서 벗어남 | 수동 정합, ICP, 품질 등급 |
| 모델과 실제 현장 차이 | distance 기반 판정 오류 | uncertain/review 상태 유지 |
| 반복 배관/덕트의 근접 객체 혼동 | 잘못된 객체가 observed로 표시됨 | visibility, temporal support, axis/normal consistency 동시 사용 |
| PAG JSON과 GLB GUID 연결 부족 | 객체별 색상 표시 어려움 | manifest 검증, GUID 매핑률 표시, PAG proxy fallback |
| 썸네일 해상도 부족 | 증거 검수 어려움 | keyframe 원본 이미지 저장 옵션 추가 |
| 모델 단위 mm/m 불일치 | 정합 실패 | 모델 manifest에 units 명시, scale 자동 적용 |
| 대응점 입력 오류 | 전체 경로와 coverage 왜곡 | 4-6개 대응점, leave-one-out residual, unstable 경고 |

## 14. 성공 지표

- 영상 업로드 후 사용자가 3분 안에 촬영 경로를 모델에서 확인할 수 있다.
- 객체 클릭 후 2초 안에 관련 evidence thumbnails와 keyframe/crop가 표시된다.
- 테스트 업로드 1건 기준 coverage 결과 JSON이 재사용 가능하게 저장된다.
- 동일 업로드를 새로고침해도 alignment와 coverage 결과가 유지된다.
- observed로 표시된 객체는 최소 3개 이상의 support frame 또는 동등한 품질 근거를 가진다.

## 15. 산출물

MVP 완료 시 다음 파일/기능이 존재해야 한다.

- `realtime/coverage.html`
- `realtime/coverage_report.html`
- `/api/models`
- `/api/uploads/{upload_id}/coverage`
- `/api/uploads/{upload_id}/coverage/analyze`
- `models/*.model_manifest.json`
- `realtime/_uploads/{upload_id}.alignment.json`
- `realtime/_uploads/{upload_id}.coverage.json`
- `realtime/_uploads/{upload_id}.{model_id}.coverage_review.json`
- `realtime/_uploads/{upload_id}.keyframes/`
- 모델 기준 이동경로/썸네일/촬영 evidence 웹뷰
