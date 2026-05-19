# BIM-VAMS 장점 흡수 계획

## 목적

현재 `lingbot-map`의 핵심 강점은 RGB 영상/이미지 시퀀스에서 카메라 포즈, 깊이, 포인트클라우드를 추정해 현장 공간을 3D로 재구성하는 것이다. 여기에 `D:\Git\BIM-VAMS`의 장점인 "현장 촬영 영상과 BIM 설계모델을 매핑하여 오늘 기준 시공/설치가 어느 단계까지 진행됐는지 확인"하는 기능을 흡수한다.

최종 목표는 단순 3D 재구성이 아니라 다음 질문에 답하는 시스템이다.

> 오늘 촬영한 현장 영상 기준으로, BIM 설계 객체 중 무엇이 설치됐고, 무엇은 미설치이며, 무엇은 연결 관계상 설치된 것으로 추론 가능한가?

## 전제와 범위

- `lingbot-map`은 BIM 분석 앱으로 바꾸지 않고, 영상 기반 현장 3D 인식 엔진을 확장한다.
- BIM-VAMS 전체 웹앱을 복사하지 않는다. 가져올 장점은 데이터 모델, 정합 흐름, 설치 상태 판정 로직, 리포팅 개념이다.
- MVP는 완전 자동 BIM localization이 아니라 기준점/마커/수동 대응점을 사용한 초기 정합을 허용한다.
- 결과에는 항상 confidence, uncertainty, evidence를 남긴다. 현장 영상과 BIM이 다를 수 있으므로 자동 판정을 확정값처럼 다루지 않는다.

## 흡수할 BIM-VAMS 핵심 장점

1. BIM 모델 기준의 객체 단위 상태 관리
   - `installed`, `detected`, `inferred`, `not_detected`, `unknown` 같은 상태를 BIM 객체 GUID 기준으로 관리한다.

2. 영상과 BIM 좌표계 정합
   - 현장 영상에서 추정한 카메라 포즈/포인트클라우드를 BIM 좌표계로 변환한다.
   - 기준점 3개 이상 또는 marker 기반으로 Sim(3) 변환을 계산한다.

3. BIM 객체 투영 기반 매칭
   - BIM 객체를 현재 카메라 화면에 투영하고 영상 관측 결과와 비교한다.
   - 객체가 화면에 보여야 하는데 검출되지 않으면 `not_detected` 또는 `needs_review`로 둔다.

4. MEP/BIM 연결 관계 기반 추론
   - A와 C가 설치 확인됐고 BIM topology상 A-B-C로 연결되어 있으면 B를 `inferred`로 판정한다.
   - 특히 배관/덕트/밸브처럼 일부가 가려지는 현장에 유용하다.

5. 공정 진척 리포트
   - 설계모델 전체 대비 설치율, 시스템별/구역별/공종별 진척률, 오늘 신규 확인 객체, 검토 필요 객체를 산출한다.

## 목표 아키텍처

```text
Site Video / Image Sequence
  -> lingbot-map inference
     -> frame camera poses
     -> depth / point cloud
     -> confidence maps

BIM Inputs
  -> GLB / glTF geometry
  -> BIM metadata JSON
     -> guid
     -> category / system / zone
     -> connectors / topology
     -> planned schedule / WBS(optional)

Alignment Layer
  -> Sim(3) transform: LingBot world -> BIM world
  -> BIM-coordinate frame poses
  -> uncertainty / drift metrics

Object Evidence Layer
  -> project BIM objects into each frame
  -> compare with video evidence
  -> aggregate per-object observations

Installation Reasoning Layer
  -> detected objects
  -> inferred objects through BIM topology
  -> not detected / unknown / needs review

Progress Report
  -> object status table
  -> zone/system/category progress
  -> daily delta
  -> review queue
```

## 단계별 계획

### Phase 0. 기능 경계 확정

검증 목표:
- `lingbot-map` 출력에서 BIM 정합에 필요한 최소 데이터가 안정적으로 나오는지 확인한다.
- BIM-VAMS의 어떤 모듈을 개념만 가져오고, 어떤 모듈은 코드 재사용 후보로 둘지 분리한다.

작업:
- `demo.py`/모델 출력에서 `extrinsic`, `intrinsic`, `world_points`, `world_points_conf`, `depth`, `depth_conf`의 저장 포맷을 정의한다.
- BIM 입력 포맷을 GLB + metadata JSON으로 고정한다.
- GUID, category, system_type, connectors, center/start/end 좌표를 필수/권장 필드로 나눈다.

산출물:
- `LingbotSceneExport` JSON 스키마
- `BimMetadata` JSON 스키마
- 샘플 영상 1개, 샘플 BIM 1개 기준 end-to-end 수동 검증 결과

### Phase 1. LingBot 결과 export 표준화

검증 목표:
- 현장 영상 한 번 처리 후 외부 시스템이 바로 읽을 수 있는 pose/pointcloud 결과가 생성된다.

작업:
- `lingbot-map` inference 결과를 다음 구조로 export한다.
  - `frames[]`: frame_number, timestamp, intrinsic, extrinsic, confidence
  - `point_cloud`: sampled points, colors, confidence
  - `source`: video path, fps, image size, preprocessing info
- 포인트클라우드는 원본 전체 저장과 lightweight sampled 저장을 분리한다.
- 좌표계 정의를 문서화한다. 특히 extrinsic이 c2w인지 w2c인지 명확히 한다.

성공 기준:
- 동일 입력 영상에서 export JSON이 재현 가능하다.
- BIM-VAMS류 분석 파이프라인이 `lingbot-map` 내부 모델을 몰라도 pose/pointcloud를 읽을 수 있다.

### Phase 2. BIM 좌표계 정합 레이어 추가

검증 목표:
- LingBot world 좌표를 BIM world 좌표로 변환해 각 프레임의 카메라 위치를 BIM 공간에 올릴 수 있다.

작업:
- 최소 3개 이상의 대응점으로 Sim(3) 정합을 계산한다.
- 대응점 입력 방식은 MVP에서 수동 지정으로 시작한다.
  - 영상/포인트클라우드 기준점
  - BIM 모델 기준점
- 정합 결과에 RMSE, scale residual, confidence를 포함한다.
- threshold 초과 시 `needs_review` 상태로 둔다.

성공 기준:
- 정합 RMSE가 0.5m 이하이면 green, 1.0m 이하이면 yellow, 초과하면 review로 분류한다.
- 변환된 카메라 frustum이 BIM 뷰어에서 현장 촬영 경로와 일관되게 표시된다.

### Phase 3. BIM 객체 투영 및 관측 evidence 생성

검증 목표:
- 각 프레임에서 "현재 카메라가 어떤 BIM 객체를 볼 수 있어야 하는지" 후보를 계산한다.

작업:
- BIM metadata의 center/start/end 또는 mesh bounding box를 이용해 객체 중심과 bbox를 만든다.
- BIM 객체를 카메라 이미지 평면에 투영한다.
- 화면 안에 들어온 객체에 대해 관측 evidence를 기록한다.
  - visible_candidate
  - projected_bbox
  - depth
  - view_angle
  - occlusion 가능성
  - frame quality
- 영상 evidence는 초기에는 색상/edge/geometry 기반으로 시작하고, 추후 객체 검출 모델을 붙인다.

성공 기준:
- 샘플 프레임에서 투영된 BIM 객체 위치가 실제 영상의 객체 위치와 대략 일치한다.
- reprojection error가 8px 이하면 pass, 20px 초과면 reject/review로 둔다.

### Phase 4. 설치 상태 판정 모델

검증 목표:
- BIM 객체별로 오늘 영상 기준 설치 상태를 산출한다.

상태 정의:
- `installed`: 영상 evidence와 BIM projection이 충분히 일치해 설치 확인됨
- `detected`: 영상에서는 감지됐지만 설치 확정 confidence가 낮음
- `inferred`: 직접 보이지 않지만 연결 관계/인접 객체 근거로 설치 추론됨
- `not_detected`: 보여야 하는 위치인데 관측되지 않음
- `unknown`: 시야 밖, 품질 낮음, occlusion 등으로 판단 불가
- `needs_review`: 자동 판정 threshold가 애매하거나 정합 품질이 낮음

작업:
- per-frame evidence를 per-object evidence로 aggregate한다.
- confidence fusion 규칙을 만든다.
  - projection confidence
  - visual match confidence
  - temporal consistency
  - alignment confidence
  - occlusion penalty
- BIM topology 기반 A-B-C 추론을 추가한다.

성공 기준:
- 객체별 상태와 그 근거 프레임/evidence를 함께 조회할 수 있다.
- 직접 관측과 topology 추론이 구분되어 저장된다.

### Phase 5. 공정 진척 리포트

검증 목표:
- "오늘 현장은 설계모델 기준 어디까지 설치됐는가?"에 대한 리포트를 만든다.

작업:
- 전체 객체 대비 설치율을 계산한다.
- category/system/zone/WBS별 진척률을 계산한다.
- 이전 촬영 결과와 비교해 daily delta를 만든다.
- `needs_review` 객체 목록을 우선순위화한다.

리포트 예시:
- 전체 설치율: 62%
- 배관 시스템 A: 74%
- 전기 트레이: 41%
- 오늘 신규 확인: 128개 객체
- 추론 설치: 37개 객체
- 검토 필요: 19개 객체

성공 기준:
- 현장 관리자 관점에서 오늘 확인해야 할 객체와 구역이 명확히 나온다.
- 리포트의 각 수치가 BIM GUID와 evidence frame으로 추적 가능하다.

### Phase 6. 운영화 및 UI/API 연계

검증 목표:
- 분석 결과를 다른 시스템 또는 BIM-VAMS UI류 앱에서 사용할 수 있다.

작업:
- REST/API 또는 파일 기반 result package를 정의한다.
- 3D viewer에서 다음 레이어를 표시한다.
  - LingBot 카메라 경로
  - 현장 포인트클라우드
  - BIM 모델
  - 상태별 객체 색상
  - uncertainty sphere/frustum
- 수동 보정 UI를 고려한다.
  - 기준점 추가/수정
  - 오판정 상태 수정
  - review 객체 승인/반려

성공 기준:
- 한 프로젝트의 영상 업로드부터 진척 리포트까지 재현 가능한 작업 순서가 있다.
- 자동 판정이 틀린 경우 사용자가 보정하고 재계산할 수 있다.

## 우선순위

1. Export 표준화
   - 현재 `lingbot-map` 결과를 외부 분석 파이프라인에서 읽을 수 있게 만드는 것이 첫 병목이다.

2. 수동 기준점 기반 Sim(3) 정합
   - 완전 자동 위치 인식보다 먼저 구현해야 한다. 정합이 안 되면 설치 판정은 의미가 없다.

3. BIM 객체 projection
   - "어떤 객체가 보여야 하는가"를 계산해야 detection 결과를 공정 상태로 바꿀 수 있다.

4. 객체별 evidence aggregation
   - 한 프레임 판단이 아니라 영상 전체에서 객체별 판단으로 누적해야 현장 리포트가 된다.

5. topology 기반 inferred 상태
   - BIM-VAMS의 차별점이다. 직접 보이지 않는 설치물을 BIM 연결 관계로 추론한다.

## 초기 데이터 스키마 초안

### LingBot scene export

```json
{
  "scene_id": "site-video-2026-05-19",
  "coordinate_system": "lingbot_world",
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "intrinsic": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
      "extrinsic_c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
      "confidence": 0.92
    }
  ],
  "point_cloud": {
    "path": "point_cloud_sampled.ply",
    "confidence_path": "point_confidence.npy"
  }
}
```

### BIM object status

```json
{
  "guid": "bim-object-guid",
  "status": "installed",
  "confidence": 0.84,
  "evidence": [
    {
      "frame_number": 42,
      "projected_bbox": [120, 80, 220, 150],
      "match_score": 0.78,
      "source": "projection_edge_color"
    }
  ],
  "inference": {
    "method": "direct_observation",
    "path": []
  }
}
```

## 기술 리스크

- RGB-only 재구성은 metric scale과 절대 좌표 정확도가 약할 수 있다.
- BIM과 실제 현장이 다르면 정합 RMSE가 낮아도 객체 판정이 틀릴 수 있다.
- 반복적인 배관/덕트 구조는 시각적으로 구분이 어렵다.
- occlusion이 많은 현장에서는 `not_detected`와 `unknown`을 신중히 구분해야 한다.
- 긴 영상은 drift가 누적되므로 keyframe re-anchor가 필요하다.

## 권장 MVP 범위

MVP는 다음만 한다.

1. 영상 입력
2. LingBot pose/pointcloud export
3. 수동 기준점 3개 이상으로 BIM 정합
4. BIM 객체 projection
5. 객체별 direct evidence 집계
6. MEP topology 기반 inferred 상태
7. 설계모델 기준 설치율 리포트 JSON 생성

MVP에서 제외한다.

- 완전 자동 BIM global localization
- 모바일 실시간 처리
- LiDAR 수준 정밀 치수 검증
- 안전 판단/무인 시공 의사결정
- 복잡한 웹 대시보드

## 다음 구현 순서

1. `lingbot_map/export/scene_export.py` 추가
   - inference 결과를 표준 JSON + PLY/NPY로 저장한다.

2. `lingbot_map/bim/alignment.py` 추가
   - Sim(3) solver와 alignment metric을 구현한다.

3. `lingbot_map/bim/metadata.py` 추가
   - BIM metadata JSON loader와 객체 중심/연결 그래프를 만든다.

4. `lingbot_map/bim/projection.py` 추가
   - BIM 객체를 frame camera pose로 2D 투영한다.

5. `lingbot_map/bim/progress.py` 추가
   - 객체별 evidence aggregation, status decision, topology inference, progress summary를 만든다.

6. `tools/analyze_bim_progress.py` 추가
   - CLI로 영상 결과와 BIM 데이터를 받아 진척 리포트를 생성한다.

## 1차 성공 기준

- 샘플 BIM 모델과 현장 영상에 대해 다음 파일이 생성된다.
  - `scene_export.json`
  - `alignment.json`
  - `object_statuses.json`
  - `progress_report.json`
- `progress_report.json`에는 전체/구역/시스템별 설치율이 포함된다.
- 모든 설치 판정은 GUID와 evidence frame으로 역추적 가능하다.
