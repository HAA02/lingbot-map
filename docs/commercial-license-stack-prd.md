# PRD: 상업 배포 가능 라이선스 스택으로 3단 파이프라인 갱신

- 작성일: 2026-09-15
- 상태: **초안 (엔지니어링 정책)** — 법률 자문이 아님. SPDX 핀 + 법무 리뷰 전 배포 금지
- 관련 문서:
  - `docs/scan2bim-auto-progress-prd.md` (기능 목표: 마커리스 정합·실적)
  - `docs/model-coverage-webview-prd.md` (coverage UI)
  - `docs/bim-vams-absorption-plan.md` (객체 상태·토폴로지)
  - `validate/build-vs-buy.md` (상용 3사·OSS 차용)
  - `docs/scout-bim-mapping-accuracy.md` (특징점 트래킹 부재)
  - `memory/INDEX.md` (현 제품 스펙·불변식)
- 전제: 이 저장소는 업스트림 LingBot-Map(재구성 코어) 위에 **정합·coverage 제품 레이어**가 이미 있다. 본 PRD는 그 3단을 **상업 배포가 가능한 라이선스만**으로 고정·교체하는 작업이다.

## 0. 한 줄 요약

폰 RGB 영상 → 점군·포즈(1단) → 설계 정합(2단) → 부재별 가시·커버리지(3단)를 **유지·강화**하되, 제품에 들어가는 코드·가중치·런타임은 **상업 이용이 허용된 라이선스만** 남긴다. NC/연구전용/강한 카피레프트는 빌드에서 제거하고, 대체재는 OSI 허용(Apache-2.0/MIT/BSD) 또는 LGPL 동적 격리로만 채운다.

## 1. 배경 / 왜 지금

업스트림 LingBot-Map은 Apache-2.0 스트리밍 재구성기다. 비교표에서 말한 대로 **설계(IFC/CAD) 자동 매핑·부재 커버리지는 코어 기능이 아니다.** 이 저장소는 이미 그 빈칸을 in-house로 메우고 있다.

| 단 | 이상 스택 (조사 시점) | 이 저장소 현실 | 라이선스 문제 |
|---|---|---|---|
| 1. 영상→PC | LingBot-Map / StreamVGGT / CUT3R / COLMAP | `lingbot_map/` + `realtime/inference_worker.py` | 코어 코드는 Apache-2.0. **가중치 SPDX 미기재**. VGGT 계열 런타임은 Meta 커스텀. CUT3R은 NC |
| 2. 설계 정합 | Open3D ICP / HLOC | `scan2bim/` + `build_coplay.py` (Umeyama, place_rigid, plan-match) | 자체 코드 OK. HLOC 기본 SuperPoint/SuperGlue는 **학술 가중치**. Open3D/TEASER++는 미도입 |
| 3. 보인 범위 | GEOMAPI | `realtime/coverage.html` + coverage API + `scan2bim/progress.py` | 자체 코드 OK. GEOMAPI는 MIT(PyPI). IfcOpenShell은 LGPL |

상업 배포(SaaS·온프레·납품)를 하려면 기능 PRD만으로는 부족하다. **코드 라이선스 ≠ 가중치 라이선스 ≠ CUDA EULA**를 분리해 게이트해야 한다.

기존 기능 불변식은 그대로 둔다 (완화 금지).

1. red 정합에서 자동 observed 생성 금지
2. 반복 객체에서 영상-only 자동 전역정합 확정 금지 — 후보는 명시 저장
3. 정합 전 scan/경로를 모델 위치에 임의 표시 금지 (`raw_scan_local`)

## 2. “상업용으로 쓸 수 있다”의 정의 (본 PRD 정책)

법무 확정 전 엔지니어링 기본값. 더 엄격한 사내 정책이 있으면 그쪽으로 올린다.

### 2.1 ALLOW (제품 바이너리·서버에 포함 가능)

| 등급 | SPDX | 조건 |
|---|---|---|
| A. 허용형 OSS | Apache-2.0, MIT, BSD-2/3, ISC, Unlicense, BSL-1.0 | NOTICE·저작권 고지 유지 |
| B. 약한 카피레프트 | MPL-2.0 | 수정 파일 공개. 나머지 코드는 자체 라이선스 유지 |
| C. LGPL 격리 | LGPL-2.1/3.0 | **동적 링크 또는 별도 프로세스**. 수정분은 LGPL. 본체 Apache와 정적 링크 금지 |
| D. 벤더 EULA | NVIDIA CUDA/cuDNN, 브라우저, OS | 재배포 조항 준수. 소스에 섞지 않음 |

### 2.2 REVIEW (포함 전 SPDX + 법무 GO)

- HuggingFace/ModelScope 카드에 `license:` 필드가 없거나 `other`인 가중치
- Meta Research Materials (VGGT License 등) — NC 문구가 없어도 OSI가 아니고, AUP·재배포 동일조건·비양도
- 학술 데모 가중치 (원본 SuperPoint / SuperGlue)
- 사내 `.dtdx` / DTDWebThree — OSS가 아님. **사용권 문서**가 필요
- 학습 데이터 라이선스가 가중치에 전이되는지 불명인 경우

### 2.3 DISALLOW (제품 경로에서 제거)

| 유형 | 예 | 이유 |
|---|---|---|
| 비영리 | CC BY-NC, CC BY-NC-SA, CUT3R | 상업 이용 금지 |
| 강한 카피레프트 | GPL-2/3, AGPL-3.0 (OpenMVS) | 제품 전체를 동일 라이선스로 오염시킬 수 있음 |
| 연구 전용 문구 | “research use only”, 학술 전용 가중치 | 배포·판매 불가 |
| 라이선스 없음 | LICENSE 파일 없는 연구 코드 | 기본은 저작권 전부 유보 |

**원칙:** 의심되면 넣지 않는다. 대체재가 있으면 대체한다. 없으면 해당 기능을 HOLD.

## 3. 현재 인벤토리 (2026-09-15 실측)

### 3.1 유지 후보 (이미 상업 친화)

| 구성 | 위치 | 라이선스 | 비고 |
|---|---|---|---|
| 본 저장소 제품 코드 | `realtime/`, `scan2bim/`, `tools/`, `lingbot_map/bim/` | Apache-2.0 (`LICENSE.txt`) | 제품 레이어. 재작성 불필요 |
| LingBot-Map 추론 코드 | `lingbot_map/` (aggregator/heads/layers) | Apache-2.0. DINOv2 레이어 헤더도 Apache-2.0 | `lingbot_map/` ML 코어 수정은 기존 Do-Not-Modify 범위. **라이선스 작업은 의존성·가중치·래퍼만** |
| DINOv2 | `lingbot_map/layers/*` (Meta, Apache-2.0 확인) | Apache-2.0 | 코드 내장 |
| OWL-v2 가중치 | `scan2bim/detect.py` → `google/owlv2-base-patch16-ensemble` | **Apache-2.0** (HF `license:apache-2.0`) | 선택 경로. 상업 가능 |
| Open3D | 렌더 extra (`pyproject` render) | **MIT** | 2단 정합에 승격할 1순위 |
| GEOMAPI | 미도입. PyPI `geomapi` | **MIT** (classifier `License :: OSI Approved :: MIT License`) | 3단 보강 후보. 재구성기 아님 |
| hloc 코드 | 미도입 | **Apache-2.0** | 매처 기본값(SuperPoint)은 별도 금지 |
| LightGlue | 미도입 | **Apache-2.0** (ETHZ) | SuperPoint 없이 ALIKED/SIFT와 조합 |
| COLMAP | 벤치마크 GT만 (`benchmark/`) | BSD-3-Clause | 런타임 1단 폴백으로 승격 가능 |
| RTAB-Map | 미도입 | BSD-3-Clause | RGB-D/루프클로저 옵션 |
| PyTorch / OpenCV / einops / safetensors / huggingface_hub | `pyproject.toml` | BSD-style / Apache-2.0 | 유지 |
| Three.js 웹뷰 | `realtime/*.html` | MIT | 유지 |
| viser / FlashInfer / Kaolin | vis·render extra | Apache-2.0 계열 (배포 시 SPDX 재확인) | 데모·렌더. 제품 필수 경로와 분리 권장 |

### 3.2 HOLD (상업 배포 전 확인)

| 구성 | 위험 | 조치 |
|---|---|---|
| `robbyant/lingbot-map` 가중치 (`lingbot-map.pt` 등) | HF API에 **license 태그 없음**. 코드 README만 Apache-2.0 | Robbyant에 **가중치 SPDX 서면 확인**. 없거나 other이면 제품 추론 경로 HOLD, COLMAP 폴백만 출하 |
| `lingbot-map-stage1.pt` → VGGT 양방향 추론 | VGGT는 **Meta Research Materials** (OSI 아님, 동일조건 재배포, AUP) | **제품에서 VGGT 런타임·가중치 로드 금지**. stage1은 연구 전용 extra |
| `skyseg.onnx` (JianyuanWang/skyseg) | VGGT 데모 계열. SPDX 미확인 | 실외 스카이 마스크를 **제품 기본 경로에서 제거**하거나 자체/허용 모델로 교체 |
| scout 권고 SuperPoint+LightGlue | SuperPoint/SuperGlue 원본 가중치는 학술 제한으로 알려짐 | **제품 매처는 SIFT(OpenCV) 또는 ALIKED+LightGlue만**. SuperPoint 기본값 금지 |
| Metric3D / UniDepth | 기존 PRD FR-1.3. 라이선스 카드 불명 | 도입 전 SPDX. 대안: DXF 실측 길이, ARKit/ARCore VIO (플랫폼 EULA) |
| `.dtdx` / DTDWebThree | 사내/제3자 포맷 | OSS 문제가 아니라 **사용권**. 납품 계약에 명시 |
| NVIDIA CUDA | 재배포 바이너리 | EULA. 소스 트리에 CUDA 헤더 대량 복사 금지 |

### 3.3 DISALLOW (넣지 않음 / 문서에서 제거)

| 이름 | 라이선스 | 대체 |
|---|---|---|
| CUT3R | CC BY-NC-SA 계열 | LingBot-Map 또는 COLMAP |
| StreamVGGT | 저장소 LICENSE 부재·VGGT 파생 가능성 | 도입 금지. 필요 시 SPDX 확인 후에만 REVIEW |
| OpenMVS | AGPL-3.0 | 쓰지 않음. 밀집 메쉬는 Open3D TSDF 또는 자체 unproject |
| PALMS / PALMS+ | 연구 코드, 평면도 로컬라이즈. FAB 3D BIM과 거리 | 자체 `plan_skeleton` + `coarse_match` 유지 |
| IronScan (IRON_SLAM) | 미완성·라이선스 불명 | 제외 |
| construction-diff | 부재 단위 IFC 커버리지 약함, SPDX 미확인 | 자체 coverage + (선택) GEOMAPI |
| NetVLAD 일부 구현 | 연구 가중치 혼재 | hloc의 허용 디스크립터만, 또는 생략 (시작 1탭으로 대체) |

## 4. 목표 아키텍처 (상업 스택)

기능은 기존 3단을 유지한다. 바꾸는 것은 **구현체와 라이선스 경계**다.

```text
[1단 재구성 — Apache-2.0 경로]
  폰 RGB
    ├─ 주경로: LingBot-Map 추론  (가중치 SPDX = Apache-2.0 확인 후에만)
    └─ 폴백:  COLMAP SfM (BSD) + OpenCV SIFT (Apache)
              옵션 RGB-D: RTAB-Map (BSD)
    → poses + colored PC + confidence
    스케일: DXF/알려진 길이 / (선택) 플랫폼 VIO
    금지: VGGT runtime, CUT3R, OpenMVS, skyseg 미확인 가중치

[2단 정합 — MIT/Apache + 자체 코드]
  설계: .dtdx 디코더(자체) 또는 IfcOpenShell(LGPL, 별도 프로세스)
  coarse: 기존 place_rigid + plan-match (자체, 불변식 유지)
          + (신규) Open3D FGR/ICP (MIT)
          + (신규) TEASER++ (MIT, SPDX 확인 후)
          + (옵션) hloc + ALIKED + LightGlue (Apache) — SuperPoint 금지
  출력: Sim3 / HOLD / 후보 JSON (명시 저장만)

[3단 커버리지 — 자체 + MIT 옵션]
  주경로: 기존 coverage API · coverage.html · 검수 JSON
  옵션: GEOMAPI (MIT) 어댑터 — IFC 부재 가시/완료율을 우리 status enum으로 사상
  금지: GEOMAPI가 정합을 대신하게 하지 않음 (입력 PC는 2단 통과분)
```

### 4.1 유지 / 교체 / 격리 / 추가

| 구분 | 대상 |
|---|---|
| **유지** | 재구성 코어 코드, coverage 게이트, coplay, plan-match, 불변식, `--plan-match off` byte-identical 계약 |
| **교체** | (조건부) 가중치 출처 문서화, skyseg, scout의 SuperPoint 경로 |
| **격리** | IfcOpenShell(LGPL) → `scan2bim/ifc_plugin/` 또는 별도 워커. 본체는 `.dtdx`/GLB |
| **추가** | Open3D 정합 경로, TEASER++, COLMAP 폴백, GEOMAPI 어댑터(opt-in), SBOM/NOTICE/CI |
| **금지 추가** | StreamVGGT, CUT3R, OpenMVS, VGGT, PALMS, IronScan을 제품 의존성으로 넣지 않음 |

## 5. 요구사항 (FR)

### Tier 0 — 라이선스 게이트 (본 PRD의 핵심)

- **FR-0.1 SPDX 인벤토리** — 직접 의존성·전이 의존성·가중치·ONNX·WASM·프론트 CDN을 `docs/licenses/THIRD_PARTY.md` + `NOTICE`에 기록. 각 행: 이름, 버전, SPDX, 출처 URL, 제품 경로(주/옵션/금지), 확인일.
- **FR-0.2 가중치 카드 필수** — 런타임이 받는 모든 `.pt/.onnx/.safetensors`는 HF/로컬 카드에 SPDX가 ALLOW이거나 서면 허가가 있어야 한다. 없으면 워커가 **로드 거부**.
- **FR-0.3 CI 라이선스 검사** — `pip-licenses` 또는 equivalent. DISALLOW SPDX(GPL/AGPL/CC-BY-NC*) 발견 시 fail. `--plan-match off` 회귀와 별개 잡.
- **FR-0.4 extra 분리** — `[project.optional-dependencies]`를 `product` / `research` / `render-demo`로 나눈다. 제품 서버 이미지에는 `research` 미포함.
- **FR-0.5 LGPL 격리** — IfcOpenShell은 본체에 import하지 않는다. CLI 서브커맨드 또는 RPC. 실패 시 `.dtdx`/GLB 경로만으로 제품 동작.
- **FR-0.6 고지 준수** — Apache NOTICE 병합, MIT/BSD 저작권 문구 배포물 포함. 웹 앱 크레딧 페이지 또는 `/licenses`.

### Tier 1 — 1단 재구성 (상업 경로)

- **FR-1.1 주경로 유지** — 현 `inference_worker.py` feed-forward. `lingbot_map/` 학습 코드 수정 없음.
- **FR-1.2 가중치 GO/HOLD** — Robbyant 가중치가 Apache-2.0(또는 상업 허용 서면)이면 주경로 GO. 아니면 주경로를 research extra로 강등.
- **FR-1.3 COLMAP 폴백** — 제품 플래그 `--recon colmap`. 출력 스키마는 현 poses/PC JSON과 **동일한 계약**. coverage/coplay는 재구성기 교체를 몰라야 한다.
- **FR-1.4 VGGT 차단** — stage1 체크포인트를 VGGT에 넣는 경로, `facebookresearch/vggt` import, Meta Research Materials 가중치를 제품 이미지에서 삭제.
- **FR-1.5 스카이 마스크** — 미확인 ONNX는 기본 off. 실외 필요 시 ALLOW 모델로만 재도입.
- **FR-1.6 스케일** — Metric3D는 SPDX 전 HOLD. 현 DXF/`place_rigid` 스케일 유지. 폰 VIO는 플랫폼 EULA로 별도 epic.

### Tier 2 — 2단 정합 (상업 경로)

- **FR-2.1 Open3D 정밀 정합** — 구조셸(천장/벽) 부분겹침 ICP/FGR를 `scan2bim/registration.py`에 **opt-in**. 기본 `--plan-match off` 출력과 byte-identical 유지.
- **FR-2.2 TEASER++ (MIT 확인 후)** — 전역 강건 정합. 실패 시 기존 coarse + HOLD. 그리드 탐색을 즉시 삭제하지 않음.
- **FR-2.3 상업 특징점 트래킹 (옵션)** — 키프레임에 OpenCV SIFT 또는 ALIKED+LightGlue → 상대 pose → 경량 pose-graph. SuperPoint/SuperGlue 가중치 로드 금지(테스트도 fixture 금지).
- **FR-2.4 hloc 로컬라이즈 (옵션)** — Apache 코드 + ALLOW 디스크립터만. 위치만 주고 IFC 커버리지를 대체하지 않음. 반복 MEP에서 자동확정 금지 불변식 유지.
- **FR-2.5 설계 수집** — `.dtdx` 자체 디코더 유지. IFC는 LGPL 플러그인. GLB IfcGUID 경로 유지.

### Tier 3 — 3단 커버리지 (상업 경로)

- **FR-3.1 기존 coverage 유지** — `observed` / `likely_observed` / `uncertain` / `not_observed` / `out_of_scope`. 사람 검수 JSON.
- **FR-3.2 GEOMAPI 어댑터 (opt-in)** — MIT 패키지. 입력 = 2단을 통과한 PC+IFC. 출력 = GUID별 가시/완료율 → 우리 enum 사상. GEOMAPI 실패해도 주경로 coverage 동작.
- **FR-3.3 상태 계층** — 보임 ≠ 시공됨. `not_observed`와 `absent` 분리 유지 (`validate/redteam.md`).
- **FR-3.4 OWL-v2** — Apache-2.0이므로 상업 가능. 다만 검출은 정합 힌트일 뿐 자동 실적 확정이 아님.

## 6. 비목표

- 상용 Doxel/Buildots/OpenSpace 대체 완성품 (취득은 여전히 단안 폰. PreFlight 한계 유지)
- `lingbot_map/` 학습·아키텍처 재작성
- 측량급 geo 좌표 / 법정 준공 대체
- AGPL 엔진(OpenMVS)으로 메시 품질 올리기
- CUT3R·StreamVGGT를 “성능이 좋아서” 제품에 넣기
- 라이선스 게이트를 통과시키려고 SPDX를 허위로 적기
- plan-match 게이트 임계를 상업 스택 도입 핑계로 완화하기

## 7. 단계별 실행 계획

각 Phase는 **디스크에 산출물 먼저, 측정은 그 다음**. 라이선스 문서를 측정보다 뒤에 두면 턴이 끊겨 0이 된다.

### Phase 0 — 정책 고정 (본 문서) · 예상 1~2일

**산출물**

- 이 PRD (`docs/commercial-license-stack-prd.md`)
- `docs/licenses/THIRD_PARTY.md` 스켈레톤 (주경로 직접 의존성만)
- `NOTICE` 초안

**DoD**

- ALLOW/REVIEW/DISALLOW 표가 코드 경로와 1:1
- 가중치 3종(lingbot-map / stage1 / skyseg) HOLD가 명시됨
- 법무 리뷰 체크리스트 첨부 (부록 A)

### Phase 1 — 제품 이미지에서 DISALLOW 차단 · 예상 3~5일

**작업**

1. `pyproject.toml` extra 분리: `product`, `research`, `render-demo`
2. 서버 기본 import 경로에서 VGGT·skyseg·미확인 ONNX 제거 또는 `--research` 가드
3. CI: SPDX fail-closed
4. `/licenses` 또는 배포 tarball에 NOTICE

**DoD**

- `pytest tests/ -q` 기준선 유지 (현 369 passed 계약. 숫자 자체보다 **0 skip 핵심 기하 테스트**)
- `--plan-match off` byte-identical
- 제품 extra로 설치 시 DISALLOW 패키지 0

### Phase 2 — 2단 상업 정합 opt-in · 예상 1~2주

**작업**

1. `scan2bim/registration.py`에 Open3D ICP/FGR 경로 (`--refine open3d`, 기본 off)
2. TEASER++ MIT SPDX 확인 후 전역 후보 (실패→HOLD)
3. (옵션) SIFT 또는 ALIKED+LightGlue 키프레임 그래프 — SuperPoint 없음
4. 기존 place_rigid / plan-match 무회귀

**DoD**

- 신규 테스트: Open3D 경로 on/off 출력 계약
- 불변식 테스트 유지 (모호 → HOLD, transform 밀반출 0)
- 라이선스 CI green

### Phase 3 — 3단 GEOMAPI opt-in + IFC 격리 · 예상 1~2주

**작업**

1. IfcOpenShell을 플러그인 프로세스로만 (LGPL)
2. GEOMAPI 어댑터: IFC GUID → coverage status 사상 표
3. `coverage.html`은 기존 API 유지. GEOMAPI는 서버 내부 구현 상세

**DoD**

- `.dtdx`만 있는 현장은 플러그인 없이 현 coverage 동작
- GEOMAPI on이어도 red 정합에서 observed 승격 0
- MIT 고지 NOTICE에 추가

### Phase 4 — 1단 폴백·가중치 분기 · 예상 1~2주 (가중치 회신 대기와 병렬)

**작업**

1. Robbyant 가중치 SPDX 회신
   - GO: 주경로 유지, 카드·NOTICE에 버전 핀
   - HOLD: 제품 기본 `--recon colmap`, LingBot은 research extra
2. COLMAP 출력을 현 업로드 산출물 스키마에 맞추는 어댑터
3. 데모 렌더(`render_demo_format.sh`)는 render-demo extra. 제품 커버리지와 분리

**DoD**

- 동일 업로드에서 recon 백엔드만 바꿔도 coplay URL 계약 유지
- VGGT/CUT3R/OpenMVS가 requirements에 없음
- smoke: `tools/coverage_web_smoke.py` (서버 가동 시)

### Phase 5 — 문서·비교표 정본화

- README “상업 배포” 절: 코드 Apache-2.0, 가중치는 카드 따름
- `validate/build-vs-buy.md`에 본 PRD 링크
- scout 문서의 SuperPoint 권고를 **상업 경로에서는 ALIKED/SIFT**로 정정 주석

## 8. 검증 / 회귀

기존 게이트를 라이선스 작업으로 깨지 않는다.

| 게이트 | 명령 | 계약 |
|---|---|---|
| 단위 | `.venv/bin/python -m pytest tests/ -q` | 기준선 통과, 핵심 기하 0 skip |
| plan-match 무회귀 | `--plan-match off` sha256 | byte-identical |
| 강건성 | `tools/check_plan_match_robust.py` | 확정오답 0 유지 |
| 웹 | `tools/coverage_web_smoke.py` | 기존 픽스처 불변 (`upload_1779439687108` 등 수정 금지) |
| 라이선스 | CI SPDX | DISALLOW 0 |
| 가중치 | 워커 시작 시 카드 검사 | SPDX 없으면 거부 |

## 9. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| LingBot 가중치가 Apache가 아님 | 1단 주경로 강등, COLMAP 출하. 코드는 오픈소스 기여로 남김 |
| COLMAP은 실시간 ~20 FPS가 아님 | 제품 SLA를 “업로드 배치 재구성”으로 명시. 실시간은 가중치 GO일 때만 주장 |
| Open3D ICP가 반복 MEP에서 틀린 방 | 불변식 ② — 자동확정 금지, 후보만 저장. Open3D는 refine이지 global lock이 아님 |
| GEOMAPI가 우리 coverage와 상태명 충돌 | 사상표 고정. 외부 라이브러리 용어를 UI에 직접 노출하지 않음 |
| LGPL 정적 링크 실수 | 플러그인 프로세스 + CI grep `import ifcopenshell` in `realtime/` fail |
| SuperPoint를 “성능”으로 다시 넣음 | FR-2.3 테스트가 해당 가중치 파일 해시를 거부 |
| 라이선스 CI를 통과시키려 패키지 라이선스 메타 위조 | 금지. 제외하거나 HOLD |
| 단안 스케일 한계는 라이선스와 무관 | PreFlight 유지: 360°/VIO는 별 epic. 본 PRD가 그 한계를 지우지 않음 |

## 10. 성공 지표

- 제품 기본 설치(`pip install .[product]`)에 DISALLOW SPDX 0, REVIEW 항목은 HOLD 문서화
- 가중치 로드 경로 100%가 SPDX 또는 서면 허가에 묶임
- 기존 기능 회귀: pytest + plan-match off byte-identical + coverage smoke
- 3단 모두에 **허용 라이선스 구현체**가 존재 (1단: LingBot 또는 COLMAP, 2단: 자체+Open3D, 3단: 자체±GEOMAPI)
- README/NOTICE만 읽고 외부 감사자가 주경로를 재현할 수 있음

## 11. 결정 기록 (이 PRD가 고정하는 것)

1. **상업 스택 = OSI 허용 + LGPL 격리 + 벤더 EULA**. NC·AGPL·라이선스 없음·학술 가중치는 제품 제외.
2. **기능 3단은 유지**한다. GEOMAPI/Open3D는 표준 부품으로 **opt-in**이지 저장소 재시작이 아니다.
3. **LingBot-Map 코드는 유지**, 가중치는 별도 GO. stage1→VGGT는 제품에서 절단.
4. **HLOC는 Apache이지만 SuperPoint 기본값은 쓰지 않는다.**
5. **IfcOpenShell은 쓰되 본체와 프로세스 분리.**
6. **비교표의 StreamVGGT/CUT3R/PALMS/IronScan/OpenMVS는 제품 로드맵에서 삭제.**

---

## 부록 A — 법무 리뷰 체크리스트 (사람)

이 문서는 법률 의견이 아니다. 배포 전 확인.

- [ ] `LICENSE.txt` Apache-2.0이 사내 배포 주체(법인)와 기여 정책에 맞는지
- [ ] Robbyant 가중치 상업 이용·재배포·SaaS 제공 가능 여부 (서면)
- [ ] DINOv2 내장 코드 NOTICE 충족
- [ ] CUDA 재배포 바이너리 EULA
- [ ] `.dtdx`/DTDWeb 포맷 사용권
- [ ] IfcOpenShell LGPL 동적 격리 방식이 사내 해석과 일치하는지
- [ ] 고객 현장 영상 학습 재사용 금지 (제품 기본: 학습 안 함)
- [ ] 수출 규제 (일부 지역 GPU 추론)

## 부록 B — 비교표 (상업 배포 관점으로 수정)

요구: 폰 영상→PC-like / 설계 매핑 / 영상에 보인 설계 범위. **제품 포함 여부 = 본 PRD 정책.**

| 이름 | SPDX (코드) | 가중치 | 영상→PC | 설계 매핑 | 보인 부분 | 제품 |
|---|---|---|---|---|---|---|
| LingBot-Map 코드 | Apache-2.0 | **카드 미기재 → HOLD** | ✅ 스트리밍 | ❌ 코어 | ❌ 코어 | 가중치 GO 시 1단 주경로 |
| 이 저장소 제품 레이어 | Apache-2.0 | n/a | 워커로 1단 호출 | △ 자체 scan2bim | △ coverage UI | **유지** |
| COLMAP | BSD-3 | n/a | ✅ 오프라인 | ❌ | ❌ | 1단 폴백 |
| RTAB-Map | BSD-3 | 의존성 재확인 | ✅ SLAM | ❌ | ❌ | RGB-D 옵션 |
| Open3D | MIT | n/a | ❌ 라이브러리 | ✅ ICP/FGR | △ 직접 | **2단 채택** |
| TEASER++ | MIT (도입 시 재확인) | n/a | ❌ | ✅ 전역 | ❌ | 2단 옵션 |
| hloc + LightGlue + ALIKED | Apache-2.0 | ALLOW만 | △ 로컬라이즈 | △ 위치 | ❌ 부재 커버리지 아님 | 2단 옵션. SuperPoint 금지 |
| GEOMAPI | MIT | n/a | ❌ 입력 PC 필요 | ✅ IFC↔PC | ✅ 부재 가시 | **3단 opt-in** |
| IfcOpenShell | LGPL | n/a | ❌ | ✅ IFC 파서 | ❌ | **격리 플러그인** |
| OWL-v2 | Apache-2.0 | Apache-2.0 | ❌ | △ 힌트 | ❌ | 옵션 유지 |
| StreamVGGT | 미확인/파생 위험 | 미확인 | ✅ | ❌ | ❌ | **제외** |
| CUT3R | NC | NC | ✅ | ❌ | ❌ | **제외** |
| OpenMVS | AGPL | n/a | ✅ 밀집 | ❌ | ❌ | **제외** |
| VGGT (Meta) | Research Materials | 동일 | ✅ | ❌ | ❌ | **제외** (stage1 경로 절단) |
| PALMS+ | 연구 | 미확인 | ✅ | △ 평면도 | △ 위치 | **제외** (자체 plan-match) |
| IronScan | 미확인 | 미확인 | △ | △ | △ | **제외** |
| Doxel/Buildots/OpenSpace | 상용 구독 | n/a | △ 360° | ✅ | ✅ | Buy 검토 (`validate/build-vs-buy.md`). 본 스택 대체 아님 |

## 부록 C — 코드 접점 (구현 시 손대는 곳)

| 작업 | 시작 파일 | 주의 |
|---|---|---|
| extra 분리 | `pyproject.toml` | `lingbot_map` 패키지 include 유지 |
| 추론 가드 | `realtime/inference_worker.py` | 가중치 SPDX 검사 |
| 정합 opt-in | `scan2bim/registration.py`, `tools/build_coplay.py` | 기본 off, 무회귀 |
| 검출 | `scan2bim/detect.py` | OWL-v2는 ALLOW. SuperPoint 넣지 말 것 |
| coverage | `realtime/server.py` coverage API | 상태 enum 변경 금지 |
| IFC 플러그인 | 신규 `scan2bim/ifc_plugin/` (제안) | `realtime/`에서 import 금지 |
| GEOMAPI | 신규 `scan2bim/geomapi_adapter.py` (제안) | optional extra |
| 고지 | 신규 `NOTICE`, `docs/licenses/THIRD_PARTY.md` | Phase 0 |
| 금지 픽스처 | `realtime/_uploads/upload_1779439687108.*` 등 | 수정 금지 |

## 부록 D — 업스트림 확인 메일 초안 (가중치)

수신: Robbyant / HuggingFace `robbyant/lingbot-map` 유지자

> 본 저장소 코드는 Apache-2.0입니다. 상업 제품(SaaS 및 온프레 납품)에서 `lingbot-map.pt` / `lingbot-map-long.pt` / `lingbot-map-stage1.pt`를 추론 전용으로 사용·재배포할 수 있는지 SPDX 라이선스를 모델 카드에 명시해 주십시오. stage1을 facebookresearch/vggt에 로드하는 경로의 상업 이용 가능 여부도 함께 확인해 주십시오.

회신 전 제품 기본값은 **COLMAP 폴백 준비 + LingBot 경로는 research extra**.
