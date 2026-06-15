# PRD: 마커리스 영상→설계모델 자동 매핑 및 일일 시공 실적등록 (scan2bim auto-progress)

- 작성일: 2026-06-15
- 브랜치: `feat/scan2bim-auto-progress`
- 관련 문서: `docs/report/report-video-ifc-automapping-20260615.html`(기법 조사), `docs/scan-to-model-improvement-prd.md`(현 정합), `docs/glb-auto-mapping-review.md`(좌표 규약)
- 대상 모델: `models/Gasan_7F/*.dtdx`(FAB 7층, 6분야), 뷰어 참고: `/run/media/iaan/1TB-WD/Github/DTDWeb/DTDWebThree`

## 0. 한 줄 요약

현장에서 사람이 **도면을 보지 않고 영상만 촬영**하면, 그 영상을 **마커(QR/AprilTag) 없이** 설계모델(.dtdx/IFC)에 자동 정합하고, 영상에 실제로 찍힌 요소를 **매일 자동으로 시공 실적(미시공/진행중/완료)으로 등록**한다. 텍스트(OCR) 없이 **색상 + 설계파일 + 구조 geometry**만으로 처리한다.

## 1. 배경 / 목표

### 프로젝트 핵심 자산 (유지)
lingbot-map의 핵심은 **영상→고품질 점군 재구성 + 카메라 위치/방향(pose)**이다. 이 자산은 잘 작동하며, 본 작업의 토대다.

### 목표
1. 영상만으로 설계모델 위에 **위치·회전·스케일 자동 정합** (마커리스).
2. 정합 결과로 **요소별 시공 상태(미시공/진행중/완료)**를 산출해 **일일 실적 원장**(4D)으로 누적.
3. **텍스트가 없는 현장**을 가정 — 라벨/표찰에 의존하지 않고 **색상 + 설계 + 구조**로 처리.
4. 시각화는 `DTDWebThree`(.dtdx Three.js 뷰어) 엔진 위에 점군·상태·실적 레이어를 얹어 제공.

### 비전
"도면 없이 영상만으로 매일 자동 시공검증/진척." 현장 작업자는 촬영만, 시스템이 설계 대조·실적 등록을 수행한다.

## 1.5 핵심 가설과 검증 (디코딩 근거)

`.dtdx` 직접 디코딩(2026-06-15)으로 확인:

- **설계가 계통 색상을 보유** — FXX(소방) material.diffuse = `[1,0,0]빨강`(소화) · `[0,0,1]파랑` · 회색. 실영상의 빨강 배관과 **직접 대응**.
- 6파일이 **분야별 사전 분리** — AXX건축 · SXX구조 · EXX전기 · FXX소방 · HXX공조 · PWW배관.
- 요소별 **IfcGUID**(`attr`) + **연결 토폴로지**(`connector` 609) + per-mesh material index.
- `DTDWebThree` = 요소별 **개별메시 + MeshRegistry(GUID→mesh)** → 상태 색칠 즉시 가능, 선택·속성패널·다분야 머지·대용량 메모리관리 완비.

→ **가설 "텍스트 없이 색상+설계만으로 시맨틱 매핑 가능"은 데이터로 뒷받침됨.** FAB은 계통 색코딩이 엄격해 색=계통 가정이 특히 강하다.

## 2. 현 상태 / 원인 분석 (코드 검증)

현 GLB 자동배치(`realtime/server.py:_auto_place_candidates`)가 구조적으로 부적합한 이유:

1. **스케일 미해결** — 단안 재구성은 up-to-scale(비metric, `head_act.py` 상대 depth). 자동배치는 `scale=1.0` 고정 → 거리 안 맞음.
2. **반복기하 모호성** — 동일 배관 다수 위 영상-only 전역 위치찾기는 다중 해. 프로젝트 불변식이 이미 금지.
3. **매칭 대상 부족** — 모델이 배관/덕트 골격만(벽·바닥 없음) → 방 전체 스캔 점 대부분 out_of_scope.
4. **대응점 UX 막힘** — GLB-native 모델은 proxy가 비어 대응점 픽이 동작 안 함(이번 세션 수정), 점군 픽 난해.

### 핵심 통찰
정합 코어만 구조적으로 잘못됐고, **재구성·객체모델·coverage·뷰어는 목표와 부합**한다. → 프로젝트 재시작이 아니라 **정합 레이어 교체**.

## 3. 설계 원칙 — 유지 / 교체 / 이동

| 구분 | 대상 |
|---|---|
| **유지** | 영상→점군+포즈 재구성(핵심 자산), 객체모델 추상(GUID/category/system/center/bbox), coverage 분석(=실적 토대), 검증 게이트(green/yellow/red, 명시 저장) |
| **교체** | `_auto_place_candidates` 그리드탐색 + scale=1.0 + repetition 휴리스틱 |
| **이동** | 모델 소스 GLB → **.dtdx/IFC**(IfcOpenShell), 시각화 셸 → **DTDWebThree** |
| **추가** | 메트릭화, 구조셸 global registration, 색/직경/토폴로지 시맨틱 매칭, 일일 실적 원장 |

### 불변식 (완화 금지)
- 신뢰도 미달(red) 정합에서 자동 실적 확정 금지.
- 모호 시 자동저장 금지 — coarse prior/확인 필요(마커 아님).
- 정합 전 점군을 모델 위치에 임의 표시 금지.

## 4. 요구사항 (FR)

### Tier 1 — 데이터 기반
- **FR-1.1 .dtdx/IFC 수집** — `.dtdx`(msgpack: material/mesh/linkMesh/attr/connector) 및 IFC(IfcOpenShell)를 요소별 `{guid, category, system, discipline, mesh, material_color, connector}` 객체모델로 적재. 분야는 파일 단위로 태깅.
- **FR-1.2 설계 색맵 생성** — material.diffuse × 분야 → "분야·색 → 계통" 사전. 회색/무채색 요소는 색 미사용 플래그.
- **FR-1.3 메트릭 스케일 복원** — Metric3D v2/UniDepth로 키프레임 metric depth → 재구성 글로벌 scale 보정. 보조: 알려진 기준 길이 1개. (현 scale=1.0/9종탐색 hack 대체)

### Tier 2 — 정합 엔진 (교체 핵심)
- **FR-2.1 구조셸 전역 정합** — 스캔의 비배관 면(슬래브·보·벽)을 SXX/AXX geometry에 TEASER++(또는 Open3D FGR)+ICP로 정합 → 카메라 포즈. 부분겹침·시공중 강건.
- **FR-2.2 색-계통 시맨틱 매칭** — 스캔 점군 RGB를 HSV 색클러스터로 분할 → 동색 설계 계통군과 대응. 채도 높은 코드색(소화=빨강) 우선.
- **FR-2.3 시그니처 정합** — 색 + 직경 + `connector` 토폴로지 조합으로 반복 인스턴스 확정(모호성 해소).
- **FR-2.4 신뢰도/모호 게이트** — inlier·RMSE·다중가설 top-K. 모호 시 coarse prior(시작 1클릭 또는 전날 place-recognition)로 tiebreak, 자동확정 보류.

### Tier 3 — 실적 엔진
- **FR-3.1 요소 상태 분류** — 관측 점/색 있음+마감외형=완료 · 위치 비었음=미시공 · 부분/미마감=진행중. v1은 present/absent부터, 트레이드별 "완료" 정의 후 상태 고도화.
- **FR-3.2 일일 4D 원장** — 요소(GUID)별 `{date, status, evidence_frame, confidence}` 누적, day-over-day diff("오늘 신규 완료 N").
- **FR-3.3 마커리스 크로스데이 일관성** — 매일 동일 설계모델에 재정합. 옵션 place-recognition(NetVLAD/hloc)로 coarse 구역 자동 회수.

### Tier 4 — 시각화 (DTDWebThree 조립)
- **FR-4.1 다분야 로드 + 계통색 모드** — 6분야 `.dtdx` 머지(MultiFileMerger), material 색 = 계통색 토글.
- **FR-4.2 점군·경로 오버레이** — 정합된 점군(THREE.Points) + 카메라 경로를 모델 좌표에 표시(coverage.html 로직 이식).
- **FR-4.3 요소 상태색 오버라이드** — MeshRegistry(GUID→mesh)로 observed/진행중/미시공/완료 색칠.
- **FR-4.4 실적 패널** — 속성패널에 요소 실적·날짜·증거프레임, `connector` 계통 그래프 표시.

## 5. 비목표 (제외 범위)

- **OCR/텍스트 인식** — 텍스트 없는 현장 가정. 라벨 의존 배제(향후 보조 옵션으로만).
- 색 자동 작도(scan→IFC 신규 생성) — 본 작업은 *기존 설계에 대조/실적*이지 모델 생성 아님.
- 실내 cm급 절대정밀(측량 대체) — 목표는 "요소 단위 실적 신뢰판정".
- 다층/전건물 동시 — v1은 단일 층(Gasan 7F) 범위.
- ML 코어(`lingbot_map/`) 변경.

## 6. 단계별 실행 계획

1. **PoC-A 시각화 셸** — DTDWebThree에 6분야 로드 + 계통색 + (더미)점군 오버레이 한 화면. *되는 그림* 확보.
2. **PoC-B 데이터** — IfcOpenShell/.dtdx 수집 → 객체모델 + 색맵 + connector 그래프.
3. **PoC-C 메트릭화** — Metric3D로 스캔 재스케일 검증.
4. **PoC-D 정합** — SXX/AXX 구조셸에 TEASER++/ICP 전역 정합 1건 성공(마커리스).
5. **PoC-E 색-계통 매칭** — 빨강 소화부터 색클러스터↔계통 바인딩.
6. **MVP** — 요소 present/absent 일일 실적 원장 + DTDWebThree 상태색·실적패널.

## 7. 검증 / 회귀

- 기존 `tools/coverage_web_smoke.py` 회귀 유지(무힌트 경로 불변 — 이미 PASS).
- 신규: 정합 정확도(구조셸 RMSE/inlier), 색-계통 매칭 정밀/재현, 실적 일치(수동 라벨 대비), 크로스데이 재정합 일관성.
- 헤드리스 웹뷰(playwright + Edge) — DTDWebThree 로드/오버레이/상태색 렌더 검증.

## 8. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| 회색/유사색 요소는 색 구분 불가 | 직경 + connector 토폴로지 + 분야파일로 보완 |
| 조명/화이트밸런스로 스캔 색 변동 | 채도 높은 색 우선, 미묘색은 기하 의존 |
| 첫날 전역 위치찾기(반복/대칭) 최난 | 구조셸 정합 + coarse prior(1클릭/전날 place-recognition) |
| 진행중 vs 완료 판정 미성숙 | v1 present/absent부터, 트레이드별 완료 정의 후 고도화 |
| revitInfo/Pset 파일마다 희소 | 분야(파일)+색+GUID+connector만으로 운용 |
| 대용량(AXX 17MB) 성능 | DTDWebThree 메모리관리·LOD 활용 |

## 9. 성공 지표

- 마커리스 전역 정합 성공률 ≥ 90%(구조셸 충분 노출 시), 모호 자동확정 0%.
- 색-계통 매칭으로 반복 배관 인스턴스 확정 정확도(수동 대비) ≥ 90%.
- 요소 실적(present/absent) 일치율 ≥ 95%, 일일 자동 갱신.
- "영상 업로드 → 실적 원장·시각화"가 사람 도면조작 없이 완결.

## 10. 부록 — 오픈소스 / 참고

- IfcOpenShell(LGPL) · Metric3D v2 · TEASER++(MIT) · Open3D(MIT) · (옵션) hloc/NetVLAD.
- `.dtdx` 포맷: sig 0xFF09 + ver 0x200000 + msgpack(material/mesh/linkMesh/attr/connector/revitInfo) + binary.
- DTDWebThree: Three.js r160, @msgpack/msgpack, vite :3333, `src/model/DTDXLoader.js`·`MaterialManager.js`·`InstancedModelManager.js`(useIndividualMeshes)·`core/MeshRegistry.js`·`selection/SelectionManager.js`·`optimization/MultiFileMerger.js`.
- 상세 기법 비교: `docs/report/report-video-ifc-automapping-20260615.html`.
