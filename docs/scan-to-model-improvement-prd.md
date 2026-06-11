# PRD: scan-to-model 자동정합 · GUID 매핑 · coverage 정확도 개선

작성일: 2026-06-11
대상: `lingbot-map` 모델 기준 영상 evidence 기능 (`model-coverage-webview-prd.md`의 후속 개선)

## 0. 한 줄 요약

현재 모델 기준 coverage는 (a) 매 업로드마다 사람이 대응점 6개를 손으로 찍어야 정합되고, (b) GUID 매핑이 0%라 실제 GLB mesh 대신 PAG proxy로만 판정한다. 코드를 직접 검증한 결과 이 두 가지의 핵심 원인은 **데이터 한계가 아니라 구현 버그/미완성**이다. 이 문서는 검증된 원인과 단계별 개선안을 정의한다.

## 1. 문제 정의

| 증상 | 현재 상태 | 비용 |
| --- | --- | --- |
| 정합이 매번 수작업 | `upload_1779439687108`은 anchor 0개·자동후보 0개 → coverage.html에서 6클릭 필수 | 업로드 1건당 사람 개입, 처리량 천장 |
| GUID 매핑 0% | PXX manifest `guid_mapping_ratio=0.0`, `coverage_geometry_source=pag_proxy`, `glb_role=visual_context_only` | 실제 배관 mesh 형상이 아니라 PAG 좌표 proxy로만 coverage 색칠 |
| 자동매핑 부재 | 영상-only 전역 정합은 의도적으로 막혀 있으나, 그 대안 자동화도 미구현 | 반복 촬영·다수 업로드 확장 불가 |

## 2. 비판적 원인 분석 (코드 검증 결과)

각 원인은 추정이 아니라 소스/데이터로 확인했다. "유지(설계상 옳음)"와 "수정(버그/갭)"을 구분한다.

| # | 원인 | 분류 | 증거 | 영향 |
| --- | --- | --- | --- | --- |
| R1 | GLB 식별자 추출 키 불일치 | **버그** | `realtime/server.py:383` 가 `("guid","GUID","ifcGuid","revitId","RevitId")`만 탐색. 실제 `PXX.glb` extras 키는 `UniqueId`·`ElementID` | GLB의 268개 식별자를 통째로 무시 → matched=0 → 0% → proxy 강제 |
| R2 | "direct" 임계 0.60 + 하이브리드 미지원 | **설계 갭** | `server.py:490,502` — `mapping_ratio>=0.60`만 `glb_guid_mesh`, 아니면 전부 proxy. 부분 매핑을 mesh+proxy 혼합으로 못 씀 | R1 수정 후에도 52.8%는 0.60 미만 → 여전히 proxy_only |
| R3 | 자동 후보 생성기 미구현 | **미완성** | PRD FR-03A의 trajectory descriptor→geometry descriptor→BIM subgraph RANSAC→ICP 흐름이 코드에 없음. `_pose_descriptor`(server.py:1189)·`_model_repetition_risk`(1214)는 경고/진단만 생성, 기하 후보 0개 | anchor 0~3개면 `needs_anchor`로 끝, 매번 수작업 |
| R4 | 재구성 ICP가 scan→모델에 미연결 | **자산 미활용** | `realtime/registration.py:68 icp_refine_rigid`·`112 chain_windows`가 이미 존재하나 윈도우 스티칭 전용. `server.py`에서 scan-to-model에 호출 0건(grep) | 거친 정합을 표면 ICP로 정밀화하는 검증된 코드를 방치 |
| R5 | cross-upload prior가 preview 전용 | **설계 갭** | `_previous_alignment_candidates`(server.py:1330)는 같은 모델의 검증된 green/yellow alignment를 `can_save=false`로만 반환 | 같은 zone 반복 촬영 시 이전 정합을 자동 부트스트랩 못 함 |
| R6 | 90% 게이트가 수동검수 100% 강제 | **처리량 제약** | report `manual_review_required` MVP 고정 `true`, `passed_90`은 수동 기록 없으면 항상 false | 자동 품질이 아무리 좋아도 1건씩 사람 검수, 확장 불가 |
| R7 | 재구성 좌표 스케일/원점 미고정 | **미검증** | 단안 재구성은 스케일 모호. `_solve_scan_to_model_alignment`는 매번 Sim(3) scale을 새로 추정 | 자유도↑ → 반복배관 환경에서 정합 모호성↑ |

### 핵심 통찰

- **R1은 검증된 한 줄 버그다.** `PXX.glb` 노드 268개의 `UniqueId`는 `pxx_pag_export.json`의 `guid`와 **267/506 = 52.8%** 정확히 일치한다(같은 Revit UniqueId 포맷, 예: `c537c446-1d8f-4caa-838e-4b3c2c734134-000f767b`). 키만 추가하면 매핑 0% → 52.8%로 즉시 회복된다.
- **설계의 보수성(R3의 "영상-only 자동확정 금지")은 옳다.** PXX는 순환수 배관 264개가 반복되어 영상만으로는 전역 위치를 단정할 수 없다. 따라서 개선 목표는 "완전 자동"이 아니라 **사람 개입을 6클릭 → 1힌트(zone 선택) + 자동 후보 + 검수**로 줄이는 것이다.

## 3. 설계 원칙 — 유지할 것 / 바꿀 것

유지(불변):
- 반복배관 환경에서 영상-only 무제약 자동확정 금지. 후보는 사람 anchor/marker로 검증 후에만 `observed` 승격.
- `red` 정합에서 자동 `observed` 생성 금지. (현재 stale red coverage 507개 전부 `not_observed` — 정상)
- 최종 90% 판정에 사람 evidence 검수 게이트 유지(단, 강도는 R6에서 재조정).

바꿈:
- "GUID 0% / proxy only"는 데이터 한계가 아니라 버그다 → 매핑된 객체는 실제 GLB mesh로 coverage 계산.
- "매 업로드 6클릭 수작업"은 유일 경로가 아니다 → coarse prior + 기하 후보 + ICP로 클릭 수 최소화, 반복 촬영은 prior 자동 부트스트랩.

## 4. 개선 요구사항 (FR)

### Tier 1 — 즉시 (검증된 버그/저위험·고가치)

#### FR-A1. GLB 식별자 키 확장
- `_extract_glb_names`의 extras 탐색 키에 `UniqueId`, `ElementID`(및 대소문자 변형 `uniqueId`, `elementId`) 추가.
- 수용 기준: `PXX` manifest `guid_mapping_ratio >= 0.5`, `guid_mapped_count >= 260`. 회귀 테스트로 0% 재발 방지.

#### FR-A2. 하이브리드 coverage geometry + 임계 재정의
- 매핑된 GUID는 GLB mesh, 미매핑은 PAG proxy를 함께 쓰는 혼합 형상 지원.
- `coverage_geometry_source`를 단일값에서 `{mapped: glb_mesh, unmapped: pag_proxy}` 혼합 모델로 확장. `glb_role`은 매핑률>0이면 `partial_coverage_geometry`.
- 0.60 단일 임계 폐지. 대신 객체 단위로 "이 객체에 GLB mesh가 있으면 mesh, 없으면 proxy".
- 수용 기준: PXX(52.8% 매핑)에서 267개는 mesh 기준, 나머지는 proxy 기준으로 coverage가 계산되고 viewer에 구분 표시된다.

#### FR-A3. ICP 표면 정밀화 연결
- 사용자가 거친 대응점(4개)으로 Sim(3)를 잡은 뒤, 정합된 scan 점군을 모델 mesh/proxy 표면에 `registration.icp_refine_rigid`(기존 자산)로 point-to-point/plane 보정.
- ICP는 RMSE를 낮추는 방향일 때만 채택, 발산 시 원본 Sim(3) 유지.
- 수용 기준: 동일 대응점 입력에 대해 ICP 적용 후 `rmse_m`이 같거나 감소하고, `quality`가 하락하지 않는다.

#### FR-A4. cross-upload prior 승격
- 같은 모델·같은 zone에서 검증된 green/yellow alignment가 있으면, 신규 업로드의 자동 후보로 제시하고 기하 corroboration(bounds overlap, trajectory 호환) 통과 시 `can_save=true`로 승격.
- 단, 반복배관 위험이 high면 여전히 anchor 1개 이상 요구.
- 수용 기준: zone-A에서 1건을 수동 정합·저장한 뒤, zone-A의 다른 업로드는 anchor 0~1개로 후보가 1개로 좁혀진다.

### Tier 2 — 자동 후보 엔진 (FR-03A 실제 구현)

#### FR-B1. Coarse prior로 search space 축소
- 사용자가 level/zone/system(또는 모델 위 1점)을 선택하면 candidate 객체를 507 → 해당 subset(수십 개)으로 제한.
- 수용 기준: zone 선택 시 `_model_repetition_risk`의 local subset 위험도가 재계산되어 high→medium/low로 떨어질 수 있다.

#### FR-B2. 기하 디스크립터 + RANSAC Sim(3) 후보
- 축소된 subset 내에서: LBP4 점군의 주방향/선형(pipe axis)·평면 구조 디스크립터 ↔ BIM proxy 축 후보를 RANSAC으로 매칭, Sim(3) 가설 생성.
- 점수는 RMSE 단독이 아니라 visibility·frame support·geometry consistency·negative evidence 합산(기존 coverage 스코어와 동일 축).
- 1등-2등 margin이 작거나 subset에 반복 위험이 남으면 `ambiguous_alignment`.
- 수용 기준: 유사객체 반복 subset에서 후보 2개 이상이면 자동확정하지 않고 `ambiguous_alignment` 반환. anchor 1~2개 추가 시 1개로 수렴.

#### FR-B3. 후보 비교/승인 UI
- 후보별 confidence, 예상 카메라 경로, 대표 keyframe, 매칭 객체 목록을 나란히 비교.
- `can_save=true` 후보만 "후보 저장" 노출. 승인 전 후보는 coverage를 `likely/uncertain`까지만.

#### FR-B4. 마커 기반 결정적 정합 (계획 촬영용)
- AprilTag/QR를 알려진 BIM 좌표에 두면 영상에서 자동 검출 → 결정적 alignment.
- 가장 신뢰도 높은 자동 경로. 사전 계획이 가능한 현장에 권장.
- 수용 기준: 마커 2개 검출 시 사람 클릭 없이 green/yellow 정합 생성.

### Tier 3 — 처리량 / 품질 정합

#### FR-C1. 자동 게이트 통과 시 수동검수 샘플링
- 자동 게이트(green 정합 + support 충분 + observed traceability ≥90%)가 통과하면, 수동 검수를 100% → 표본(예: observed 객체 10% 또는 N개)으로 완화하는 모드 추가.
- 안전장치: 샘플에서 불일치 1건이라도 나오면 전수 검수로 격상.
- 수용 기준: `passed_90`을 "자동 게이트 + 샘플 검수 통과"로 달성 가능. 단 이 모드는 명시적 opt-in.

#### FR-C2. 재구성 metric 스케일/원점 고정
- 단안 재구성 스케일을 기준(층고/측정 객체/모델 metric depth)으로 1회 고정해 Sim(3)→SE(3)(scale 고정)로 자유도 축소.
- 수용 기준: 동일 업로드 재정합 시 추정 scale 분산이 임계 이하. scale이 기대 범위를 벗어나면 경고.

## 5. 비목표 (유지되는 제외 범위)

- 무제약 영상-only 전역 BIM localization의 **자동 확정** (Tier 3에서도 금지 유지, 후보 제시까지만).
- 실사 texture baking, Gaussian Splatting production viewer.
- 시공 완료/미시공 확정 판정.

## 6. 단계별 실행 계획

| Phase | 범위 | 완료 기준 |
| --- | --- | --- |
| P1 | FR-A1, FR-A2 | PXX 매핑률 ≥0.5, 267객체 mesh 기준 coverage, viewer에 mesh/proxy 구분 |
| P2 | FR-A3, FR-A4 | ICP로 RMSE 비악화, zone 반복 촬영 prior 자동 부트스트랩 |
| P3 | FR-B1, FR-B2, FR-B3 | coarse prior 축소 + RANSAC 후보 + 모호성 게이트 + 후보 UI |
| P4 | FR-B4 | 마커 검출 결정적 정합 |
| P5 | FR-C1, FR-C2 | 샘플 검수 모드, metric 스케일 고정 |

## 7. 검증 / 회귀 테스트 (`tools/coverage_web_smoke.py` 확장)

- GUID 매핑 회귀: `/api/models`의 PXX `guid_mapping_ratio >= 0.5`, `guid_mapped_count >= 260` (R1 재발 차단).
- 하이브리드 coverage: 매핑 객체는 mesh 기준, 미매핑은 proxy 기준 status가 생성됨.
- ICP 안정성: 동일 대응점에 ICP 적용 후 `quality` 하락 없음.
- prior 자동후보: zone 정합 저장 후 같은 zone 신규 업로드 후보 수렴.
- 기존 안전 게이트 유지: red→observed 금지, 정합 없는 업로드 `needs_alignment`·report `blocked`.

## 8. 리스크와 대응

| 리스크 | 영향 | 대응 |
| --- | --- | --- |
| R1 수정 후에도 52.8% < 0.60 | mesh coverage 미적용 | FR-A2 하이브리드로 객체 단위 적용 (임계 의존 제거) |
| 자동 후보가 반복배관에서 오정합 | false observed | margin 게이트 + 사람 anchor 강제 + negative evidence |
| ICP 발산 | 정합 악화 | RMSE 개선 시에만 채택, 원본 보존 |
| 샘플 검수가 누락 유발 | 과대판정 | 불일치 1건 시 전수 격상, opt-in 한정 |
| GLB-PAG 단위/축 불일치 | 매핑돼도 위치 오차 | manifest `pag_to_glb_transform`·`up_axis` 검증 유지 |

## 9. 성공 지표

- PXX coverage가 PAG proxy 단독 → **267개 객체 실제 mesh 기준**으로 전환.
- 같은 zone 2번째 이후 업로드는 사람 클릭 **0~1회**로 정합(현재 6회).
- 자동 후보가 `ambiguous`일 때는 절대 자동확정하지 않음(반복배관 안전성 회귀 0건).
- 첫 정합 후 재구성 scale/origin이 고정되어 재정합 결과가 재현됨.
