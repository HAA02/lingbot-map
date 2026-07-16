# 탐색 결과: 스케일 정확도 + 반복패턴 자동매핑 (2026-07-16)

대상 질문
1. 모델↔실영상 매핑이 이상함. 실거리/이동거리/회전을 정확히 못 냄.
2. 반복 패턴 현장(공사장)+설계모델에서 자동 매핑이 가능한가 — 검증 필요.

각 항목은 실제 `파일:줄` 근거로 확인함(추측 아님).

---

## ① 스케일: 웹 경로에 metric scale이 **아예 없다**

- `scan2bim/metric_scale.py` — 다중앵커 스케일 모듈(커밋 d0bcb14). 잘 만들어져 있음.
- `tools/build_coplay.py:305-311, 451-465` — **유일한 import 지점**(오프라인 데모 영상 렌더러).
- `realtime/server.py` — `metric_scale` import **0건**(grep 확인). → 웹 coverage 경로는 스케일 복원 단계 자체가 없음.

### `path_length_m`은 미터가 아니다 (단위 거짓말)
- `realtime/server.py:1650-1672` `_pose_descriptor()` — raw pose translation으로 `path_length_m`/`displacement_m` 계산. **scale 미적용**.
- `realtime/server.py:2620` — 그 값이 API 응답 `scan.trajectory`로 그대로 노출.
- 필드명은 `_m`(미터), 실제 단위는 단안 recon unit(up-to-scale). 사용자가 보는 이동거리는 우연히 맞지 않는 한 틀림.

### auto-place는 scale=1.0 고정
- `realtime/server.py:1940` — `_auto_place_candidates(scale: float = 1.0)` 기본값.
- `realtime/server.py:2077-2082` — `start_hint`가 있을 때만 scale 탐색(0.5~2.0, 9후보). 없으면 `scale_cands=[scale]` → **recon unit = 미터라는 무근거 가정**.

### build_coplay 경로마저 "가정 앵커"
- `scan2bim/metric_scale.py:29-35` — `camera_height_scale(assumed_height=1.5)`. 1.5m는 측정이 아니라 **가정**(사람별 편차가 그대로 스케일 오차로 전이 — 추정치, 실측 아님).
- `scan2bim/metric_scale.py:3-6` — 모듈 docstring이 단일앵커의 취약성을 스스로 인정("silently over/under-shoots, observed both ways").
- `scan2bim/metric_scale.py:52` — 앵커 불일치 시 1차 앵커(s_vert=천장고) 유지. 그 s_vert는 floor-to-ceiling 전체 촬영을 전제.
- `docs/scan2bim-progress.md:53` — Metric3D(학습기반 metric depth) **보류**(Python 3.14 휠 부재).

### 회전(rotation)은 별개 원인 — 같이 묶으면 안 됨
- similarity 변환은 각도 보존 → **스케일 오차는 회전각을 틀리게 하지 않는다.**
- 회전 이상의 실제 후보: `realtime/server.py:2068` yaw 후보 그리드가 `(-10, 0, 10)`도 **10° 간격**(거친 초기화) + 중력정렬 정확도.

---

## ② 반복패턴: 게이트는 설계대로 작동, 단 자동화율 0

- `realtime/server.py:1675-1708` `_model_repetition_risk()` — **개수 세기 휴리스틱**. `max_count>=40` 또는 반복 카테고리>=3 → "high". 실제 기하 자기유사성은 측정 안 함 → "배관이 많다"를 플래그할 뿐 "이 배관들이 서로 구분 불가능하다"가 아님.
- `realtime/server.py:2582` — repetition high면 `anchor_ok`(잔차<=0.30m) 필수 → 반복 모델은 **항상 수동 앵커**. 불변식대로 동작(옳음), 대신 자동화 0.
- `realtime/server.py:2593, 2600` — 근소차 후보 다수 → `ambiguous_alignment`, "auto-applied 금지" 경고.
- `docs/scan-to-model-improvement-prd.md:35` — PXX 순환수배관 **264개 반복** → 영상-only 확정 금지(설계상 옳음).
- `docs/glb-auto-mapping-review.md:78` — 시각 place recognition = **후순위 제외**(반복배관 내성 낮음).

### PRD R3은 stale (문서 오류)
- `docs/scan-to-model-improvement-prd.md:26` R3 "자동 후보 생성기 미구현, 기하 후보 0개".
- 실제로는 `realtime/server.py:1936` `_auto_place_candidates` 구현됨 + `2511`에서 호출 연결됨.
- PRD 최종수정 = 커밋 90b49ce(2026-06-12)이고, 그 커밋이 바로 자동배치를 추가한 커밋 → 문서가 자기 커밋을 반영 못 함. PRD가 인용한 줄번호(`_pose_descriptor` 1189)도 현재 1650으로 이동.

---

## ③ 종합: 두 문제는 하나다

`docs/scan-to-model-improvement-prd.md:30` **R7이 이미 정확히 지적**:
> 단안 재구성은 스케일 모호. `_solve_scan_to_model_alignment`는 매번 Sim(3) scale을 새로 추정 → 자유도↑ → 반복배관 환경에서 정합 모호성↑

- scale이 자유변수로 남으면 7DoF 탐색이고, **"작은 배관 가까이" vs "큰 배관 멀리"가 구분 불가** → 반복 구조에서 가짜 해가 폭증.
- metric scale을 앵커로 고정하면 Sim3→SE3(7→6DoF), 스케일 착시 해가 제거되어 반복 배관에서도 후보 급감.
- **∴ 스케일 고정이 반복패턴 자동매핑의 선행조건.** 순서는 스케일 먼저, 자동매핑 그 다음.

### 반복패턴 자동매핑, 원리적으로 가능한가
순수 기하 + 영상-only는 **다중해가 원리적**(가능/불가능의 문제가 아니라 정보 부족). 깨려면 반복을 깨는 **비반복 신호**를 넣어야 함:

| 접근 | 근거/현황 | 스케일도 해결? |
| --- | --- | --- |
| (a) VIO/IMU metric pose (ARKit·ARCore) | 휴대폰 관성으로 metric scale 직접 획득 + drift 제한 | ✅ 동시 해결 |
| (b) 비반복 앵커를 **건축**에서 추출 | 배관 264개 반복이지만 기둥/출입구/실번호는 덜 반복. `scan2bim/anchors.py:53`이 이미 distinctive 객체(HXX/EXX/SXX) 추출 중 | ❌ |
| (c) 1힌트(zone) + 자동후보 | `start_hint` 인프라 존재(server.py:2077). PRD 목표 "6클릭 → 1힌트" | 부분(힌트 시 scale 탐색 켜짐) |
| (d) cross-upload prior | R5(`server.py` `_previous_alignment_candidates`) — 현재 `can_save=false` preview 전용 | ❌ |

---

## 다음 액션 (권장 순서)

1. **`metric_scale.py`를 `realtime/server.py`에 연결** + `path_length_m` 단위 정직화(scale 적용, 또는 미적용이면 `path_length_recon_units`로 개명). 최소 변경·즉시 효과.
2. `speed_warning`/`bbox_height_warning`을 웹 경로에도 노출(현재 build_coplay 전용).
3. 스케일 고정 후 **SE3로 반복배관 후보 수를 재측정** → "자동화율이 실제로 오르는가"를 수치로 확인. (이게 사용자가 요청한 '검증'의 핵심 실험)
4. 모바일 업로드가 ARKit/ARCore metric pose를 실을 수 있는지 확인 → 실린다면 (a)가 1~3을 대체하는 근본해.
5. PRD R3 stale 표기 갱신.

## 미검증 / 열린 질문
- 모바일 업로드 페이로드에 IMU/VIO pose가 포함되는지 — **미확인**(이번 탐색 범위 밖).
- 실현장(Gasan 7F) 영상 미확보 → 실데이터 기준 스케일 오차 실측치 **없음**(`docs/scan2bim-progress.md:48-50`). 현재 수치는 전부 합성 GT.
- `assumed_height=1.5`의 실제 오차 기여도 — 미측정.
