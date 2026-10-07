# plan-match

> 범위: 현장이 BIM과 다를 때 자동 정합을 어디까지 확정하는가 · 갱신 2026-10-07 · 근거 handoff: handoff/2026-07-31-coplay-planmatch-01.md

## 정본

- 제품 스펙: 도면 대비 변경비율 T≈25%(랜드마크 12개 중 8개 이상 생존)까지 자동 정합. 그 이상은 확정오답 0을 유지하며 HOLD.
- 근거는 합성 기하 420건(7 T지점 × 3 seed × 20)에서 `ok_outside_d2` 0건. 실도면 DXF로 재측정한 값은 아니다.
- 반복 기하에서 영상만으로 전역정합을 자동 확정하지 않는다. 점수차가 마진 미만이면 `hold`, `best=None`.
- `--plan-match off`는 `place_rigid`와 byte-identical. 이 계약을 깨는 변경은 넣지 않는다.
- red 정합에서 observed를 만들지 않는다. 정합 전 scan은 `raw_scan_local`.

## 할 일

- 실도면 DXF를 받아 `tools/check_plan_match_robust.py` 하네스를 실규모로 바꾼다.
- `scan2bim/coarse_match.py`의 `DEFAULT_SCALE_BAND` (0.05, 50.0)는 절대 스케일 게이트로 동작하지 않는다. 재검토 전.

## 함정

### 게이트가 매처 대신 하네스를 잰다 (2026-07-31)

벽 10개·랜드마크 12개 합성 도면이라 벽 하나가 사라지면 코너가 전멸한다. 실측 T와 혼동하지 않는다.

## 결정

### 2026-07-31 · T≈25%에서 자동, 그 이상은 HOLD

확정오차보다 미확정을 택했다. 실도면이 생기기 전에는 이 숫자를 현장 스펙으로 올리지 않는다.
