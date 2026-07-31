# lingbot-map — 작업 지식 진입점

새로 합류했다면 여기부터. 상세 이력은 `handoff/`, 반복 함정은 `gotchas.md`, 결정 근거는 `decisions.md`.

---

## 현재 상태 (2026-07-31)

**정본 브랜치**: `feat/auto-scale-fusion` @ `0ee7de5`
**검증 기준선**: `pytest tests/ -q` → **369 passed** (이전 기준선 168 → +201)

영상 업로드 → 점군 재구성 → BIM(GLB) 위에 촬영 경로·coverage 표시. 최근 작업은 **현장이 BIM과
다를 때(현장 변경분 미반영)도 정합이 되게 하는 것**이었고, 팀 `coplay-planmatch-01`이 20커밋으로
마감했다(STATUS: CEILING — 아래 참조).

### 제품 스펙 (측정으로 확정)
> 도면 대비 실측 변경비율 **T ≈ 25%**(랜드마크 12개 중 8개 이상 생존)까지 자동 정합 가능.
> 그 이상은 성공률이 급락하지만 **확정오답 0을 유지하며 HOLD**로 빠진다.

근거: 7 T지점 × 3 seed × 20건 = **420건 전수에서 확정오답(ok_outside_d2) 0건**.

---

## 지금 손대면 되는 곳 (가치 순)

| # | 항목 | 시작 지점 | 왜 |
|---|---|---|---|
| 1 | **실도면 DXF 확보** | 사람에게 요청 | 20사이클 전부 합성 기하 검증. 이게 있어야 D2① 진짜 달성가능성을 최초 측정하고, 복도폭 1.82·L_end를 실측 검증한다 |
| 2 | 하네스를 실규모로 교체 | `tools/check_plan_match_robust.py` 섭동 생성기 | 현 하네스는 벽 10개·랜드마크 12개뿐이라 벽 하나만 사라져도 코너 전멸 → 게이트가 매처가 아니라 하네스를 측정 중 (1번 선행 필요) |
| 3 | `upload_1779442357085` 데모 렌더 | `tools/render_demo_format.sh <video> <out_dir>` | 이 업로드도 PiP가 raw 원본으로 fallback 중 (약 15분 소요) |
| 4 | `DEFAULT_SCALE_BAND` 재검토 | `scan2bim/coarse_match.py` | (0.05, 50.0) 1000배 폭 = 사실상 무력한 절대 스케일 게이트. QA 지적, 미해소 |

---

## 이 저장소의 게이트·규칙

**설계 불변식 (완화 금지)** — CLAUDE.md에도 있으나 여기 근거를 붙인다
1. red 정합에서 자동 observed 생성 금지
2. **반복 객체 모델에서 영상-only 자동 전역정합 확정 금지** — 후보는 명시 저장만.
   `coarse_match.finalize_match()`가 강제: 점수차 < margin_min → `hold` + `best=None`, 후보는 노출.
   거부 시 transform 밀반출 0(6경로 일괄 테스트). 변이검증으로 강도 입증(M1→4 failed, M2→2, M3→1)
3. 정합 전 scan/경로를 모델 위치에 임의 표시 금지 (`raw_scan_local`)

**무회귀 계약**: `--plan-match off`(기본)는 기존 `place_rigid` 출력과 **byte-identical**.
QA가 독립 sha256으로 3-way + CLI 2-way 재현 확인. 이 계약을 깨는 변경은 리뷰 없이 넣지 않는다.

**검증 명령**
```bash
.venv/bin/python -m pytest tests/ -q                                    # 369 passed
.venv/bin/python tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7
.venv/bin/python tools/check_plan_match_robust.py --sweep --n-perturb 20   # 열화곡선
.venv/bin/python tools/check_plan_match_robust.py --selftest-flood          # 게이트가 범람을 잡는지
```

---

## 최근 handoff

- [2026-07-31 coplay-planmatch-01](handoff/2026-07-31-coplay-planmatch-01.md) — 현장≠BIM 강건 정합, CEILING 판정, PiP 심링크 수정
