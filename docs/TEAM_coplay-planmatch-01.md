# Agent Team: coplay-planmatch-01

목표: **현장과 BIM이 100% 일치하지 않는 조건**(현장 변경분 BIM 미반영)에서도, DXF/BIM 평면 스켈레톤(복도중심선·코너·문 그래프)에 궤적을 **outlier-허용 부분매칭**(trimmed/RANSAC)해 경로/위치/회전/스케일을 강건하게 추적한다. 현장 변경분은 outlier로 배제하고, 반복 기하에서는 다중후보를 랭킹해 **명시 저장**(자동확정 금지 — 설계 불변식 ②)한다.

근거: `docs/scout-bim-mapping-accuracy.md` (2026-07-30 실측 검증 — 영상↔BIM 픽셀 대응 전무, 기하통계 정합의 한계 s_h1.97/s_f3.53 이방성).

## 팀 구조도

```
리더(메인 세션)
└─ pm (fable/xhigh) — 계획·품질평가·반복조절
   ├─ P0-Skeleton  dev-core (opus)   ┐ 병렬
   ├─ P0-Perturb   qa       (sonnet) ┘
   ├─ P1-Match     dev-core (opus)   ┐ 병렬 (API 계약 확정 후)
   ├─ P1-Wire      dev-wire (sonnet) ┘
   ├─ P2-Robust    dev-core (opus)
   └─ P3-QA-final  qa       (sonnet)
```

## 역할 및 모델·effort 배정

| 역할 | kind | 모델 | effort | tools | maxTurns | 배정 근거 |
|---|---|---|---|---|---|---|
| pm | pm | opus | xhigh | — | — | 계획수립/최종종합 — router 기준 기본 Opus 고정 (2026-07-30 사용자 요청으로 fable→opus 변경) |
| dev-core | dev | opus | xhigh | Read,Write,Edit,Bash,Grep,Glob | 30 | 신규 알고리즘 설계(부분매칭 목적함수·트리밍 정책·HOLD 판정). 반복 기하 오수렴이 시리즈에서 2회 실증된 고위험 — router "설계/근본원인 → Opus" |
| dev-wire | dev | sonnet | xhigh | Read,Write,Edit,Bash,Grep,Glob | 30 | 정형 구현 — `--forward-scale`(:1293)·`--dxf`(:1286) 등 opt-in 플래그 선례가 동일 파일에 실재. router "기존 패턴 답습 → Sonnet" |
| qa | qa | sonnet | xhigh | Read,Write,Bash,Grep,Glob | 20 | 위음성 대가 큼(오정합이 그럴듯하게 보인 실증 이력) + 기하/의미 판정형. Haiku 불가 — 단순 대조 슬롯 없음, 재검증 전제로 총비용 증가. owns 있으므로 Write 필수 |
| ceiling-reviewer | reviewer | opus | xhigh | Read,Bash,Grep,Glob | 15 | 천장 판정 비판 검토(PreFlight RedTeam 강도) — 잘못된 CEILING이 루프를 조기 종료시킴 |

effort는 전 역할 **xhigh 고정**(사다리 없음). 상향 축은 모델뿐: 같은 DoD 2연속 실패 → sonnet→opus 1회(`escalation_policy`).

## 파일 소유권 (충돌 방지)

| 역할 | 소유(쓰기) | 참조(읽기) |
|---|---|---|
| dev-core | `scan2bim/plan_skeleton.py`(신규), `scan2bim/coarse_match.py`(신규), `tests/test_plan_skeleton.py`(신규), `tests/test_coarse_match.py`(신규) | `scan2bim/{dxf_plan,pipe_path,forward_scale,metric_scale}.py`, `tools/build_coplay.py`, `docs/scout-bim-mapping-accuracy.md`, `docs/glb-auto-mapping-review.md` |
| dev-wire | `tools/build_coplay.py`, `tests/test_build_coplay_planmatch.py`(신규) | `scan2bim/{coarse_match,plan_skeleton}.py`(API 계약), `tools/check_coplay_geometry.py`, `docs/TEAM_coplay-fwdscale-01.md` |
| qa | `tools/check_plan_match_robust.py`(신규 — 시드 고정 섭동 생성기 내장), `tools/check_coplay_geometry.py`(게이트 플래그 확장) | `**` 전체 |

- 겹침 없음: dev-wire는 check_coplay_geometry.py **읽기만**, qa는 build_coplay.py **읽기만**.
- **섭동 픽스처는 QA 소유** — dev가 자기 픽스처에 과적합하는 자기채점을 구조적으로 차단(D2 게이트 신뢰의 근거).
- 불변: `lingbot_map/**`, `realtime/_uploads/**`, `scan2bim/{dxf_plan,localize,pipe_path,turn_desmear}.py`, `demo.py`

## Phase / 의존관계

| Phase | 역할 | 작업 | 병렬 가능 | 의존성 |
|---|---|---|---|---|
| P0-Skeleton | dev-core | DXF→복도중심선·코너·문 그래프 추출(`plan_skeleton.py`), 기지값(복도폭≈1.82m·L_end·문위치) 대비 검증. **후보 JSON 스키마(API 계약) 확정 → dev-wire 착수 신호**. 실도면 SXX DXF 경로 확인(레포에 없음 — 미확보 시 합성 plan으로 진행) | P0-Perturb와 병렬 | — |
| P0-Perturb | qa | 세그먼트 레벨 섭동 생성기: 벽 10~30% 제거·0.3~1.0m 이동·잡음 세그먼트 추가, 시드 고정. `load_wall_segments` (N,2,2) 포맷 위에서 동작 | P0-Skeleton과 병렬 | — |
| P1-Match | dev-core | trimmed/RANSAC 부분매처(`coarse_match.py`): 코너+leg arclen+문통과 이벤트 대응, 다중후보 랭킹, inlier ratio·outlier 세그먼트 목록, 모호 시 HOLD | P1-Wire와 병렬 | P0-Skeleton |
| P1-Wire | dev-wire | `--plan-match` opt-in 통합, 후보 명시 저장/수락 배선, reginfo 진단 노출, off 시 무회귀 | P1-Match와 병렬 | P0-Skeleton(API 계약만) |
| P2-Robust | dev-core | qa 섭동 스위트 대상 수렴: 트리밍 임계 튜닝, outlier recall 확보(게이트 판정식은 qa 소유 그대로) | — | P1-Match + P0-Perturb |
| P3-QA-final | qa | 실업로드 독립재빌드, 무회귀(byte-identical), s_f 2소스 교차검증 종결, 적대검증(chirality 뒤집기·엉뚱한 복도 latch·모호입력 자동확정 여부) | — | P1-Wire + P2-Robust |

## DoD (기계 판정)

| # | 명령 | 판정 |
|---|---|---|
| D1 | `.venv/bin/python -m pytest tests/ -q` | exit 0 (기준선 155 + 신규 전부) |
| D2 | `.venv/bin/python tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7` | exit 0. 시드 고정 섭동 plan 20개(합성 현장변경)에 대해 ① 무섭동 기준해 대비 yaw≤5°·이동≤0.5×복도폭·축별 스케일 상대오차≤10% 성공률 **≥90%** ② 주입 변경 세그먼트 outlier 검출 recall **≥80%** ③ 모호 케이스(점수차<마진) 자동확정 없이 HOLD |
| D3 | `check_coplay_geometry.py reports/coplay/upload_1781521406685.coplay.html` 기존 플래그 + `--post-turn-heading-span-min 120`(:119 실재) | exit 0 |
| D4 | `--plan-match` 미지정 빌드 → 현행 place_rigid 출력과 `cmp` | **byte-동일** (fwdscale D8 선례) |
| D5 | 불변식 ② behavioral test: 명시 수락 없이 빌드 → 후보 JSON 존재하되 확정 표기 없음 | pytest assert |
| D6 | 매처 문통과타이밍 s_f vs `_forward_scale_auto`(:826) s_f | 상대오차 ≤15%, 초과 시 HOLD(억지 일치화 금지) |
| D7 | fixtures-untouched + 소유 파일만 변경 | exit 0 |

D2가 목표의 기계 판정 그 자체: 실제 현장 변경 GT가 없으므로 **합성 변경에 대한 정합 성공률+outlier recall**을 대리 지표로, 실업로드는 D3로 이중 확인.

## 제외 트랙 (후속 이관 — 이번 팀 범위 아님)

- (b) `scan2bim/localize.py` solvePnP의 BIM 구조 엣지 확장 — 픽셀 대응 인프라(BIM 엣지 렌더러+2D 엣지 검출기) 별도 트랙
- (c) 각속도 가중 키프레임 재배분 — `demo.py`가 쓰기 허용 경로 밖 + GPU recon 재실행 필요 + 기존 s_h/s_f 캘리브레이션 무효화. 제약 해제 합의 후 별도 팀

## loop-engineer 연동

이 팀은 `~/.claude/loop-settings/-run-media-iaan-1TB-WD-Github-lingbot-map/config.yml`의 `team:` 블록으로 연결되어 있다.
루프 실행: `/loop-engineer` (이 문서를 직접 붙여넣을 필요 없음 — config.yml이 이미 참조한다).
binding_mode: **headless**(이번 세션 — `.claude/agents/`를 이번에 신설해 핫리로드 미부착). 다음 세션부터 inseason.

## 재생성/갱신

    /TeamPM [목표]   ← 이 문서·`.claude/agents/coplay-planmatch-01-*.md`·config.yml의 team 블록을 함께 갱신
