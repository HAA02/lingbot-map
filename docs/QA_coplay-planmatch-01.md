# QA 종결 보고 — coplay-planmatch-01 (사이클 16, P3-QA-final)

작업 루트: `/run/media/iaan/1TB-WD/Github/lingbot-map/.claude/worktrees/loop-coplay-planmatch-01`
브랜치: `loop/coplay-planmatch-01`, 측정 시작 HEAD=`500741e`

## ⚠️ 측정 도중 발견한 충돌 (진행중 상태 경고)

`git status --short` (측정 도중 실측):
```
 M scan2bim/coarse_match.py
 M tests/test_coarse_match.py
```
**dev-core가 이 QA 측정과 동시에 `scan2bim/coarse_match.py`·`tests/test_coarse_match.py`를 실시간 편집 중**(씨앗 개선, 확정오답 잔여 1건). 두 파일은 QA 소유가 아니므로 건드리지 않았고 working tree도 stash 등으로 건드리지 않았다(dev 작업 유실 위험 회피).

**대응**: D1(pytest)·처음 실행한 D2(robust gate) 측정은 이 WIP 포함 상태를 반영한다(측정 시각 명시). **적대검증(P2 공격, 반복기하 latch)은 `git show HEAD:scan2bim/{coarse_match,plan_skeleton}.py`로 커밋 500741e 버전을 별도 프로세스에 얼려(import 시점에 `sys.modules` 치환) dev-core의 WIP과 완전히 분리해 실행**했다(아래 적대검증 절 — "HEAD-frozen" 표기). D4는 애초에 `coarse_match.py`를 import하지 않는 경로라 WIP과 무관(코드로 확인).

---

## D1~D7 집계표 (최종)

| # | 판정식 | 결과 | 근거 |
|---|---|---|---|
| D1 | `pytest tests/ -q` | **PASS** | `355 passed in 29.67s`, exit 0 (측정 시각 기준 WIP 포함 상태) |
| D2(기본) | `check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7` (팀 DoD 문구 그대로) | **FAIL(미달)** | exit 1. ①성공률 15.0%(<90%) ②outlier recall 38.8%(<80%) ③PASS |
| D2(sweep) | `--sweep` (열화곡선, 참고측정) | **FAIL(전 구간 미달)** | 7개 T지점 전부 게이트 `FFP` — recall이 5~50% 전 구간에서 80%를 단 한 번도 못 넘음(최댓값 60.0%@T=40%, 그 지점 성공률 8.3%로 붕괴). **재현 편차 발견**(아래 참조) |
| D3 | `check_coplay_geometry.py <배포 html>` 기존 플래그 + `--post-turn-heading-span-min 120` | **FLAG SET에 따라 갈림 — 아래 참조** | 최신(desmear) 플래그 전체 세트로는 FAIL(turn_z_min), rigid-01 기준 플래그로는 PASS |
| D4 | `--plan-match` off byte-identical | **PASS(QA 독립 재현)** | 함수레벨 3-way + 전체 CLI 2-way 모두 byte-identical, 두 방법 모두 QA 자체 실행 |
| D5 | 불변식② behavioral test | **PASS** | pytest 38/38(`test_build_coplay_planmatch.py`) + QA 독립 코드경로 확인(아래) |
| D6 | s_f 2소스 교차검증 | **불가로 종결** | 실업로드 `door_times()==[]` QA 독립 재현 완료 — 원리적으로 문통과 타이밍 소스 없음 |
| D7 | fixtures-untouched + 소유 파일만 변경 | **PASS** | `git diff --stat db93f40 500741e` — 팀 소유 파일 11개만 변경, `realtime/_uploads`·`lingbot_map` diff 0 |

**종합**: D1 PASS / **D2 FAIL** / D3 조건부(플래그 세트 의존) / D4 PASS / D5 PASS / D6 불가(단일소스로 종결) / D7 PASS.
**D2가 이 팀의 핵심 목표(강건 정합)의 기계 판정 그 자체이므로, D2 FAIL은 이 사이클 전체를 "미완료"로 판정하는 근거다.**

---

## D2 상세

### 기본 게이트 (팀 DoD 문구 그대로)
```
$ .venv/bin/python tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7
EXIT=1
[structured] ① 성공률: 3/20 = 15.0% (gate >= 90%) [ok_outside_d2(위험: 확정오답)=0 hold=12 reject=5 error=0]
[structured] ② outlier recall: 31/80 (scoreable) = 38.8% (radius=5.46m, gate >= 80%)
[structured] ③ 모호 시 HOLD: PASS (자동확정 위반 0건)
[structured] 게이트 판정: FAIL (①FAIL ②FAIL ③PASS)
```
`--upload`는 스크립트 자체 docstring상 "accepted but UNUSED"(합성 STAIR 하네스만 사용, 실업로드 자체를 섭동시키지 않음 — 설계·실측 일치).

**팀 DoD에 문자 그대로 적힌 명령은 이번 사이클도 실패(exit 1)한다.** 사이클 지시문의 "게이트 90% 달성" 서술은 이 기본 명령이 아니라 `--sweep`의 특정 지점(T=10%) **평균 성공률**만을 가리킨다.

### 열화곡선(`--sweep`, ratios=5/10/15/20/30/40/50%, seeds=7/42/123, n=20/point)
```
   목표T     실측T평균 |    성공률평균(범위) |   recall평균(범위)         | 게이트①②③
    5%     13.5% |  93.3%(85-100%) |  30.1%(15-52%)  | FFP
   10%     18.5% |  90.0%(85-95%)  |  30.9%(17-51%)  | FFP
   15%     21.6% |  88.3%(80-95%)  |  37.4%(24-52%)  | FFP
   20%     25.5% |  86.7%(85-90%)  |  48.6%(24-71%)  | FFP
   30%     33.8% |  36.7%(20-50%)  |  22.8%(0-38%)   | FFP
   40%     46.2% |   8.3%(5-10%)   |  60.0%(38-100%) | FFP
   50%     55.9% |   5.0%(5-5%)    |  24.1%(0-50%)   | FFP
```
(첫 실행 EXIT=0, 실행시간 75초)

**핵심 발견 1 — recall은 전 구간에서 게이트 미달**: `게이트①②③` 열이 7개 지점 전부 `FFP`. ②outlier recall은 5%~50% 전 구간에서 80%를 단 한 번도 넘지 못했다(최댓값 60.0%@T=40%, 그 지점은 성공률이 8.3%로 붕괴). "게이트 90% 달성"은 T=10%에서 ①의 **평균값**만을 가리키며, D2가 요구하는 ①∧②∧③ 종합 게이트가 어느 지점에서도 통과했다는 뜻이 아니다. P2-Robust 수정 전(commit `f84ab5a`, 5%→73.3%/12.3%recall, 10%→68.3%/24.0%recall)과 비교해 ①은 크게 개선됐으나 **②는 개선 전후 모두 구조적으로 80% 미달**.

**핵심 발견 2 — sweep 결과 자체가 완전히 재현되지 않는다 (QA 소유 도구의 신규 결함, 실측)**: 동일한 `--sweep` 명령을 반복 실행하면 케이스 경계에 걸린 소수 항목의 판정이 실행마다 바뀐다.
```
1회차(원본, 세션 시작 시): T=20% seed=42 -> 성공률=85.0%(17/20) ok_outside_d2=1 recall=71.0%
2회차(재실행, 동일 명령)  : T=20% seed=42 -> 성공률=90.0%(18/20) ok_outside_d2=0 recall=60.9%
3회차(재실행, 동일 명령)  : T=20% seed=42 -> 성공률=90.0%(18/20) ok_outside_d2=0 recall=60.9%
```
직접 라이브러리 호출로 분리 검증: `generate_perturbation_suite_at_total_ratio`(섭동 생성 자체)는 완전 결정적(동일 인자·동일 호출 순서·이전 호출 이력 무관 — 해시 100% 일치, 확인함). 차이는 `_evaluate_suite`(매처 실행+채점) 쪽에서 발생. `PYTHONHASHSEED=0`, `OMP/MKL/OPENBLAS_NUM_THREADS=1` 두 가지 통상 원인을 개별 테스트했으나 재현되지 않음(둘 다 90.0%/0 결과만 나옴) — **근본 원인 미확정**(경계에 걸린 케이스의 실수 연산 순서 의존 추정, 추가 조사 필요). **이 문제는 QA 소유 파일(`tools/check_plan_match_robust.py`) 자체의 결함**이며, 이번 사이클엔 시간상 수정하지 않고 사실만 보고한다 — "시드 고정"을 전제로 한 이 도구의 재현성 주장에 예외가 있다는 뜻. **D2 최종 판정에는 영향 없음**: recall이 80%를 못 넘는 것은 재현되는 모든 실행에서 공통(60.9%~71.0% 등 어느 쪽이든 80% 미달)이고, 성공률도 매 실행 85~90% 경계라 ①도 안정적으로 90%를 넘지 못한다.

→ **D2 최종 판정: FAIL(미달)**. 기본 게이트도, 열화곡선의 어떤 지점도 종합 게이트를 통과하지 못한다. 임계 완화 없음.

---

## D3 상세 (실측, 배포 산출물 `realtime/_uploads/upload_1781521406685.coplay.html` 대상)

"기존 플래그"의 정확한 정의가 팀 문서에 문자 그대로 박혀 있지 않아, git 이력에서 확인 가능한 두 세트를 모두 실행했다.

**세트 1 — coplay-desmear-01 D2 플래그(가장 최근 기록된 전체 세트) + 이번 사이클 지시 플래그**:
```
$ check_coplay_geometry.py <html> --turn-z-max 4 --turn-z-min 2 --end-x -8,4 --end-z-max 2.5 --pre-turn-x-range 2,6 --post-turn-heading-span-min 120
turn_point x=3.37 z=0.60 (fraction=0.83 angle=47.2deg) turn_z_max=4.0 -> OK
turn_z_min z=0.60 turn_z_min=2.0 -> FAIL
endpoint x=1.72 z=1.74 -> OK
pre_turn_x drift=1.56 band=[2.0,6.0] -> OK
post_turn_heading_span=923.8 deg min=120.0 -> OK
verdict=FAIL, EXIT=1
```
**세트 2 — coplay-rigid-01 D2 플래그(더 이른 기록) + 이번 사이클 지시 플래그**:
```
$ check_coplay_geometry.py <html> --turn-z-max 4 --end-x -8,4 --end-z-max 2.5 --post-turn-heading-span-min 120
verdict=PASS, EXIT=0
```
차이는 `turn_z_min 2`(및 `pre_turn_x_range`) 유무 하나 — desmear 세트를 쓰면 FAIL(turn_point z=0.60 < 2.0), rigid 세트를 쓰면 PASS. `turn_z_min=2`는 desmear 사이클 당시의 특정 geometry(다른 회전점)를 겨냥해 QA가 붙인 값으로 보이며, fwdscale 이후 회전점이 t=19.5s(라운지 입구, z=0.60 — desmear 시절보다 얕은 지점)로 재정의됐다는 점(팀 문서 기록)과 부합한다. **QA가 자체 판단으로 어느 세트가 "정답"인지 결정하지 않고 양쪽 다 실측·보고**한다 — turn_z_min을 이번 배포 형상에 재적용할지는 PM 판단 필요.

post_turn_heading_span=923.8도(2바퀴 이상 회전)는 fwdscale팀의 "라운지는 둘러보기, desmear 미적용, 자기형상 보존" 설계 결정과 일치(라운지에서 실제로 여러 번 돈 궤적을 그대로 보존).

---

## D4 상세 (QA 독립 재현 — dev-wire 주장 그대로 믿지 않음)

### 방법 1: 함수레벨 3-way 해시 비교 (HEAD-unspecified / HEAD-off / db93f40-baseline)
실 fixture: `realtime/_uploads/upload_1781521406685.lbp2` + `models/Gasan_7F/*.dtdx`. 사전 확인: `place_rigid`가 의존하는 `scan2bim/{metric_scale,forward_scale,pipe_path,dxf_plan}.py`는 db93f40↔HEAD 간 diff 없음(4개 파일 전부 무출력).
```
hash HEAD(unspecified) pose_json: bc65319ea09532df2161bc8a3f3ee425b91c4c039242917d052153e27d664b2a
hash HEAD(off)         pose_json: (동일)
hash db93f40(baseline) pose_json: (동일)
hash HEAD(unspecified) info     : cd333b731c501eefc41ab24f2a835c8c1660c928aa7dc9fbdb2ae63a61e19957
hash HEAD(off)/db93f40 info     : (동일)
```
(스크립트 `/tmp/qa_d4/d4_hash_compare.py`, kwargs: `horizontal_scale_override=2.30, turn_time_s=19.5, duration=35.3`)

### 방법 2: 전체 CLI 엔드투엔드 (QA가 직접 기동한 실서버 경유, `--upload` 실사용)
```
cmp qa_rebuild_unspecified.html qa_rebuild_off.html  → 무출력(byte-identical)
sha256sum 둘 다: b7af036102f8e20d531871115d077b5a675dc72f017cd1d6b25f20cf62806c37
(4.1MB, tris=84,092, poses=144, 둘 다 동일)
```
**D4 판정: PASS.** dev-wire가 주장한 해시(`8ce3c6cc...64b4`)는 QA의 해시 방식(HTML/필드 sha256)과 달라 문자 그대로 재현되진 않았으나, 같은 성질(byte-identical)을 QA가 독립적으로 두 가지 방법으로 재확인했다. `scan2bim/coarse_match.py` WIP과 무관(off 경로는 `_plan_match_auto`를 호출하지 않음 — `do_plan_match=False`, 코드 확인).

---

## D5 / 불변식② 상세

pytest `tests/test_build_coplay_planmatch.py` 38/38 통과(`TestAutoAloneConfirmsNothing`, `TestExplicitAcceptIsTheOnlyWayIn`, `TestHoldCannotBeAccepted`, `TestCliDefaultsCannotAutoConfirm` 등). QA 독립 코드 확인: `tools/build_coplay.py`의 `_place_from_plan_match(...)` 호출은 **오직** `if accept_plan_match is not None:` 블록 내부에서만 일어난다(구조적으로 `--plan-match auto` 단독 실행 시 이 함수 자체가 호출되지 않음 — diff로 직접 읽어 확인). 아래 적대검증(chirality) 항목에서 추가 확인.

---

## D6 판정 — 종결

`reports/coplay/rgb_doors/upload_1781521406685.door_rgb.json`을 QA가 직접 로드해 재현(커밋 `5a26a78` 서술을 그대로 믿지 않고 산출물 자체를 확인):
```python
door_times(d) == []          # scan2bim.door_detect_rgb.door_times, min_confidence 기본값
len(d['doors']) == 0         # 확정된 문 0건
len(d['candidates']) == 42   # 검토된 후보 전체
max(c['confidence'] for c in d['candidates']) == 0.0   # 전원 신뢰도 0
```
가장 근접한 두 후보(t=11.80s, t=17.25s)의 거부 사유: `reject=['no_lintel:no_member_spanning_the_pair', 'below_min_confidence']` — 폭은 각각 recon 1.308/1.109 유닛(× s_h≈1.97 ≈ 2.58m/2.18m, commit 서술과 일치)으로 문 규격이 아니라 복도 폭 스케일이고, 상인방(스팬부재)이 관측되지 않아 기각. 점군 고도분포(82%가 카메라보다 위 등)는 QA가 이번 사이클엔 재도출하지 않음(원 커밋 주장만 인용, 미검증 — 다만 D6 판정에 필수적이지 않음: `door_times()==[]` 자체가 이미 충분한 근거).

**→ 판정: "미실행"이 아니라 "이 데이터로는 불가 — s_f는 단일소스(DXF 복도끝)로 남는다."** 문통과 타이밍은 이 업로드에 원리적으로 존재하지 않는 증거(개방형 오피스, 실제 문 없음, RGB 검출기 확정 0건)이며, 재시도로 얻어질 수 있는 종류의 실패가 아니다.

**대안 교차검증 소스 검토**: 코너(t=19.5s) 통과 타이밍은 이미 s_f 산출의 1차 소스(DXF L_end / leg-A recon arclen)의 분자·분모 구성요소 그 자체이므로 이를 재사용해도 독립 소스가 아니다(순환논증). RGB 검출기가 확인한 바로는 복도 구간에 문 외의 식별 가능한 고정 랜드마크(예: 별도 개구부·기둥·바닥재질 경계)도 관측되지 않았다(개방형 오피스, 천장지향 촬영으로 벽 자체가 프레임에 거의 담기지 않음). **결론: 이 업로드에서 두 번째 독립 기하 앵커는 QA가 검토한 범위 내에서 발견되지 않았다** — s_f 단일소스 상태를 이번 사이클에 "불가"로 명시 종결하는 것이 맞다.

---

## 적대검증 (실측 — HEAD 500741e 고정 버전, dev-core WIP과 분리)

방법: `git show HEAD:scan2bim/{coarse_match,plan_skeleton}.py`를 별도 파일로 추출, `sys.modules['scan2bim.coarse_match'/'scan2bim.plan_skeleton']`에 직접 등록해 로드(dev-core가 편집 중인 working-tree 버전을 우회). `scan2bim.forward_scale`/`scan2bim.dxf_plan`은 db93f40 이후 diff 없음을 먼저 확인했으므로 실제 패키지에서 그대로 로드.

### 1) "말도 안 되는 스케일 통과" 공격 — over-run 비용 제거 직접 공격
`DEFAULT_SCALE_BAND = (0.05, 50.0)`(m/recon-unit, 1000배 폭)를 확인 — 매처 자체의 절대 스케일 방어선은 사실상 **비율 밴드(`DEFAULT_ANISO_BAND = (0.2, 5.0)`)** 하나뿐이다. 이를 겨냥해 STAIR 하네스에서 leg 절단률을 QA D2 밴드(10~30%)보다 훨씬 크게 밀어붙였다:

```
문 증거 없음(실업로드와 동일 조건):
  leg B 74.1% 제거 -> hold(ambiguous_margin)
  leg B 87.8~99.8% 제거 -> reject(inlier_ratio_below_min)   [전부 REJECT, ok 없음]
문 증거 있음(최선 조건):
  leg B 74.1% 제거 -> ok, s_f/s_h 오차 0.0% (2번째 코너 seed로 정확 복구)
  leg B 87.8~99.8% 제거 -> reject(inlier_ratio_below_min)   [전부 REJECT]

P2 커밋이 인용한 정확한 버그 사례(벽 세그먼트 1개 제거 = leg 시작 0.91m 삭제) 재현:
  문 증거 없음 -> hold(ambiguous_margin)   [확정오답 아님]
  문 증거 있음 -> ok, s_h 오차 0.00%, offset 0.0000m  [정확 복구]
```
**결론: 이번에 시도한 단일-leg 극단 절단 공격으로는 "말도 안 되는 스케일"이 `ok`로 통과하는 사례를 만들지 못했다** — 문 증거가 없을 때(실업로드 조건)는 극단적 절단에서 항상 HOLD 또는 REJECT, 문 증거가 있을 때는 오히려 second-corner-seed 메커니즘이 정확히 복구한다. 다만 **D2 sweep 자체는 T=15~20%에서 `ok_outside_d2`(확정오답)가 실제로 발생한다**(각 1건, T=20%는 위 재현성 문제로 실행마다 0~1건) — 이는 STRUCTURED 다중 벽 동시 섭동(제거+이동+잡음이 한 케이스에 동시 발생) 조건이며, 이번 시간 내 단일 벡터 공격으로는 그 정확한 발생 경로를 못 밟았다. **DEFAULT_SCALE_BAND(0.05, 50.0)이 사실상 무력한 절대 게이트라는 사실 자체는 실측 확인됐고, 이는 실질적 위험(다른 조건에서 극단적 스케일이 통과할 여지)으로 남아있다** — QA는 이를 코드 사실로 보고하되, 이 문서의 시간 범위 내에서 그 정확한 재현 케이스를 만들지는 못했음을 명시한다.

### 2) 엉뚱한 복도(반복기하) latch 공격
QA가 dev 픽스처와 무관하게 직접 구성: 동일한 2-leg L 복도 2개(50m 이격, 문 전혀 없음 — 실업로드와 동일 조건)를 하나의 skeleton에 넣고, 그중 하나(코리더 A)를 실제로 걸음:
```
status: hold, hold_reason: ambiguous_margin, margin: 0.0
후보 8개, translation-X 버킷: [-7, 9, 12, 43, 59, 62] -- 코리더 A(~0)와 B(~50) 양쪽에 후보가 걸쳐있음
```
**결론: 엉뚱한 복도로 확정되는 경로 없음** — 반복 기하 + 문 증거 부재 조건에서 정확히 HOLD, 두 복도 모두 후보로만 노출되고 자동확정 없음(설계대로).

### 3) chirality 뒤집기 누출 경로
코드 구조 확인(`tools/build_coplay.py`): 미러(chi 반전) 변환을 실제로 pose_json에 적용하는 `_place_from_plan_match(...)`는 **오직** `if accept_plan_match is not None:` 분기 안에서만 호출된다 — `--plan-match auto` 단독 실행 경로에는 이 함수 호출 자체가 존재하지 않는다(diff로 직접 확인, D5 절 참조). pytest `TestAutoAloneConfirmsNothing`(6건, 최상위 후보가 완벽히 맞는 매치인 최악의 유혹적 케이스 포함) 전부 통과 — `plan_match='auto'`만으로는 pose_json이 `off`와 byte-identical임을 검증. **QA는 이번 사이클에 chi가 뒤집힌 것이 최상위 후보인 별도 시나리오를 직접 구성한 신규 테스트는 만들지 못했음**(시간 제약) — 위 코드 구조 증명 + 기존 pytest로 결론 대체.

### 4) P2 4건 수정 관대화 여부 종합
- **over-run 비용 제거**: 위 1)에서 직접 공격 — 시도한 범위에서 확정오답 유발 못 찾음(단, sweep 자체에선 드물게 발생 — 근본 벡터 미특정).
- **median→mean(RESID_AGG)**: 별도 미공격(시간 제약) — pytest만 근거.
- **residual 정규화**: 별도 미공격(시간 제약) — pytest만 근거.
- **문 xy_pass**: 별도 미공격(시간 제약) — pytest만 근거.

---

## 서버
QA가 D4용으로 직접 기동한 `realtime/server.py`(PID 632744, `127.0.0.1:8767`)는 **측정 종료 후 정리(kill) 완료**.

---

## 재현 명령 모음
```
cd /run/media/iaan/1TB-WD/Github/lingbot-map/.claude/worktrees/loop-coplay-planmatch-01
PY=/run/media/iaan/1TB-WD/Github/lingbot-map/.venv/bin/python

# D1
$PY -m pytest tests/ -q

# D2 기본게이트 / 열화곡선
$PY tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7
$PY tools/check_plan_match_robust.py --sweep

# D3 (배포 html 대상, 두 플래그 세트)
$PY tools/check_coplay_geometry.py realtime/_uploads/upload_1781521406685.coplay.html \
  --turn-z-max 4 --turn-z-min 2 --end-x -8,4 --end-z-max 2.5 --pre-turn-x-range 2,6 \
  --post-turn-heading-span-min 120
$PY tools/check_coplay_geometry.py realtime/_uploads/upload_1781521406685.coplay.html \
  --turn-z-max 4 --end-x -8,4 --end-z-max 2.5 --post-turn-heading-span-min 120

# D4 (스크립트 /tmp/qa_d4/d4_hash_compare.py, 전체 CLI는 서버 기동 후 tools/build_coplay.py 직접 호출)

# D6 (재현)
$PY -c "
import json,sys; sys.path.insert(0,'.')
from scan2bim.door_detect_rgb import door_times
d = json.load(open('reports/coplay/rgb_doors/upload_1781521406685.door_rgb.json'))
print(door_times(d), len(d['candidates']))"

# 적대검증 (스크립트, HEAD-frozen 로더 /tmp/qa_adv/setup_head.py 선행 필요)
#   /tmp/qa_adv/probe_a_overrun_attack.py   (over-run/스케일 공격)
#   /tmp/qa_adv/probe_b_repeated_corridor.py (반복기하 latch)
```

---

## 사이클 19 — 게이트② 설계 결함 보정: 정밀도 지표 편입

작업 루트: 동일. 시작 HEAD=`8d7fcbf`(dev-core "span 변경 스캔" 커밋, recall 38.8%→91.2%, 정밀도 결여를 자진신고).
수정 파일: `tools/check_plan_match_robust.py`만(패치 스크립트로 정확히 16개 앵커 지점 편집, `git diff --stat` 확인 예정). `scan2bim/**`·`tools/build_coplay.py`는 읽기만.

### 문제 (dev-core 자진신고, 커밋 8d7fcbf 원문)
> FLOOD 상한: 84개 span 을 전부 깃발 꽂으면 recall 100%. ②는 커버리지로 포화되므로 recall 수치만으로는 검출과 범람을 구별할 수 없다. 구별하는 숫자는 정밀도다 — 정상 도면 오경보 0/28 span, 발화는 24~25/84.

게이트②(outlier recall≥80%)에는 정밀도 짝이 없어, "모든 span에 깃발을 꽂는" 미래의 회귀가 recall만으로는 항상 PASS로 통과한다. 이번 사이클은 이 설계 결함을 막는 것이 목적이다.

### 설계 — 새 게이트 항목 ④⑤ (QA 소유, 근거 명시)

| 항목 | 정의 | 임계 | 근거 |
|---|---|---|---|
| ④ 무섭동(clean) 도면 span 오경보율 | 무섭동 STAIR 도면(clean baseline)에서 발화한 span 수 / 전체 span 수 | **0% (엄격)** | 무섭동 도면은 정의상 실제 변경이 전무하므로 발화는 전부 오경보. R1/R2 판정식(`SPAN_CROSS_MARGIN=0.20m`, `SPAN_WIDTH_TOL=0.30m`)이 이미 기하 노이즈를 흡수하도록 설계됐으므로, 이 조건에서 0이 아니면 그 자체가 톨러런스 결함. 현재 실측 0(아래)과 일치 — **이 목표에 맞춰 임계를 역산한 게 아니라, 물리적으로 "변경이 없으면 발화도 없어야 한다"는 요구를 먼저 세우고 그것이 우연히 현재값과 같음을 확인한 것**(FLOOD 자가진단이 이 항목을 즉시 100%로 깨는 것으로 증명 — 아래). |
| ⑤ span 정밀도(precision) vs. prevalence | `compute_span_precision`: 변환이 확정된(item①) 케이스들의 top candidate에 대해 span 단위 TP/FP/FN/TN 집계. `precision=TP/(TP+FP)`, `prevalence=(TP+FN)/n`("실제 변경분이 차지하는 span 비율", **매 실행마다 ground truth로 새로 계산**, 하드코딩 없음) | **precision ≥ 1.5 × prevalence** | "모든 span에 깃발을 꽂는" flood 구현은 TP=전체 양성, FP=전체 음성이 되어 precision이 수학적으로 **정확히 prevalence와 같아진다**(그 이상 절대 못 감) — 이건 정의상 항상 참인 명제이지, 관측치에 맞춘 임계가 아니다. 1.0배(경계값)가 아니라 1.5배로 여유를 둔 이유는 표본이 작을 때(3케이스×spans) 경계선 근처의 우연한 통과를 막기 위함. **24~25/84라는 관측 수치는 임계에 전혀 쓰이지 않았다** — 매 실행 자신의 prevalence를 기준으로 삼는 상대 검정이라 미래의 다른 섭동 강도·다른 업로드에도 적응적으로 작동한다. |

두 항목 모두 `gate_ok = gate1 AND gate2 AND gate3 AND gate4 AND gate5`로 편입되며, `_print_single_gate_report`가 ④⑤를 별도 줄로 분해 출력하므로 FAIL 시 어느 항목 때문인지 항상 특정 가능하다.

### FLOOD 회귀 테스트 (`selftest_flood_detection`, `--selftest-flood`) — 이 과제의 핵심 증거

`scan2bim.coarse_match._span_outliers`를 "모든 span을 무조건 발화"하는 가짜 함수로 **인메모리 monkeypatch**(디스크 미수정, `finally`에서 원복 — `scan2bim/**` 읽기전용 제약 준수)한 뒤 같은 D2 하네스로 재측정.

```
$ .venv/bin/python tools/check_plan_match_robust.py --selftest-flood --seed 7 --n-perturb 20
[flood selftest] 무섭동(clean) 도면(패치 하에서 재빌드): 24/24 span 발화 (오경보율=100.0%) -> ④ FAIL
[flood selftest] 섭동 스위트(seed=7 n=20): precision=65.3% prevalence=65.3% (TP=47 FP=25 FN=0) -> ⑤ FAIL
[flood selftest] 참고로 ①②③(가짜 구현이 판정 경로를 건드리지 않았다는 격리 확인): ①FAIL ②PASS ③PASS
[flood selftest] 결과: PASS -- 게이트가 범람을 잡았다(④ 또는 ⑤가 FAIL)
EXIT=0
```

**핵심 관찰**:
- **②(기존 recall)는 flood 하에서도 PASS**(별도 측정: 91.2%→98.75%(79/80)로 오히려 상승) — dev-core의 우려가 정확했음을 실측으로 재확인. recall 단독으로는 이 결함을 못 잡는다.
- **④는 즉시 FAIL**: 무섭동 도면에서 24/24(100%) 오경보 — 실제 변경이 0건인데 전부 "변경됨"으로 발화.
- **⑤도 FAIL**: precision(65.3%) == prevalence(65.3%) **소수점까지 정확히 일치** — "flood의 precision은 수학적으로 prevalence와 같다"는 설계 근거가 실측으로 그대로 검증됨(우연이 아니라 항등식).
- exit code 0 = "self-test 통과"(=게이트가 범람을 실제로 잡았다는 뜻, 가짜 구현이 뚫었으면 exit 1).

재현:
```
cd /run/media/iaan/1TB-WD/Github/lingbot-map/.claude/worktrees/loop-coplay-planmatch-01
PY=/run/media/iaan/1TB-WD/Github/lingbot-map/.venv/bin/python
$PY tools/check_plan_match_robust.py --selftest-flood --seed 7 --n-perturb 20
```

### 하위호환 확인 — 기존 판정 뒤집히지 않음

```
$ $PY tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7
[structured] ① 성공률: 3/20 = 15.0% (gate >= 90%)  -- 불변(수정 전과 동일)
[structured] ② outlier recall: 73/80 = 91.2% (gate >= 80%)  -- 불변(수정 전과 동일), PASS
[structured] ③ 모호 시 HOLD: PASS  -- 불변
[structured] ④ 무섭동(clean) 도면 span 오경보율: 0/24 = 0.0% (gate <= 0.0%) => PASS  -- 신규, 통과
[structured] ⑤ span 정밀도: TP=22 FP=0 FN=25 TN=25 precision=100.0% recall_span=46.8% F1=63.8%
             prevalence=65.3% 발화율=30.6% (gate: precision >= 1.5x prevalence) => PASS  -- 신규, 통과
[structured] 게이트 판정: FAIL (①FAIL ②PASS ③PASS ④PASS ⑤PASS)
EXIT=1
```
①②③ 수치와 개별 PASS/FAIL은 수정 전(사이클 16 P3-QA-final, 사이클 18 dev-core 커밋)과 **완전 동일**하게 재현됐다(①15.0%/②91.2%/③PASS, 종합 FAIL). 신규 ④⑤도 현재 실구현에선 PASS이므로, **종합 판정(FAIL, ①때문)은 지표 추가로 뒤집히지 않았다** — 정밀도 임계를 현재 구현이 통과하도록 역산하지 않았다는 방증이기도 하다(⑤는 100.0% vs 임계 97.95%로 여유가 크지 않은 정직한 통과이지 느슨한 임계가 아니다).

`⑤`의 표본이 3케이스(item①이 확정한 case #08/#11/#13)뿐이라 `n_cases_included=3`으로 작다는 점은 정직하게 출력에 노출된다(`n_cases`) — item①이 90% 게이트를 통과하지 못하는 한(현재 15%) ⑤의 통계적 힘도 구조적으로 제한된다는 뜻이며, 이는 은폐하지 않고 그대로 보고한다.

### pytest / 전체 재현

```
$ $PY -m pytest tests/ -q
369 passed in 32.81s   (수정 전과 동일 — tests/, scan2bim/ 무변경이므로 당연)

$ $PY tools/check_plan_match_robust.py --sweep --sweep-ratios 0.10,0.20 --sweep-seeds 7,42 --n-perturb 10
EXIT=0 (정상 동작 확인 -- ④⑤ 열/게이트 문자열 5자리로 확장, precision 평균 열 추가)

$ $PY tools/check_plan_match_robust.py --perturb-mode both --n-perturb 5 --seed 7
EXIT=1 (정상 동작 확인 -- structured/fragment 양쪽 다 ④⑤ 출력, FAIL 분해 메시지에도 ④⑤ 반영)
```

### 재현 명령 모음 (사이클 19 신규분)
```
cd /run/media/iaan/1TB-WD/Github/lingbot-map/.claude/worktrees/loop-coplay-planmatch-01
PY=/run/media/iaan/1TB-WD/Github/lingbot-map/.venv/bin/python

# pytest 회귀 없음 확인
$PY -m pytest tests/ -q

# D2 기본 게이트 (④⑤ 포함, 하위호환 확인)
$PY tools/check_plan_match_robust.py --upload upload_1781521406685 --n-perturb 20 --seed 7

# FLOOD 회귀 테스트 (이 사이클의 핵심 증거)
$PY tools/check_plan_match_robust.py --selftest-flood --seed 7 --n-perturb 20

# sweep/both 모드 정상 동작 확인(스모크)
$PY tools/check_plan_match_robust.py --sweep --sweep-ratios 0.10,0.20 --sweep-seeds 7,42 --n-perturb 10
$PY tools/check_plan_match_robust.py --perturb-mode both --n-perturb 5 --seed 7
```

---

## 사이클 20 — D2 게이트 재정의: T(총변경비율) 축을 1차 판정축으로 승격 (사용자 결정)

작업 루트: 동일. 시작 HEAD=`09cc84f`(사이클 19, ④⑤ 정밀도 지표 + FLOOD 회귀 테스트). 수정 파일: `tools/check_plan_match_robust.py`만.

### 재정의 근거 — "통과시키려고가 아니라, 정답을 줘도 20%인 시험은 매처를 측정하지 못하기 때문"

천장 리뷰어 오라클 실험(메인 세션이 두 실험 모두 직접 재현, 이 사이클에 QA가 다시 독립 재현):
```
ORACLE(매처에게 정답 변환을 그냥 건네줬을 때) 하드게이트 통과: 4/20 = 20.0%   ← 게이트는 90% 요구
실측 T(--legacy-gate, 즉 재정의 이전 기본 게이트) = 52.2%
```
정답 변환(harness의 CONSTRUCTIVELY KNOWN true transform)을 매처 자신의 하드게이트(`min_inlier_ratio`/
`max_residual`, `scan2bim.plan_skeleton.DEFAULT_GATES`)에 그대로 통과시켜도 20%뿐이라는 것은 "이 T
에서는 어떤 검색 알고리즘도 90% 게이트를 넘을 수 없다"는 산수이지, 매처의 결함이 아니다. 그래서 이번
사이클은 **임계(90%/80%)도, 섭동 크기 범위(제거 10-30%·이동 0.3-1.0m 등)도 전혀 손대지 않고**, 게이트가
판정하는 **지점**(T축 위 어디)만 옮긴다. 이 구분은 `tools/check_plan_match_robust.py`의 모듈 docstring
(CLI/Exit codes 절)과 `GATE_DEFAULT_TOTAL_RATIO` 상수의 주석 블록에 그대로 명시했다.

### 판정 지점 산정 — 목표 T=20%가 실측 T≈25-26% 창에 대응

Section 2c의 기존 배분 규칙(`SWEEP_ALLOC_WEIGHTS`, REMOVE/SHIFT/NOISE_FRAC_RANGE 중앙값 비율, 변경 없음)을
목표 T=15/20/25% 세 점에 그대로 적용해 재측정(`--sweep --sweep-ratios 0.15,0.20,0.25 --sweep-seeds 7,42,123
--n-perturb 20`):
```
   목표T     실측T평균
    15%     21.6%
    20%     25.5%   (seed=7 단독 26.3%)
    25%     29.7%
```
사용자가 지정한 "실측 T≈25-26%" 창에 들어오는 목표 지점은 **20%뿐**이다(15%는 21.6%로 아래, 25%는 29.7%로
위). 이미 있던 배분 규칙을 그 창에 맞는 목표 T에 그대로 적용한 결과이며, 현재 관측치에 임계를 역산한 것이
아니다 — `GATE_DEFAULT_TOTAL_RATIO = 0.20`으로 확정, 근거는 상수 옆 주석 블록에 그대로 남겼다.

### 구현

- `compute_oracle_upper_bound(ctx, suite)`(신규): harness의 참 변환을 `scan2bim.coarse_match._associate`에
  직접 통과시켜 `DEFAULT_GATES`(min_inlier_ratio/max_residual)를 만족하는지만 확인(탐색 없음) — 어떤 검색
  전략도 넘을 수 없는 ①의 이론적 상한. `_evaluate_suite`에 항상(옵트인 아님) 편입해, 레거시·신규 게이트·
  (부수적으로) `--sweep` 모두에서 계산되도록 했다(단, `--sweep`은 오라클을 출력하지 않음 — 이번 사이클
  범위는 "게이트 출력"이라 판단해 표에는 추가하지 않았다, 계산 자체는 부작용 없이 항상 됨).
- `run_d2_gate_at_ratio(seed, n_perturb, total_frac=GATE_DEFAULT_TOTAL_RATIO)`(신규): Section 2c의
  `generate_perturbation_suite_at_total_ratio`(기존, `--sweep`이 이미 쓰던 것)로 섭동 생성, `_evaluate_suite`
  로 채점 — **새 기본 게이트 경로**.
- `--legacy-gate`(신규 플래그): 기존 `run_d2_gate`/`_run_d2_gate_single`(Section 2b 독립범위 중첩추첨)을
  그대로 실행, 실행마다 경고 배너 출력(아래 실측 인용). 삭제 없음 — `--perturb-mode fragment/both`도
  `--legacy-gate`와 함께일 때만 유효(신규 기본 T축 경로는 structured 전용, `--sweep`과 동일 설계).
- `--total-ratio`(신규 플래그, 기본 `GATE_DEFAULT_TOTAL_RATIO`=0.20): 신규 기본 경로의 판정 지점 T를
  바꿀 수 있게 노출(측정용, DoD 지점 자체는 여전히 기본값 0.20).
- `_print_single_gate_report`: 오라클 상한 줄(①헤드룸 표시, 초과 시 버그 경고) + (신규 경로일 때) 판정
  지점(목표T/실측T) 줄을 추가. `main()`의 D2 게이트 실행부를 `--legacy-gate` 분기 / 신규 T축 분기로
  재구성, FAIL 사유 조립을 `_print_fail_misses` 헬퍼로 공유(오라클 상한도 FAIL 메시지에 포함).

### 측정 1 — 재정의된 기본 게이트 (`--n-perturb 20 --seed 7`, T=20% 지점)

```
$ .venv/bin/python tools/check_plan_match_robust.py --n-perturb 20 --seed 7
[structured] [판정 지점, CYCLE 20] 목표 총변경비율 T=20.0% -> 실측 T평균=26.3% (범위 22.0-32.8%, n=20건, seed=7) -- 사용자 지정 창(실측 T≈25-26%) 내부
...
[structured] ① 성공률: 17/20 = 85.0% (gate >= 90%) [ok_outside_d2(위험: 확정오답)=0 hold=3 reject=0 error=0]
[structured]    실패 사유 분포(P2 입력, ①에서 ok_within_d2 아닌 모든 케이스): {"insufficient_events": 3}
[structured] [오라클 상한, CYCLE 20] 정답 변환을 그대로 매처의 하드게이트(min_inlier_ratio>=0.60, residual<=0.60)에 통과시켰을 때: 17/20 = 85.0% -- 이 섭동 지점에서 어떤 검색 전략도 ①을 이 값보다 높일 수 없다(탐색 품질과 무관, 정답을 이미 줬으므로). ①실측 대비 헤드룸=+0.0%p
[structured]    recall: 202/242 (scoreable, transform-confirmed case만) = 83.5% (radius=5.46m, gate >= 80%) [case 미확정 전이 exclude=113 no_candidate=0]
[structured] ③ 모호 시 HOLD(...) => PASS
[structured] ④ 무섭동(clean) 도면 span 오경보율(...): 0/24 = 0.0% (gate <= 0.0%) => PASS
[structured] ⑤ span 정밀도(...): precision=93.8% ... prevalence(실제변경 span 비율)=52.5% ... => PASS
[structured] 게이트 판정: FAIL (①FAIL ②PASS ③PASS ④PASS ⑤PASS)
D2 게이트: FAIL(T=20% 지점, 실측T평균=26.3%) -- ①85.0%<90%; [오라클상한=85.0%]
EXIT=1
```

**핵심 결과 — T축 이동만으로는 90%에 도달 불가**: ①실측(85.0%)이 그 지점 오라클 상한(85.0%)과 **정확히
일치**한다(헤드룸 +0.0%p) — 매처가 이 지점에서 이미 이론적 상한에 도달해 있고, 상한 자체가 90%보다
낮다. 즉 사이클 7~19에 걸친 매처 개선(트리밍/두 번째 코너 시드/span 정밀도 등)과 무관하게, **이 지점의
하드게이트 통과 상한 자체가 90% 미만**이므로 T를 20~25% 창 안에서 어디로 옮겨도(15%→21.6%, 25%→29.7%
포함) ①이 확정적으로 90%를 넘는다고 보장할 근거가 없다(사용자 지시대로 T를 통과하도록 고르지 않고
지정 창을 지켰다 — 통과가 나오지 않았다는 사실을 그대로 보고한다). ②③④⑤는 모두 PASS.

**실패 3건 전부 `insufficient_events`, 원인(코드 확인, `scan2bim/coarse_match.py:1128-1130`)**: 이
hold_reason은 `not P["corner"]`(섭동된 평면 스켈레톤에 코너가 하나도 안 남음)에서만 발생한다(워크 쪽
코너는 고정 워크라 항상 있음). T=20%에서 제거 예산은 19/232 densified fragment(≈8.2%)이고, `structured`
모드는 예산을 원 벽(wall_id) 단위로 통째로 제거한다(Section 2b) — STAIR 하네스의 **원 벽이 10개뿐**이라
그중 코너를 이루는 짧은 벽(예: dead-end cap, 길이=폭 1.82m, densify 후 조각 수 적음) 하나가 통째로
뽑히면 그 조각들이 예산 안에 다 들어가 코너 자체가 스켈레톤에서 사라진다. **가설(실도면이 이 레포에
없어 검증 불가 — 이 팀 전 사이클 공통 한계, 그대로 승계)**: 실제 DXF는 벽이 수십~수백 개일 것이므로,
같은 %제거가 훨씬 많은 벽에 분산돼 특정 코너 하나를 전멸시킬 확률이 하네스보다 낮을 것으로 추정된다 —
이것은 추정일 뿐, 이 사이클엔 실 DXF가 없어 검증하지 못했다(D2 DoD의 원문이 이미 명시한 한계, P0-Skeleton
단계부터 "실도면 SXX DXF 경로 확인(레포에 없음)"으로 기록됨).

### 측정 2 — `--legacy-gate` (기존 수치 재현 + 경고 배너)

```
$ .venv/bin/python tools/check_plan_match_robust.py --legacy-gate --n-perturb 20 --seed 7
[LEGACY GATE 경고, CYCLE 20] --legacy-gate: 이 설정(REMOVE/SHIFT/NOISE_FRAC_RANGE 각각 독립 10-30%/10-30%/5-20% 범위에서 매 케이스 중첩 추첨, 범위 자체는 이 사이클도 미변경)은 실측 총변경비율 T≈52%를 만든다(seed=7 n=20, 참고치 -- 아래 결과 자체는 이번 실행값을 그대로 출력). 이 T에서는 매처에게 정답 변환을 그대로 건네줘도(오라클, compute_oracle_upper_bound) 하드게이트(min_inlier_ratio/max_residual) 통과율이 4/20=20.0%(실측, 재현됨)뿐이다 -- 정답을 줘도 90% 게이트를 못 넘는 시험이라는 뜻, 즉 이 설정은 매처를 측정하지 못한다. 기록 보존 목적으로 계속 실행 가능하게 남겨두되 (삭제 아님), 판정은 참고용으로만 취급할 것 -- CYCLE 20부터 기본 게이트는 --total-ratio(기본 0.2, 실측 T≈25-26%)로 대체됐다. 아래에도 오라클 상한이 매 케이스 함께 출력된다(같은 혼동 재발 방지).
...
[structured] ① 성공률: 3/20 = 15.0% (gate >= 90%) [ok_outside_d2(위험: 확정오답)=0 hold=12 reject=5 error=0]
[structured] [오라클 상한, CYCLE 20] ... 4/20 = 20.0% -- ... ①실측 대비 헤드룸=+5.0%p
[structured]    recall: 73/80 (scoreable, transform-confirmed case만) = 91.2% (radius=5.46m, gate >= 80%) [case 미확정 전이 exclude=477 no_candidate=0]
[structured] ③ 모호 시 HOLD(...) => PASS
[structured] ④ 무섭동(clean) 도면 span 오경보율(...): 0/24 = 0.0% (gate <= 0.0%) => PASS
[structured] ⑤ span 정밀도(...TP=22 FP=0 FN=25 TN=25 n_cases=3): precision=100.0% ... prevalence(실제변경 span 비율)=65.3% ... => PASS
[structured] 게이트 판정: FAIL (①FAIL ②PASS ③PASS ④PASS ⑤PASS)
D2 게이트[LEGACY]: FAIL(mode=structured) -- ①15.0%<90%; [오라클상한=20.0%]
EXIT=1
```

**①15.0%(3/20) ②91.2%(73/80) ③④⑤PASS — 사이클 16/19에 기록된 수치와 완전히 동일하게 재현됐다.**
경고 배너가 "실측 T≈52%" · "오라클 상한 20.0%(4/20)" · "매처를 측정하지 못한다"는 취지를 실행마다
그대로 출력한다(위 인용 그대로). 삭제가 아니라 맥락을 붙인 것 — 기록 보존.

### 측정 3 — `--selftest-flood` (필수 회귀, 재정의 후에도 범람을 잡는지)

```
$ .venv/bin/python tools/check_plan_match_robust.py --selftest-flood --seed 7 --n-perturb 20
[flood selftest] scan2bim.coarse_match._span_outliers 를 '모든 span 무조건 발화'로 인메모리 monkeypatch(디스크 미수정, finally 에서 원복) -- dev-core 8d7fcbf 자진신고 재현: recall 38.8%->91.2% 개선과 함께 '84개 span 전부 깃발 꽂으면 recall 100%'을 self-report 했던 바로 그 가짜 구현
[flood selftest] 무섭동(clean) 도면(패치 하에서 재빌드): 24/24 span 발화 (오경보율=100.0%) -> ④ FAIL
[flood selftest] 섭동 스위트(seed=7 n=20): precision=65.3% prevalence=65.3% (TP=47 FP=25 FN=0) -> ⑤ FAIL
[flood selftest] 참고로 ①②③(가짜 구현이 판정 경로를 건드리지 않았다는 격리 확인): ①FAIL ②PASS ③PASS
[flood selftest] 결과: PASS -- 게이트가 범람을 잡았다(④ 또는 ⑤가 FAIL)
EXIT=0
```
재정의 후에도 동일하게 범람을 잡는다(`--selftest-flood`는 legacy suite 생성기를 그대로 쓰므로 이 사이클
변경과 무관 — ④⑤ 계산 경로 자체는 `_evaluate_suite` 안에서 공유되지만 로직 변경 없음, 새로 추가된 것은
오라클 계산뿐이고 그건 `_span_outliers` 패치와 무관한 corner/leg/door 하드게이트 채점이라 flood 패치의
영향을 받지 않는다 -- 위 출력에서도 ①②③은 정상, ④⑤만 잡힌 것으로 격리가 확인된다).

### pytest 회귀 없음

```
$ .venv/bin/python -m pytest tests/ -q
369 passed in 33.29s / 33.50s (재실행)   -- 감소 0, 증가 0 (이 파일은 tests/ 어디서도 import되지 않음, grep 확인)
```

### 종합 판정

| 항목 | 결과 |
|---|---|
| 재정의된 기본 게이트(T=20%, seed=7, n=20) | **FAIL** — ①85.0%(17/20)<90%, 오라클상한도 85.0%(헤드룸 +0.0%p, 구조적 상한). ②③④⑤ PASS. EXIT=1 |
| `--legacy-gate` | **FAIL**(수치 불변, ①15.0% ②91.2% ③④⑤PASS), 경고 배너 실측 출력 확인. EXIT=1 |
| `--selftest-flood` | **PASS**(범람 여전히 잡힘, ④/⑤ FAIL로 검출). EXIT=0 |
| pytest | **369 passed**, 감소 0 |

**D2는 재정의 후에도 미달(FAIL)이다.** T를 사용자 지정 창(20~25%) 안에서 옮겼음에도 ①은 90%를 넘지
못했고, 그 지점의 오라클 상한 자체가 85%(이 하네스·이 T에서 검색이 이미 상한에 도달)라는 사실이 원인을
명확히 한다 — 매처의 탐색 품질 문제가 아니라 **이 합성 하네스(원 벽 10개)의 구조적 한계**로 보인다는
가설(위 "실패 3건" 단락, 검증 불가)과 일치한다. 통과하도록 T나 임계를 다시 고르지 않았다.

### 재현 명령 모음 (사이클 20 신규분)
```
cd /run/media/iaan/1TB-WD/Github/lingbot-map/.claude/worktrees/loop-coplay-planmatch-01
PY=/run/media/iaan/1TB-WD/Github/lingbot-map/.venv/bin/python

# 재정의된 기본 게이트 (T=20% 지점)
$PY tools/check_plan_match_robust.py --n-perturb 20 --seed 7

# 레거시 게이트(경고 배너 포함, 기존 수치 재현)
$PY tools/check_plan_match_robust.py --legacy-gate --n-perturb 20 --seed 7

# FLOOD 회귀(필수, 재정의 후에도 범람 검출)
$PY tools/check_plan_match_robust.py --selftest-flood --seed 7 --n-perturb 20

# 판정 지점 산정 근거(목표T -> 실측T 매핑 재현)
$PY tools/check_plan_match_robust.py --sweep --sweep-ratios 0.15,0.20,0.25 --sweep-seeds 7,42,123 --n-perturb 20

# pytest 회귀 없음
$PY -m pytest tests/ -q
```
