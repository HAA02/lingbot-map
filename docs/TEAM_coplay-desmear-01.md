# Agent Team: coplay-desmear-01

목표: `upload_1781521406685`의 coplay 궤적에서, 급회전(t=13.5s, 목재클래딩 벽 개구부) 중 특징점 부족으로 recon이 방향전환을 9초(t≈16.5~25s)에 걸쳐 스미어링한 구간을 후처리로 보정 — 회전이 실제 위치에서 일어나도록 만든다. `place_rigid`(스케일·강체정렬, 검증완료)는 건드리지 않는다.

## PM 판단 (착수 전 확정)
- **방법 (b) 국소 헤딩 재분배 채택** (arc-length 이상적-L 스냅(a)은 폐기 — axis-split 스케일과 충돌하고 recon 자기형상을 모델 각도로 덮어써 불변식과 긴장).
  - step 길이 보존 재적분: 스미어 구간에 분산된 net Δheading을 알려진 코너(pose~55, turn-time-s 유래)에 계단으로 집중, 90°로 스냅하지 않고 recon 측정 net 회전량 그대로 보존.
  - 코너 인덱스 이전(legA) 좌표는 절대 불변.
- **범용 후처리 스테이지 + opt-in 게이트**: `desmear_turn()` + `--desmear-turn` 플래그. 기존 패턴(corridor-width-hint/horizontal-scale-override/turn-time-s)과 동일 — 미지정 시 byte-동일 무회귀.
- **DoD의 "목재 클래딩 개구부 좌표" 정의**: 마감(목재)은 LOD상 모델에 없음(validate/why.md:51) → 복도-코너 프록시로 정의: place_rigid docstring(Z≈3) + FXX 복도 중심선(X≈4.57) + 현재 스미어 코너(3.94,3.15) 3중 근거로 **X≈4~5, Z≈3** 확정.

## 팀 구조도
```
PM (opus, 루프 오케스트레이션)
├─ P0-Dev(opus)  방법 확정+TDD red   ┐ 병렬
├─ P0-QA(sonnet) 게이트 옵션 추가    ┘
├─ P1 구현(opus)   desmear_turn()+플래그
├─ P2 자기수렴(opus) 실빌드→게이트exit0
└─ P3 QA게이트(sonnet) 독립재현+정확부훼손 적대검증
```

## 역할 및 모델 배정
| 역할 | kind | 모델 | 배정 근거 |
|---|---|---|---|
| pm | pm | **opus** | 사용자 지정(2026-07-18) |
| dev-lead | dev | opus | 비강체 궤적보정=근본원인·비가역 수치변환 설계(헤딩 재적분·정확부 보존) |
| qa | qa | sonnet | de-smear 정확성=시각·수치 판단. 위음성(정확구간 훼손 못잡음) 대가 큼 → Haiku 부적격 |

## 파일 소유권
| 파일 | 권한 | 소유 |
|---|---|---|
| `tools/build_coplay.py` | 수정(신규 `desmear_turn()`+`--desmear-turn`+place_rigid 뒤 디스패치) | dev-lead |
| `scan2bim/turn_desmear.py`(신규, 선택) | 생성(순수 헤딩재분배 기하) | dev-lead |
| `tests/test_turn_desmear.py`(신규) | 생성(TDD) | dev-lead |
| `tools/check_coplay_geometry.py` | 수정(`--turn-fraction-range`, `--turn-z-min` 추가형 게이트, default off) | qa |
| `place_rigid`/`place_registered`/`place_pipe_auto`/`place_gtpath` 본문, `scan2bim/{wall_anchor,metric_scale,dxf_plan,pipe_path}.py` | 불변(읽기·재사용만) | — |
| smoke 픽스처, `realtime/_uploads/**`, `lingbot_map/`, 기존 테스트 | 불변 | — |

## Phase
| Phase | 담당 | 내용 | 의존 |
|---|---|---|---|
| P0-Dev | opus | 방법(b) 확정, 스미어 윈도우 검출식, TDD red | — |
| P0-QA | sonnet | check_coplay_geometry 게이트 옵션 추가(default off 무회귀) | — (P0-Dev와 병렬) |
| P1 | opus | desmear_turn()+플래그+디스패치, 단위테스트 green | P0-Dev |
| P2 | opus | 실빌드→새게이트 exit0까지 자기수렴 | P1, P0-QA |
| P3 | sonnet | 독립재빌드+전게이트재현+무회귀+정확부훼손 적대검증 | P2 |

## DoD
빌드: `build_coplay.py … --auto-rigid --turn-time-s 13.5 --desmear-turn --duration 35.3 --out reports/coplay/upload_1781521406685.coplay.html`

| # | 명령/확인 | 기대 |
|---|---|---|
| D1 | 빌드 실행 | 예외없이 HTML, reginfo에 desmear 로그 |
| D2 | `check_coplay_geometry.py <산출> --turn-fraction-range 0.29,0.49 --turn-z-max 4 --turn-z-min 2 --end-x -8,4 --end-z-max 2.5 --pre-turn-x-range 2,6` | **exit 0** |
| D3 | turn_point | x≈4~5, z≈3 |
| D4 | 정확부 보존(pose≤corner_idx) | c 좌표 차이 ≤1e-3, path_m 편차 ≤5% |
| D5 | 꼬리 직선성(pose[110:]) | de-smear 후 최대수직편차가 증가하지 않음 |
| D6 | `--desmear-turn` 미지정 재빌드 + `--auto-pipe` | 기존과 byte-동일(무회귀) |
| D7 | pytest 전체 + fixtures-untouched | 전부 통과 |
| D8 | git diff --stat | 소유 파일만 |

## 반복정책
initial_max=3, hard_max=6. 정체 시 PM 재판정(방법(a) 폴백 또는 `--desmear-window-s` 명시입력 강등).

## 리스크
1. de-smear가 9초 밖 정확부를 훼손 — D4/D5로 감지
2. net Δheading 과/부족 추정 — legB 방향 각잔차·종점 밴드 이탈로 감지, 90° 스냅 금지로 방지
3. 노이즈 헤딩에 윈도우검출 스파이크 latch — D2 `--turn-fraction-range` 이탈로 감지, `--desmear-window-s` 명시 오버라이드로 방지

## loop-engineer 연동
`~/.claude/loop-settings/-run-media-iaan-1TB-WD-Github-lingbot-map/config.yml`의 `team:` 블록. 루프 실행: `/loop-engineer`.

## 재생성/갱신
    /TeamPM [목표]   ← 이 문서와 config.yml team 블록 함께 갱신
