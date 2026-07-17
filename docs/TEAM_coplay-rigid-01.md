# Agent Team: coplay-rigid-01

목표: `upload_1781521406685`의 coplay 궤적을 화면상 정확한 위치에 안착(회전점 Z≤4, 종점 open zone). axis-split metric scale(wall-scale-x3, 검증완료)은 재사용만 — 이 팀은 손대지 않는다.

## PM 판단 (착수 전 확정)
- `place_pipe_auto`: 코너가 FXX CAD polyline에 scale-invariant 비율로 100% 스냅됨(`pipe_path.py` `run_L_polyline`) — 배관 실적추종용, walk 재현 도구 아님. **수정 안 함(유효한 별도 목적)**.
- `place_registered`: ICP+천장격자 탐색이 이 반복배관 건물에서 rmse=0.609/yaw=150°로 로컬미니마 수렴. 코드 버그가 아니라 method-fit 미스매치, 불변식("반복객체 영상-only 자동 전역정합 확정 금지")과도 충돌하는 방향. **수정 안 함(범위 밖 한계)**.
- **결론**: 사용자가 이미 수동 검증한 파이프라인(axis-split 스케일 + 2D 강체정렬, ICP·CAD스냅 없음)을 재현하는 **셋째 경량 모드 `place_rigid`(`--auto-rigid`) 신설**.
- 핵심 미결정(Opus 근본결정): 강체정렬이 무엇에 정렬하는가(자동 대응/앵커) — 잘못되면 `place_registered`를 깬 것과 동일한 평행배관 yaw-flip/미러 실패 재발.

## 팀 구조도
```
PM (fable, 루프 오케스트레이션)
└─ P0 설계 → P1 구현 → P2 자기수렴  (Dev-lead, Opus, 연속)
   └─ P3 QA 게이트 (Sonnet)
```

## 역할 및 모델 배정
| 역할 | kind | 모델 | 배정 근거 |
|---|---|---|---|
| pm | pm | fable | 사용자 지정(2026-07-17). 역할배정 자체는 router 기준 그대로 |
| Dev-lead | dev | opus | 강체정렬 타깃 설계=근본원인·비가역결정 |
| QA | qa | sonnet | 안착 정확성=시각·수치 판단, 위음성 대가 큼 |

## 파일 소유권
| 파일 | 권한 | 소유 |
|---|---|---|
| `tools/build_coplay.py` | 수정(신규 `place_rigid`+`--auto-rigid`+디스패치) | Dev-lead |
| `scan2bim/metric_scale.py`, `wall_anchor.py`, `pipe_path.py` | 읽기·재사용만(수정금지) | Dev-lead |
| `tools/check_coplay_geometry.py` | 읽기·실행만 | QA |
| `place_pipe_auto`/`place_registered`(동일 파일 내) | 불변 | — |
| `tests/test_wall_anchor.py`, `test_metric_scale.py`, smoke 픽스처, `lingbot_map/` | 불변 | — |

## Phase / 의존관계
| Phase | 담당 | 내용 | 의존 |
|---|---|---|---|
| P0 설계 | Dev(Opus) | 강체정렬 타깃 결정 + no-scale 2D 정렬 헬퍼 확정 | — |
| P1 구현 | Dev(Opus) | `place_rigid`+`--auto-rigid`, axis-split/pose꼬리 재사용 | P0 |
| P2 자기수렴 | Dev(Opus) | 빌드→check_coplay_geometry exit 0까지, path≈20.5m/0.58m/s 교차확인 | P1 |
| P3 QA게이트 | QA(Sonnet) | 독립 재빌드+gate재현+무회귀(D4/D5/D6) | P2 |

## DoD
| # | 명령 | 기대 |
|---|---|---|
| D1 | build_coplay.py --auto-rigid 빌드 | 예외없이 HTML, s_v≈1.75/s_h≈3.64 출력 |
| D2 | check_coplay_geometry.py --turn-z-max 4 --end-x -8,4 --end-z-max 2.5 | **exit 0** |
| D3 | D2 출력 path_m/avg_speed_ms | ≈20.5m / ≈0.58m/s |
| D4 | 동일 인자 --auto-pipe 빌드(무회귀) | 예외없이 생성 |
| D5 | coverage_web_smoke.py | 통과 |
| D6 | git diff --stat | build_coplay.py(+산출 HTML)만 변경 |

## 반복정책
initial_max=4, hard_max=8 (단일 함수 크럭스, 소수 반복 수렴 기대). 정체 시 PM 재판정(정렬 앵커 재설계 or GT 대응점 fallback).

## 리스크
1. 정렬타깃 모호→yaw-flip/미러 (place_registered와 동일 실패류) — D2 turn-Z·종점 동시통과로 감지
2. axis-split 위 이중스케일(등방 scale 재도입) — D3 path_m이 ~40/~10이면 즉시 신호
3. poses 소스 불일치(fetch_scan 서버 vs demo-html 캐시) — Dev/QA 동일 소스 고정

## loop-engineer 연동
`~/.claude/loop-settings/-run-media-iaan-1TB-WD-Github-lingbot-map/config.yml`의 `team:` 블록. 루프 실행: `/loop-engineer`.

## 재생성/갱신
    /TeamPM [목표]   ← 이 문서와 config.yml team 블록 함께 갱신
