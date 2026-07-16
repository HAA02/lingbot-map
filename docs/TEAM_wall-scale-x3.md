# Agent Team: wall-scale-x3

목표: 영상→점군 재구성의 metric scale ~3배 과소(실측: upload_1781521406685, 10.53m/35.3s=0.298m/s)를
**벽 평면(복도 폭) 기반 독립 앵커**로 수정하고, 실패가 자동 감지되게 속도 게이트를 강화한다.

## 팀 구조도

```
PM (fable, 메인 세션이 루프 오케스트레이션)
├─ Phase 1a  anchor-core (Opus)   — 벽 RANSAC·복도폭·3앵커 융합·속도게이트  ┐ 병렬
├─ Phase 1b  coplay-wire (Sonnet) — 실데이터 red 베이스라인 재현 스크립트    ┘
├─ Phase 2   coplay-wire (Sonnet) — build_coplay 연동 (API 확정 후)
├─ Phase 3   scale-qa (Sonnet)    — 독립 재실행 + 적대적 검증
└─ Phase 4   결함 회송 루프 (소유 dev → scale-qa 재검증)
```

## 역할 및 모델 배정

| 역할 | kind | 모델 | 배정 근거 |
|---|---|---|---|
| pm | pm | opus | 계획수립/최종종합 — router 기준(Opus 고정). track1~2 초기는 fable override였으나 2026-07-16 사용자 지시로 opus 전환 |
| anchor-core | dev | opus | 코드베이스에 선례 없는(AXX 참조 0건) 신규 기하 알고리즘 설계 — 벽쌍 모호성·노이즈 강건성·융합 가중치 근본 판단 필요 |
| coplay-wire | dev | sonnet | d0bcb14(카메라높이 앵커) 커밋과 동일 패턴의 정형 연동 + lbp2 리더 재사용 검증 스크립트 |
| scale-qa | qa | sonnet | 위음성 대가 큼(게이트가 3배 오차를 조용히 통과시키면 목표 무효) + 수치 대역 판단 필요 → Haiku 부적격. 단순 대조(픽스처 불변)도 겸무 |

## 파일 소유권 (충돌 방지)

| 역할 | 소유(쓰기) | 참조(읽기) |
|---|---|---|
| anchor-core | `scan2bim/metric_scale.py`, `scan2bim/wall_anchor.py`(신규), `tests/test_metric_scale.py`, `tests/test_wall_anchor.py`(신규) | `scan2bim/registration.py`, `realtime/server.py`, `docs/glb-auto-mapping-review.md`, `realtime/_uploads/upload_1781521406685.lbp2` |
| coplay-wire | `tools/build_coplay.py`, `tools/validate_wall_anchor.py`(신규), `tests/test_build_coplay_anchor.py`(신규) | `scan2bim/**`, `realtime/server.py`, `**/*.glb`, `realtime/_uploads/upload_1781521406685.lbp2` |
| scale-qa | 없음 (읽기 전용, 결과는 리포트 회송) | 저장소 전체 |

인터페이스 계약(Phase 1a 종료 시 anchor-core가 확정·공지):
`estimate_wall_scale(pts_yup, model_corridor_widths) -> float | None`

## Phase / 의존관계

| Phase | 역할 | 작업 | 병렬 가능 | 의존성 |
|---|---|---|---|---|
| 1a | anchor-core | 실패 테스트 먼저(합성 복도 점군→벽쌍 간격, 3앵커 융합, `speed_warning(10.53, 35.3)`→경고) → 구현 → API 시그니처 확정 | 1b와 병렬 | 없음 |
| 1b | coplay-wire | 검증 스크립트 골격: lbp2 로드 + 현행 스케일 0.298 m/s 재현(red 베이스라인) | 1a와 병렬 | 없음 |
| 2 | coplay-wire | `place_pipe_auto`/`place_registered`에 벽 앵커 연결, 검증 스크립트 완성 | 불가 | 1a (API 확정) |
| 3 | scale-qa | DoD 전체 독립 재실행 + 실데이터 속도 대역 + 픽스처/`lingbot_map/` 불변 + 설계 불변식 적대적 확인 | 불가 | 2 |
| 4 | 소유 dev → scale-qa | 결함 회송·수정·재검증 | 결함별 병렬 | 3 |

## DoD 수용 기준

| # | 명령 | 기대 결과 |
|---|---|---|
| 1 | `.venv/bin/python -m pytest tests/ -q` | 전부 통과 (기존 72 + 신규 ≥8) |
| 2 | `.venv/bin/python -m pytest tests/test_metric_scale.py -q` | `speed_warning(10.53, 35.3)` 경고 반환, 정상 0.93 m/s는 None |
| 3 | `.venv/bin/python tools/validate_wall_anchor.py realtime/_uploads/upload_1781521406685.lbp2` | 벽 앵커 적용 후 평균 보행속도 **0.6~1.4 m/s** → exit 0. `--no-wall-anchor` 시 0.298 재현+경고 |
| 4 | git status 확인 | smoke 픽스처(upload_1779439687108/1779442357085)·`lingbot_map/` 변경 0건 |
| 5 | `grep -n "fuse_scale_estimates" tools/build_coplay.py` | 양쪽 place_*에서 3앵커 호출, 벽 앵커 None 시 2앵커 폴백 |
| 6 | scale-qa 코드 리뷰 | 설계 불변식 유지, `_auto_place_candidates` 미접촉(범위 외) |

게이트 규칙: 1~4는 scale-qa 독립 재실행(개발자 자기보고 불인정), 전 항목 통과 전 종료 금지.

## loop-engineer 연동

이 팀은 `~/.claude/loop-settings/-run-media-iaan-1TB-WD-Github-lingbot-map/config.yml`의 `team:` 블록으로 연결되어 있다.
루프 실행: `/loop-engineer` (이 문서를 직접 붙여넣을 필요 없음 — config.yml이 이미 참조한다).

## Track 2: axis-split-scale (2026-07-16, 트랙1 CEILING 후속)

트랙1 재검증에서 이방성 확정(수직 s≈1.75 정확, 수평 ~2배 추가 압축 → s_h≈3.64 필요). 등방 Sim3 CEILING은 유효하나 축분리 diag(s_h, s_v, s_h)는 구현 가능 — 적용 시 회전점 라운지 입구 안착 확인(aniso_overlay.png).

### 고정 계약 (PM 선고정, Phase 1 병렬화 전제)
- `estimate_wall_scale`: straddle 쌍 없으면 `(None, info{fail})` — 비straddle 스케일 반환 금지. 근접 프리필터 모듈 내재화(build_coplay 층 중복 제거)
- `scan2bim/metric_scale.py`에 `apply_axis_split_scale(poses_yup, pts_yup, s_h, s_v)` 신설 — diag(s_h,s_v,s_h), **forward/up 재정규화·직교화 필수**
- 복도폭 출처 AXX→**SXX**(AXX엔 수직스팬>1.5m 삼각형 0개, 벽은 SXX 9009개)
- s_h 미가용 시 s_h=s_v 폴백(현행 동일) + **폴백 사유 필드 의무화**
- `realtime/_uploads/**` 전원 쓰기 금지(서빙 html 불가침, 배포는 루프 후 PM)

### 소유권 변경분
- `tools/check_coplay_geometry.py`(신규), `reports/coplay/`(산출물) → coplay-wire

### Track 2 DoD
| ID | 명령 | 기대 |
|---|---|---|
| E | pytest 전체 | 93+신규(≥12) green |
| A-1 | wall_anchor/metric_scale 테스트 | straddle 강제, \|f\|=\|u\|=1·f⊥u 불변식 |
| A-2 | validate --dtdx SXX | 선택폭 ∈[2.5,4.5], s_h ∈[3.2,4.1], exit 0 |
| C | validate 전체 dtdx | s_v ∈[1.6,1.9], s_h/s_v ∈[1.8,2.4], 속도 ∈[0.4,1.4], 폴백 미발생 |
| B-폴백 | pytest -k fallback | 앵커 실패 시 현행 단일스케일과 수치 동일 |
| D-1 | build_coplay 리빌드 | reports/coplay/ 출력, _uploads 변경 0 |
| D-2 | check_coplay_geometry | 회전점 Z≤4, 종점 X∈[−8,4]·Z≤2.5 |

### 반복: initial 12 / +6 / hard_max 30. 리스크 게이트: SXX 외벽 끌림(w∉[2.5,4.5]→exit 2), straddle 전멸→조용한 폴백(픽스처에서 폴백=DoD 실패), 이방 방향벡터 왜곡(불변식+체커 교차대조).

## 재생성/갱신

    /TeamPM [목표]   ← 이 문서와 config.yml의 team 블록을 함께 갱신
