# Agent Team: coplay-fwdscale-01

목표: `upload_1781521406685`을 화면에 정확히 안착. 사용자가 실제 이동경로를 모델 위에 직접 그려 보내면서 이전 두 트랙(coplay-rigid-01, coplay-desmear-01)의 핵심 전제(단일 s_h로 충분·t=13.5s가 진짜 코너·라운지는 직선)가 무효화됨 — 근본 재설계.

## 사용자 그라운드트루스 (재검증 안 함, 입력으로 고정)
1. 복도(leg A)가 지금 모델링보다 훨씬 길다(현재 회전점을 한참 지나 계속 이어짐)
2. t=13.5s 목재 개구부는 통과점이었음. 진짜 방향전환은 **t≈19~20s**(라운지 진입, PM이 144포즈 heading 재스캔으로 확인: t=0~18s는 -80~-90도 안정, "iaan LOUNGE" 사인 t=21s와 일치)
3. 라운지 진입 후는 직선이 아니라 **둘러보기**(heading 160도+ 진동 후 거의 원위치, unwrap 순변화 0.9도) — 단일 코너 수렴 모델(desmear) 자체가 부적합
4. 전체 경로(현재 9.86m)가 과소 의심 — 복도(측방향 s_h≈1.97)와 진행방향 스케일이 다를 가능성(heading-relative anisotropy, 이 세션에서 이미 확인된 현상)

## PM 판단 (착수 전 확정)
1. **corner_idx**: 사용자 확정 t≈19.5s 고정입력. 자동검출(heading 표준편차/곡률)은 라운지 wiggle에 latch될 구조적 결함 있어 미채택 — 이 세션에서 2번 발생한 "그럴듯한 오수렴" 재발 방지.
2. **라운지 구간**: recon 자기형상 그대로 보존, desmear 미적용(`--desmear-turn` 이번 경로에서 제외). 강제 직선화/코너스냅은 사용자가 "일어나지 않았다"고 확인한 경로를 날조하는 것.
3. **진행방향 스케일 s_f**: DXF 복도끝(L_end) 앵커 + 문통과 타이밍 교차검증(2개 독립 소스, tol 내 불일치 시 HOLD — 임의채택 금지). 스크린샷 픽셀비는 sanity 대역으로만 참고.
4. **place_rigid 본문 수정 허용**(이전 두 트랙의 "본문 불변" 원칙 파기, 근거: 스케일은 강체변환 이전에 적용돼야 하므로 후처리 스테이지로는 원리적 불가 + 코드 자체가 `build_coplay.py:1013-1015`에서 "per-leg scale not yet implemented"로 자기진단해둔 지점).

## 팀 구조도
```
PM (opus)
├─ P0-Landmark(opus) DXF L_end 확정 + corner_idx=19.5 고정   ┐ 병렬
├─ P0-Gates(sonnet)  게이트 옵션 추가                         ┘
├─ P1-Scale(opus)      forward_scale.py + place_rigid 통합    ┐ 병렬
├─ P1-Crosscheck(sonnet) L_end 독립재도출+문타이밍 s_f 교차검증 ┘
├─ P2-Converge(opus)    자기수렴
└─ P3-QA-final(sonnet)  독립검증+무회귀+라운지보존 적대검증
```

## 역할 및 모델 배정
| 역할 | kind | 모델 | 근거 |
|---|---|---|---|
| pm | pm | opus | 사용자 지정 |
| dev-lead | dev | opus | place_rigid 비가역 내부변경+신규 이방성 스케일 설계 |
| qa | qa | sonnet | L_end 독립재도출·시각/수치 적대검증, 위음성 대가 큼 |

## 파일 소유권
| 파일 | 권한 | 소유 |
|---|---|---|
| `tools/build_coplay.py` | 수정(place_rigid 스케일스텝 이방성 일반화+`--forward-scale`, opt-in) | dev-lead |
| `scan2bim/forward_scale.py`(신규) | 생성(궤적프레임 이방성 스케일+L_end→s_f) | dev-lead |
| `tests/test_forward_scale.py`(신규) | 생성(TDD) | dev-lead |
| `tools/check_coplay_geometry.py` | 수정(`--path-min`, `--post-turn-heading-span-min`, default off) | qa |
| `scan2bim/turn_desmear.py` | 불변(이번 미사용) | — |
| `place_pipe_auto`/`place_registered`/`place_gtpath` 본문, `scan2bim/{wall_anchor,metric_scale,dxf_plan,pipe_path}.py` | 불변(read/재사용만) | — |
| smoke 픽스처, `realtime/_uploads/**`, `lingbot_map/`, 기존 테스트 | 불변 | — |

## Phase
| Phase | 담당 | 내용 | 의존 |
|---|---|---|---|
| P0-Landmark | opus | DXF 복도평행벽 종단+FXX Z범위 교차→L_end 확정, leg-A recon arclen 측정 | — |
| P0-Gates | sonnet | 게이트 옵션 추가(default off, 무회귀) | — (병렬) |
| P1-Scale | opus | forward_scale.py+place_rigid 통합+`--forward-scale`, TDD green | P0-Landmark |
| P1-Crosscheck | sonnet | L_end 독립재도출+문통과타이밍 s_f, dev값과 tol일치 확인 | P0-Landmark (병렬) |
| P2-Converge | opus | 실빌드(`--turn-time-s 19.5 --forward-scale auto`, desmear 없음)→전게이트 자기수렴 | P1-Scale, P0-Gates, P1-Crosscheck |
| P3-QA-final | sonnet | 독립재빌드+무회귀+라운지보존 적대검증+s_f 2소스 재확인 | P2 |

## DoD
빌드: `--auto-rigid --turn-time-s 19.5 --dxf <SXX dxf경로> --forward-scale auto --duration 35.3` (desmear-turn 없음)

| # | 확인 | 기대 |
|---|---|---|
| D1 | 빌드 | 예외없이 HTML, reginfo에 s_h/s_f/L_end/2소스 s_f 로그 |
| D2 | check_coplay_geometry 전 게이트(P0값 대입) | exit 0 |
| D3 | turn_point | Z 유의미 상향(≥L_end_Z−tol), X≈복도중심 |
| D4 | `--post-turn-heading-span-min 120` | 코너이후 heading 총변화 ≥120° (미달=직선화 회귀) |
| D5 | s_f 2소스 일치 | 상대오차 ≤15%, 초과 시 HOLD |
| D6 | s_f 물리대역 | s_h < s_f ≤ 3·s_h |
| D7 | leg-A 보존 | pre-turn-x drift 복도폭 이내 |
| D8 | 무회귀 | `--forward-scale` 미지정 시 byte-동일 |
| D9 | pytest+픽스처+타경로 | 전부 통과 |
| D10 | git diff --stat | 소유 파일만 |

## 반복정책
initial_max=4, hard_max=8. 서킷브레이커: 2개 독립 s_f가 tol 내 수렴 안 되면 튜닝 지속 금지 — PM 재판정 HOLD(억지 일치화는 오수렴).

## 리스크
1. 그럴듯한 오수렴(이 세션 2회 발생) — 2개 독립 s_f 강제 일치(D5), 불일치=HOLD
2. L_end 오지정 — FXX Z범위+문개수 교차확인, QA 독립재도출
3. corner_idx가 라운지 wiggle 포착 — refined index time-fraction 드리프트 확인, 점진진입 시 refine 비활성
4. 라운지 형상왜곡(이방성 스케일 과신장) — heading-span+endpoint 동시게이트, 등방 폴백 문서화

## loop-engineer 연동
`~/.claude/loop-settings/-run-media-iaan-1TB-WD-Github-lingbot-map/config.yml`의 `team:` 블록. 루프 실행: `/loop-engineer`.

## 재생성/갱신
    /TeamPM [목표]   ← 이 문서와 config.yml team 블록 함께 갱신
