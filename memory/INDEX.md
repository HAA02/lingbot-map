# lingbot-map — 작업 지식 진입점

주제는 목차의 `topics/` 한 장만 연다. handoff/·archive/ 는 통째로 읽지 않는다.
레거시(동결, 새로 쓰지 않음): `gotchas.md` · `decisions.md`. 필요한 절만 `grep -n`.

## 지금 (2026-10-07)

브랜치 `feat/auto-scale-fusion`. 정본 커밋은 이 기록 직전 `7f62a80`.
검증: 신규 모듈 26 passed. 전체 pytest는 2026-07-31 기준 369, 이번엔 전수 재실행 안 함.
진행: 상업용 ICP·COLMAP 파서·HOLD 게이트와 점군 배관/기둥 치수를 넣었다. 업로드와 `place_rigid`에는 미연결.
다음: [commercial-recon](topics/commercial-recon.md) opt-in 배선 → [plan-match](topics/plan-match.md) 실도면 DXF → [monocular-scale](topics/monocular-scale.md)의 단일소스 `s_f`는 유지.
최신 handoff: [2026-10-07 commercial-recon](handoff/2026-10-07-commercial-recon.md)

## 목차

- [plan-match](topics/plan-match.md) — 현장≠BIM일 때 T≈25%까지 자동, 그 이상은 HOLD
- [commercial-recon](topics/commercial-recon.md) — 상업 스택, 미연결 정합 모듈, 점군 부재 치수
- [monocular-scale](topics/monocular-scale.md) — 단안 스케일 `s_h`/`s_f`와 대응점 확정
- [coplay-demo](topics/coplay-demo.md) — PiP는 H.264 점군 데모
- [coordinates](topics/coordinates.md) — dtdx X 반전, 모델 Z-up, COLMAP 프레임
