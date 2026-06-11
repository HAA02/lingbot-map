# 모델 기준 영상 Evidence 작업 요약

갱신일: 2026-06-12 (AutoPM-260612)
이전 버전은 PXX/PAG 시절 내용 — git 히스토리 참조.

## 현재 아키텍처

- 기준 모델: `models/*.glb` 자동 발견 (GLB-native: PAG JSON 불필요, IfcGUID/Category/실정점 bbox)
  - 현재 등록: `pipe_duct` (454 객체, GUID 매핑 100%, 콘텐츠 높이 z≈27m)
  - 익스포터 특성 대응: accessor min/max Y 오기록→실정점 디코드, IfcGUID 타입공유→노드별 객체(base@k)
- 정합 경로 3종:
  1. 수동 대응점 4~6개 → Umeyama Sim(3) → green/yellow/red
  2. **자동 배치**(원칙): 중력정렬 + metric scale(제자리 팬 실측 근거) + yaw prior(배관 방위) +
     클러터 평면 제거 + 표면반경 보정 채점 → yellow(med≤0.25·inlier≥0.50)/red 후보 → 명시적 저장
  3. prior 부트스트랩: 같은 모델의 검증 정합 + corroboration/ICP → can_save 승격
- coverage: 객체별 observed/likely/uncertain/not/out_of_scope (표면반경 가산 판정)
- 뷰어: GLB mesh 직접 색칠(쓰레기 bbox 재계산 포함), 경로/frustum, 자동배치 버튼,
  후보 패널 corroboration 수치, 대응점 1개 = 자동배치 시작 힌트

## 검증 체계

- `tools/coverage_web_smoke.py` — 6게이트: static/manifest/unaligned/candidates/stale-red/auto-place
  - StaleFixture가 픽스처를 설치/복원 → 데모·실사용 상태와 무관하게 결정적
- playwright + Edge 헤드리스(웹뷰 실측: per-object 454/454 오차 0, 버튼 클릭 E2E)
- 디버그 훅 `window.__cov`

## 알려진 한계 / 다음 항목

- 자동배치는 가설(반복 랙에서 위치 모호) — 시작 힌트 1클릭 또는 후보 미리보기 확인이 보정 경로
- M5(장기): keyframe 시각 특징 결합 — PRD Tier-2 FR-B2
- server.py(~3.4k줄)·coverage.html(~2.3k줄) 모듈 분리 — 리팩토링 후보
- 진짜 현장 정합(사용자 확인) 1건 확보 시 observed(녹색) 게이트까지 실검증 가능
