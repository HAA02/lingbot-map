# 리팩토링 후보 (2026-06-12)

- 🔴 server.py ~3,400줄 단일 파일 — 모델 레지스트리/정합/coverage/검수 모듈 분리 가치. 단 동작 변경 리스크로 단계적 접근
- 🔴 테스트 갭: 자동배치(auto_place)·candidate 저장·표면반경 채점에 자동 회귀 테스트 없음 (smoke 미커버) ← 최우선
- 🟡 coverage.html ~2,300줄 — JS 모듈 분리(현재 단일 인라인 스크립트)
- 🟡 후보 UI에 corroboration 수치(med/inlier/ICP) 미표시 — API에는 있음 (M4)
- 🟡 _load_scoped_upload_json 호출부들이 model_id 기본값 계산 중복
- 🟢 viewer.html(구 뷰어)와 coverage.html 기능 중복 일부
- 🟢 docs/coverage-progress-summary.md가 PXX 시절 내용 — 갱신 필요
