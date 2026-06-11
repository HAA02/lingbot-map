# 프로젝트 구조 (2026-06-12, AutoPM)

- `lingbot_map/` — 코어 ML 패키지 (재구성 모델: aggregator/heads/layers/models/utils/vis)
  - `bim/` — BIM 정합·메타데이터 (alignment.py: Umeyama Sim3 솔버 — coverage 핵심 의존)
  - `export/` — scene export
- `realtime/` — FastAPI 서버 + 웹뷰 (제품 본체)
  - `server.py` (~3,400줄) — 모든 API: 모델 레지스트리(GLB-native 동적), scan-to-model 정합, 자동배치, coverage 분석, 검수
  - `coverage.html` (~2,300줄) — three.js 모델 기준 evidence 뷰어 (DRACOLoader, 자동배치 UI)
  - `coverage_report.html`, `upload.html`, `audit.html`, `viewer.html`, `index.html`
  - `inference_worker.py` — GPU 영상→점군 재구성 (LBP/LBM)
  - `registration.py` — 윈도우 스티칭 ICP (icp_refine_rigid 재사용됨)
  - `_uploads/` (gitignore) — 업로드 영상·산출물·정합/coverage JSON
- `models/` — 기준 GLB (`pipe_duct.glb` 자동발견) + `old/`(구모델 아카이브, gitignore)
- `tools/` — `coverage_web_smoke.py` (핵심 회귀 게이트), pag_to_bim 등
- `docs/` — PRD 3편 + GLB 좌표 규명 보고서
- `ckpts/`, `out/`, `video/` (gitignore) — 가중치/출력
