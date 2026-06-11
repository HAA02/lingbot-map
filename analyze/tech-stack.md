# 기술 스택 (2026-06-12)

- Python 3.14 / .venv
- FastAPI 0.136 + uvicorn 0.47 (HTTP, --no-ssl 개발모드, 기본 8443 SSL)
- torch 2.10 cu128 (추론 워커 — GPU OOM 시 내결함 기동: coverage 웹은 워커 없이 동작)
- numpy 2.4 / scipy 1.17 (cKDTree — corroboration/coverage) / trimesh 4.12 (주의: Draco 디코드 불가)
- three.js 0.160 (unpkg CDN, importmap) + GLTFLoader + DRACOLoader(CDN wasm)
- playwright 1.60 + 시스템 Edge (헤드리스 검증, --enable-unsafe-swiftshader)
- 테스트: tools/coverage_web_smoke.py (라이브 서버 대상 게이트 5종)
