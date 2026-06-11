# 네이밍 (2026-06-12)

- Python: snake_case, 내부 헬퍼 `_언더스코어`, FastAPI 핸들러는 동사형
- 업로드 산출물: `upload_{ts}.{model_id}.{종류}.json` (모델 스코프) + 레거시 `upload_{ts}.{종류}.json`
- 모델: models/{이름}.glb + {이름}.model_manifest.json, 레지스트리 id = 소문자 파일명
- guid: GLB-native는 IfcGUID, 중복 시 `base@k` 접미사 (서버·뷰어 동일 규칙)
- JS: camelCase, 상태색 STATUS 맵, 디버그 훅 window.__cov
