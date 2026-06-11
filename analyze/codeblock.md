# 수정 금지/주의 (2026-06-12)

- ❌ `realtime/_uploads/upload_1779439687108.*`, `upload_1779442357085.*` — smoke 픽스처 (unaligned / stale-red). 변경 시 smoke 깨짐. 데모 적용 후엔 백업 복원 필요
- ❌ `lingbot_map/` ML 코어 (aggregator/heads/models) — coverage 작업 범위 밖, GPU 의존
- ⚠️ `realtime/server.py`의 안전 게이트: red 정합 observed 금지, 반복모델 anchor 요구, can_save 게이트 — 완화 금지
- ⚠️ `models/pipe_duct.glb` — 익스포터 특성(accessor min/max Y 오기록, IfcGUID 타입공유)이 코드에 반영돼 있음. 재export 시 재검증 필요
- ⚠️ 검증 명령: `python tools/coverage_web_smoke.py --base-url http://127.0.0.1:8767 --model pipe_duct --upload upload_1779439687108 --stale-upload upload_1779442357085`
