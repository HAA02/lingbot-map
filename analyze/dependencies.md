# 의존 관계 (2026-06-12)

- server.py → lingbot_map.bim.alignment(solve_sim3_umeyama), registration(icp_refine_rigid), inference_worker(GPU, 실패 허용)
- server.py 모델 파이프: _model_registry → _load_glb_native_model(실정점 디코드)/_load_model_objects(PAG) → _model_manifest → coverage/정합/자동배치
- coverage.html → /api/models, /api/uploads/{id}/{scan,alignment(+candidates),coverage(+analyze,status,report,review),keyframes} → three.js CDN(인터넷 필요)
- smoke → 라이브 서버 + 픽스처 2종
