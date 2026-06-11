# 주요 기능 (2026-06-12)

1. 영상 업로드 → GPU 재구성(LBP/LBM 점군+포즈+썸네일) — inference_worker
2. 모델 레지스트리 — models/*.glb 자동발견, GLB-native 메타데이터(IfcGUID/Category/실정점 bbox, 노드별 객체+접미사 guid), manifest 생성
3. scan-to-model 정합 — 수동 대응점(Umeyama Sim3, green/yellow/red 게이트) + 자동배치(중력정렬·metric scale·yaw prior·클러터 제거·표면반경 채점) + prior corroboration/ICP
4. coverage 분석 — 객체별 observed/likely/uncertain/not/out_of_scope (visibility·support·표면반경 보정)
5. 웹뷰 — 모델 GLB 직접 색칠(하이브리드 proxy 폴백), 카메라 경로/frustum, 썸네일 타임라인, 대응점 클릭, 자동배치 버튼, 검수 저장
6. 리포트 — 90% 게이트(자동+수동), blocker 표시
