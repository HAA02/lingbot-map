# 상업용 정합 보조와 점군 부재 치수

작성: HAA02 / 2026-10-07 / feat/auto-scale-fusion

## 완료

- 상업 라이선스 비교 `docs/commercial-library-comparison.md`. 저장소 LICENSE를 2026-10-07에 조회. COLMAP 본체가 궤적 형상에 가장 유리하고, plan-match+DXF가 BIM 배치의 정본, gsplat은 데모.
- 미연결 라이브러리: `scan2bim/shell_icp.py`, `colmap_poses.py`, `refine_decision.py`, `shell_refine.py`. `accept_id` 없으면 HOLD. 테스트 21건 통과 후 부재 테스트와 합쳐 26 passed.
- 점군 xyz에서 기둥·배관 지름·길이·연결: `scan2bim/members.py`, `ply_xyz.py`, `tools/extract_members.py`. 가우시안 PLY는 means만. 합성 장면(둘레가 다 있는 원통)에서 기둥 1·배관 2·조인트 1. 벽 면은 배관으로 안 잡힘.
- 전역 메모리 5건(단안 스케일, PreFlight, PiP 데모, dtdx X반전, scan2bim 방향)을 주제 페이지로 회수.

## 미완

- 새 모듈은 `place_rigid` / 업로드 / coverage 저장 경로에 없음. `--plan-match off` byte-identical은 유지(그 파일을 이번 연결에 쓰지 않음).
- 영상 업로드만 재구성까지. ply·가우시안 업로드 없음. 정합 미저장 시 `raw_scan_local`.
- 배관 치수는 합성 풀서라운드만 확인. 한 면만 보이는 실제 스캔은 미측정.
- COLMAP 바이너리, Open3D, TEASER++, gsplat은 이 머신에 없음.
- `data/refsite3d/`(약 2.7GB)는 커밋하지 않음.
- 전체 `pytest tests/ -q`는 이번 세션에서 다시 돌리지 않음. 기준선은 2026-07-31의 369 passed.

## 다음

1. `--shell-refine`를 opt-in으로만 붙이고, 플래그 없을 때 배치 출력은 그대로.
2. 같은 업로드에서 COLMAP 궤적과 피드포워드의 축별 스케일 비를 측정.
3. `extract_members`를 실제 단면 점군으로 재측정한 뒤에야 업로드에 연결.

## 함정

- Open3D는 Python 3.14 휠이 없어 ICP 기본은 numpy.
- 가우시안 scale/opacity는 배관 지름이 아님.
- s_f≈3.53은 DXF 단일소스. 문통과 교차검증은 이 데이터에 문이 없어 종료된 상태.

## 결정

- 점군·가우시안으로 모델 생성까지 가더라도 저장소를 새로 파지 않고, 현 브랜치에 보조 경로로 둔다. 자동 전역정합 확정은 열지 않는다.
- Inria 3DGS, SuGaR, GaMeS, DUSt3R, MASt3R, InstantSplat 기본 경로, VGGT는 제품에서 제외.
