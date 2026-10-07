# commercial-recon

> 범위: 상업 라이선스로 영상·점군을 BIM에 올리고, 점군만으로 배관 치수를 읽는가 · 갱신 2026-10-07 · 근거 handoff: handoff/2026-10-07-commercial-recon.md

## 정본

- 비교 결론은 `docs/commercial-library-comparison.md`. 궤적 형상은 COLMAP, BIM 배치는 기존 plan-match+DXF, 조임은 numpy ICP(Open3D 휠 없음), 가우시안은 데모.
- 코드는 업로드에 안 붙어 있다. `shell_icp` / `colmap_poses` / `refine_decision` / `shell_refine`는 `accept_id` 없이 HOLD만 한다.
- `scan2bim/members.py`는 xyz에서 기둥·배관 지름·길이·끝점·조인트를 낸다. 가우시안 PLY는 `ply_xyz.py`가 중심점만 읽는다. 합성 풀서라운드에서 통과(26 passed에 포함). 단면 스캔은 미측정.
- 업로드는 `video/*`만. ply·splat 업로드는 없다. 정합을 저장하기 전에는 coverage 분석이 열리지 않는다.
- 2026-10-07 이 머신에 COLMAP, Open3D, TEASER++, gsplat 없음.
- 제품에서 제외: Inria 3DGS, SuGaR, GaMeS, DUSt3R, MASt3R, InstantSplat 기본 경로(MASt3R 의존), VGGT.

## 할 일

- `tools/build_coplay.py`에 `--shell-refine` opt-in. 플래그 없을 때 출력은 지금과 같아야 한다. `tools/build_coplay.py:1552`
- 같은 업로드에서 COLMAP 궤적과 피드포워드의 `s_h`/`s_f` 비를 측정.
- `extract_members`를 실제 단면 점군으로 재측정한 뒤 업로드에 연결.

## 함정

### numpy ICP는 근처 초기값이 있을 때의 조임이다 (2026-10-07)

전역 배치를 대신하지 않는다. `accepted`는 항상 거짓이고, 확정은 `decide_refine`의 명시 id만.

### 가우시안 스케일은 지름이 아니다 (2026-10-07)

PLY의 scale·opacity를 배관 직경으로 쓰지 않는다. 한 면만 보이면 축이 그쪽으로 기운다.

## 결정

### 2026-10-07 · 현 저장소에 보조 경로로 둔다

점군·가우시안 모델 생성은 새 저장소로 갈라지 않는다. 자동 전역정합 확정은 열지 않는다.

### 2026-10-07 · 상업 금지 스택은 제품 의존에서 제외

→ 대체: 같은 날짜 commercial-recon 정본의 제외 목록. 코어 SPDX가 Apache여도 NC 의존이 있으면 그 경로는 제외한다.
