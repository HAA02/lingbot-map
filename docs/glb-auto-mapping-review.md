# 검토: GLB 기준 영상 위치 자동매핑 (좌표계 규명 + 설계안)

작성일: 2026-06-11
선행 문서: `model-coverage-webview-prd.md`, `scan-to-model-improvement-prd.md`
검토 질문: **촬영 영상의 특정 위치를 PXX.glb 기준으로 어떻게 자동 매핑할 수 있는가?**

---

## ① 검증된 사실 (전부 실측, 부록에 프로브 수치)

### F1. GLB↔PAG 변환 확정: `PAG_m = 0.3048 × GLB_ft` (이동 0, 축 동일)

GUID로 1:1 대응시킨 **267쌍**(GLB 노드 bbox-center ↔ PAG center)을 노드 행렬 적용 후 비교한 결과:

| 카테고리 | n | H1 잔차 median (m) | p90 (m) | max (m) |
|---|---|---|---|---|
| pipes | 132 | **0.0000** | 0.0000 | 1.0201* |
| elbows | 63 | 0.0045 | 0.0228 | 0.0317 |
| reducers | 35 | 0.0036 | 0.0694 | 0.0700 |
| mepAccessories | 24 | **0.0000** | 0.0000 | 0.0000 |
| valves | 11 | 0.0139 | 0.0177 | 0.0177 |
| tees | 2 | 0.0239 | — | 0.0239 |

\* pipe 이상치(최대 1.02m, 5건)는 경사/곡선 배관에서 bbox-center ≠ 구간 중점인 정의 차이. 전체 75분위 잔차 **4.5mm**.

Umeyama 자유 풀이(회전·스케일 비고정)도 동일 결론: scale **0.3078**(≈0.3048, bbox-center 노이즈 내), rotation ≈ **I**(최대 비대각 0.00175), translation ≈ **0**.

**의미: GLB 내부 좌표는 Revit 내부단위(피트)·Z-up이며, PAG mm와 같은 원점·같은 축이다. 변환은 상수 하나(0.3048)뿐이다.**

### F2. GLB root 노드가 단위·축 변환을 전담

`PXX.glb`(generator: `Revit2GLTF by cowboy1997`)의 root 노드 행렬(column-major):
```
x' = 0.3048·x,  y' = 0.3048·z,  z' = −0.3048·y
```
즉 root가 **피트→미터 + Z-up→Y-up(glTF 표준)** 을 한 번에 수행. three.js가 렌더하는 glTF 월드좌표 = `(x_pag, z_pag, −y_pag)` (미터).

### F3. 이전 세션의 "12m 불일치 블로커"는 측정 아티팩트였다 (해명 완료)

- 원인: **trimesh가 이 GLB의 Draco(KHR_draco_mesh_compression) 정점을 디코드하지 못하고 0으로 채움** — 173개 geometry 전부 정점 mean/min/max = (0,0,0) 실측.
- 따라서 이전 "centroid" 비교는 실제로는 *노드 이동값 vs PAG center* 비교(무의미)였고, 12m·축분포 불일치 모두 여기서 발생.
- 올바른 프레임(accessor min/max 메타데이터 + 노드 행렬)으로 재실행한 것이 F1이며 잔차 0으로 수렴. **실데이터 문제가 아니었다.**

### F4. (파생 발견) viewer에서 GLB가 proxy 대비 −90°(X) 돌아가 표시 중 — 실재 버그

`coverage.html`은 Z-up 장면(`camera.up.set(0,0,1)`, 224행)에 PAG 미터 좌표로 proxy/경로를 그리지만, GLB root는 회전 보정 없이 추가됨(`loadModel`) → glTF Y-up 그대로 렌더. **docs에 기록된 "모델 축이 어긋나 보임" 증상의 원인.**
수정: GLB 추가 시 `root.rotation.x = Math.PI / 2` 1줄 (→ GLB 표시좌표 == PAG 미터 == proxy 좌표).

### F5. GUID 구조 (보강 사실)

- UniqueId 보유 노드 268개 중 **132개(pipes)는 mesh 직접 보유 + local transform 전부 identity**(정점에 위치 baked, 피트 단위).
- **136개(피팅류: elbow 63·reducer 35·mepAcc 24·valve 11·tee 2·generic 1)는 자식 노드가 mesh+행렬 보유** → 객체 위치 계산 시 자식 행렬 적용 필수.
- 서버측 정점 샘플링에는 Draco 디코더 필요(trimesh 불가 — F3). 단 **accessor min/max + 노드 행렬만으로 객체별 world bbox는 디코더 없이 정확히 얻을 수 있음**(F1이 그 증명).

---

## ② 좌표계 규명 결과: "영상 위치 → GLB" 변환 사슬

영상의 특정 위치(스캔 점, 카메라 포즈)를 GLB에 올리는 전체 사슬은 이제 **모든 상수가 확정**됐다:

```
p_scan (영상 재구성 local)
  → [Sim(3) scan-to-model 정합]        ← 유일하게 자동화가 필요한 단계
p_model (PAG 미터, Z-up) == model_world
  → ÷0.3048                            (확정 상수)
p_glb_ft (GLB 내부 좌표, 피트, Z-up)
  → three.js 표시: (x, z, −y)_model    (root 행렬, 확정)
```

**결론: "GLB 기준 매핑"과 "PAG 기준 매핑"은 같은 문제다.** GLB와 model_world는 동일 프레임(상수 변환)이므로, 자동매핑의 본질은 기존과 동일하게 **scan→model Sim(3) 정합의 자동화**이고, 정합만 생기면 GLB 위 표시는 공짜다.

---

## ③ 자동매핑 옵션 비교 (영상 위치 → 모델/GLB 정합의 자동화)

반복배관 264개 제약(영상-only 전역 자동확정 금지)은 유지한다. 모든 옵션은 후보 생성→게이트→(필요시) 사람 확인 구조.

| | **옵션 A. 마커 기반** ★추천(신규 촬영) | **옵션 B. 이전 정합 prior + ICP 검증** ★추천(반복 촬영) | **옵션 C. 1-anchor + 기하 RANSAC** | 옵션 D. 시각 장소인식 |
|---|---|---|---|---|
| 방식 | AprilTag/QR를 알려진 BIM 좌표에 부착, 영상에서 자동 검출→결정적 정합 | 같은 모델·구역의 검증된 green/yellow 정합을 초기값으로 적용 후 GLB/PAG 표면 ICP로 검증·정밀화 | 사용자가 구역 선택 or 대응점 1개 → subset 내 pipe-axis 디스크립터 RANSAC Sim(3) 후보 | GLB 렌더 뷰 vs 영상 keyframe 매칭 |
| 자동화 수준 | **완전 자동**(마커 사전 부착 필요) | **거의 자동**(첫 1회만 수동 정합) | 반자동(힌트 1회) | 자동(연구개발 필요) |
| 필요 입력 | 마커 2개+ BIM 좌표 등록 | 동일 구역 선행 정합 1건 | zone/level 선택 or anchor 1개 | 대규모 렌더링·학습 |
| 반복배관 내성 | **면역**(절대 좌표) | 높음(구역 고정) | 중간(subset 축소+margin 게이트로 보강) | 낮음~중간 |
| 실패 모드 | 마커 미검출/가림 → `needs_anchor` 폴백 | 구역 오판·드리프트 → ICP RMSE 게이트로 reject | 후보 동률 → `ambiguous_alignment` | 유사 구간 오인식 |
| 코드 접점 | `save_alignment`(server.py:2356), inference 파이프라인에 검출기 추가 | `_previous_alignment_candidates`(server.py:1330), `icp_refine_rigid`(registration.py:68), `_candidate_from_alignment`(server.py:1271) `can_save` 승격 | `alignment_candidates`(server.py:1671), `_model_repetition_risk`(server.py:1214), 개선 PRD FR-B1/B2 | 신규 모듈 |
| 선행 조건 | 현장 마커 운영 절차 | **F4 수정 + Draco 디코더(또는 PAG 샘플 표면)** | LBP4 점군 디스크립터 구현 | 제외(후순위) |

권장 로드맵: **B를 먼저**(코드 자산 재사용 최대, 이미 설계된 FR-A4·A3와 동일) → **A 병행**(신규 촬영 SOP) → C는 B/A로 못 푸는 "처음 찍는 구역 + 마커 없음" 케이스 한정.

ICP 표면 소스: 서버측 GLB 정점은 Draco 디코더(`DracoPy` 등) 설치 또는 비압축 재export가 필요. **대안: PAG start/end/bbox에서 샘플한 표면점**으로도 ICP 가능(현재 coverage 샘플러 재사용) — 디코더 없이 옵션 B 착수 가능.

---

## ④ Go / No-Go 결론

| 판단 | 결정 | 근거 |
|---|---|---|
| GLB를 **표시·evidence 기준**으로 사용 | **GO** | F1·F2로 GLB==model_world 확정(잔차 mm급). viewer 1줄 수정(F4)으로 GLB·proxy·경로·coverage가 한 프레임에 정확히 겹침 |
| GLB를 **PAG를 대체하는 별도 기준좌표계**로 채택 | **NO-GO** | 두 좌표계는 동일 프레임이라 "대체"가 무의미. 또한 GLB는 507객체 중 268개만 포함 — 객체 메타데이터 정본은 PAG 유지, GLB는 mesh 형상 공급원 |
| 영상-only 무제약 자동 전역정합 | **NO-GO 유지** | 반복배관 264개 제약 불변. 자동화는 옵션 A/B/C의 anchor·prior·marker 경로로 |
| 다음 구현 1순위 | **F4 viewer 회전 수정 → 옵션 B(prior+ICP)** | 1줄 수정으로 즉시 시각 정합성 확보, B는 기존 자산으로 최단 구현 |

---

## ⑤ 부록: 프로브 명령·수치

프로브 스크립트(임시, `$CLAUDE_JOB_DIR/tmp/`): `probe_glb_pag.py`, `probe2_vertices.py`, `probe3_transform.py`

핵심 수치:
- root 행렬(column-major): `[0.3048,0,0,0, 0,0,-0.3048,0, 0,0.3048,0,0, 0,0,0,1]`
- trimesh Draco 디코드 실패 실측: 173/173 geometry 정점 mean=min=max=(0,0,0)
- accessor 기반 GLB world-ft 정렬 spans(×0.3048): [3.142, 8.133, 8.578] m vs PAG [3.142, 8.638, 9.781] m
- H1(`PAG=0.3048×GLB+0`) 전체 잔차 quantiles [0,25,50,75,100] = [0.0, 0.0, 0.0, 0.0045, 1.0201] m
- Umeyama 자유 풀이: rmse 0.1208 / scale 0.307770 / rotation≈I / translation [0.0995, −0.0961, −0.2659]
- viewer Z-up 증거: `coverage.html:224 camera.up.set(0,0,1)`, GLB root 회전 보정 부재(`loadModel`)
- 표본: 267쌍 (≥20 충족), 카테고리 6종 전수
