# scout: 현장↔BIM 매핑 부정확 원인 탐색 (2026-07-30)

질문: 스케일/회전/이동거리가 현장 기준으로 Model 위치를 잘못 찾는다.
① 특징점 매칭으로 BIM 트래킹 카메라를 처리하는가 ② 연산을 모서리에만 집중해도 되는가 ③ PiP 영상 교체.

---

## ① 특징점 매칭 기반 카메라 트래킹 — **없음**

현재 재구성은 **피드포워드 신경망 1패스**다. 특징점 검출·매칭·BA·loop closure가 전무하다.

- `lingbot_map/heads/camera_head.py:23-76` — 트랜스포머가 카메라 pose/FoV를 **직접 회귀**
- `lingbot_map/heads/dpt_head.py:21-50` — DPTHead가 밀집 depth 예측
- `realtime/inference_worker.py:343` — `enable_point=False → depth unprojection` 경로
- `realtime/inference_worker.py:553`, `lingbot_map/utils/geometry.py:25-54` — depth를 pose로 unproject
- `realtime/inference_worker.py:50-110` — `_voxel_fuse_weighted`: 신뢰도 가중 복셀 융합(1.5cm/8mm)
- `benchmark/datasets/general.py:77` — COLMAP SIFT는 **벤치마크 GT 생성 전용** (런타임 아님)
- `scan2bim/localize.py:107` — `cv2.solvePnPRansac` 존재하나 입력이 **객체 앵커 검출**, 특징점 아님

### 정합은 "기하 통계 맞추기"이지 트래킹이 아니다
`tools/build_coplay.py` 3개 배치모드 — `place_registered:633`(ICP), `place_pipe_auto:1101`, `place_rigid:862`

| 성분 | 출처 | 파일:줄 |
|---|---|---|
| s_v (수직) | 천장고 + 카메라높이 2앵커 융합 | `build_coplay.py:1002,1141` |
| s_h (측방) | DXF 복도폭 / recon 간격, 게이트 (1.8,2.8) | `build_coplay.py:408,452`, `scan2bim/dxf_plan.py:135` |
| s_f (진행) | DXF 복도끝 / leg-A arclen | `build_coplay.py:826,1058-1075` |
| yaw | leg-A 라인핏 → FXX run 정렬, 폴백 천장 PCA | `build_coplay.py:1014-1027` |
| chi | recon L턴 vs 모델 FXX 가지 handedness | `build_coplay.py:990-993` |
| translation | anchor 고정 또는 궤적중심=모델중심 | `build_coplay.py:1033-1041` |

**영상과 BIM 사이에 픽셀 단위 대응(correspondence)이 하나도 없다.** 모델 기하 통계(복도폭·천장고·PCA축·L턴 방향)와 궤적 기하 통계를 맞추는 방식이다.

### 이것이 오차의 근본 원인
1. **monocular feed-forward = up-to-scale.** 절대 스케일 정보가 원리적으로 없어 DXF/천장고에서 외부 주입해야 한다.
2. **s_h≈1.97 vs s_f≈3.53 (1.8배 차이)** = pose가 방향별로 다르게 틀렸다는 직접 증거. 등방 스케일 하나로는 못 고쳐서 이방성으로 우회한 상태.
3. **재투영 오차 최소화 단계가 없음** → drift/smearing이 pose에 그대로 남는다. 후단 `place_rigid`는 강체(scale+R+t)라 궤적 **내부 형상 왜곡은 원리적으로 보정 불가**. `coplay-desmear-01`이 이 증상 패치였고 전제 불일치로 실패.

### 권고 (2갈래)
- **(A) 현 구조 유지·최소개선** — pose 사후 최적화: 벽/천장 평면 제약 + DXF 복도 중심선에 궤적을 non-rigid 스냅. 증상 완화, 근본 미해결.
- **(B) 특징점 트래킹 도입 (근본)** — 2단계
  1. *영상 내부 일관성*: keyframe에 SuperPoint+LightGlue(또는 ORB) → 상대 pose → pose-graph/BA로 drift 제거. up-to-scale은 남지만 형상이 metric-consistent 해져 **단일 스케일로 해결 가능** (s_h/s_f 분리 불필요).
  2. *BIM 대응*: BIM에서 렌더한 코너·엣지(문틀, 벽-천장 교선)를 2D 특징으로 던져 `solvePnP`로 절대 pose 확정 = "BIM 트래킹 카메라"의 정석. **`scan2bim/localize.py:107`이 이미 씨앗** — 앵커 입력을 BIM 렌더 엣지로 확장하는 지점.

---

## ② 연산 집중 — 현재 균일, "모서리에 **더**"는 정답 / "모서리**만**"은 역효과

### 현황: 균일 샘플링
- `demo.py:479-487` — auto keyframe_interval: 총 프레임 기준 최대 320장
- `gct_profile.py:118` — `(i - scale_frames) % kf_int == 0` 모듈로 균일 선별
- `realtime/inference_worker.py:34-47` — `_voxel_downsample` 0.025m 균일 복셀
- `realtime/inference_worker.py:566-567` — `max_points_per_frame=270000` 후 랜덤 샘플
- `realtime/inference_worker.py:287-288` — 윈도우 32프레임(64→감소, 오버플로 방지)
- 속도 게이트(정지/저속 프레임 배제) — **미발견**

### 코너 적응형은 후처리 기하뿐, 연산 밀도는 동일
- `scan2bim/turn_desmear.py:80-208` `desmear_turn()` — 회전 window 내에서만 heading 재분배
- `scan2bim/pipe_path.py:67-82` `trajectory_turn_fraction()` — 두 직선 분할로 주 회전점 감지

### 판단
**직선 구간을 누락하면 안 된다** — s_f가 복도 직선 구간의 arclen에서 나온다(`build_coplay.py:826`). 직선 프레임이 없으면 이동거리 오차를 측정할 근거 자체가 사라진다. 문 통과 타이밍 프레임도 s_f 교차검증(미실행 항목)에 필요하다.

**코너는 지금 정보량이 부족한 곳이다** — 급회전이 9초에 걸쳐 스미어링된 것은 회전이 제대로 관측되지 않았다는 뜻.

→ **올바른 형태: 각속도(yaw rate) 가중 적응형 키프레임.** 총 프레임 수는 320 그대로 두고 배분만 바꾼다. 회전 중 5°마다 1장 / 직선 중 0.5m마다 1장. 최소 변경 지점은 `demo.py:479-487`의 균일 interval → 가중 재배분.

- 안전한 절감: 정지 구간(속도 게이트 신규), 중복 시야(voxel이 이미 커버)
- 절감 불가: 직선 구간 자체, 문 통과 프레임

---

## ③ PiP 영상 — 배선은 완성, **산출물이 없어서 raw로 fallback**

- `realtime/coplay.html:32` — `<video id="rvid" src="../out_kakao_fmt/..._demo_format.mp4">` (템플릿 기본값)
- `realtime/server.py:189-199` `_serve_per_upload_coplay()` — regex로 모든 `<video src>`를 `/api/uploads/{id}/video`로 재작성
- `realtime/server.py:1436-1449` — 라우트: `_demo_video_path()` → 없으면 `_upload_video_path()` **raw fallback**
- `realtime/server.py:1264-1267` `_demo_video_path()` — `_uploads/_demo/{id}/{id}_demo_format.mp4`, `p.exists()`만 검사
- `tools/render_demo_format.sh:51` — 출력 `${OUT}/${STEM}_demo_format.mp4` (follow 렌더 → birdeye → ffmpeg 합성)
- `demo_render/batch_demo.py` — 점군 렌더 엔진 (GCTStream + CUDA)

### 실측 파일 상태 (원인 확정)
| upload | coplay.html | demo_format.mp4 | PiP 실제 표시 |
|---|---|---|---|
| `upload_1779442357085` | 있음 (13MB) | **없음** (_demo 폴더만) | raw 117MB 폰영상 ← **증상** |
| `upload_1781521406685` | 있음 | **100 bytes** (깨진 스텁) | exists() 통과 → 검은화면 |
| `upload_1782706997880` | 있음 | 24MB 정상 | 정상 점군 데모 |

**코드 수정 불필요.** 필요한 것:
1. `tools/render_demo_format.sh`로 `upload_1779442357085` 데모 렌더 생성
2. 100바이트 스텁 삭제·재생성
3. (권장, 3줄) `_demo_video_path()`에 최소 크기 검사(`p.stat().st_size > 1_000_000`) — 깨진 스텁이 fallback을 막는 버그 차단

---

## 종합

- 매핑 부정확의 근본은 **특징점 기반 트래킹의 부재**다. pose가 피드포워드 회귀 1패스로 나오고 재투영 최적화가 없어 방향별로 다르게 틀리며(s_h 1.97 vs s_f 3.53), 후단 강체 정합으로는 궤적 내부 왜곡을 고칠 수 없다. 스케일 상수를 더 정교하게 맞추는 것으로는 한계에 도달했다 — `localize.py`의 solvePnP를 BIM 렌더 엣지 대응으로 확장하는 것이 정공법.
- 연산 절감은 "모서리만"이 아니라 **각속도 가중 재배분**이어야 한다. 직선 구간은 s_f 앵커의 유일한 근거이므로 누락 금지.
- PiP는 이미 점군 데모를 우선 서빙하도록 배선돼 있고, 해당 업로드의 렌더 산출물이 없어서 raw로 떨어진 것뿐이다.
