# 상업용 라이브러리 비교 — 카메라 영상 기준 BIM 매핑

- 작성일: 2026-10-07
- 상태: 엔지니어링 비교. 법률 자문이 아님. 배포 전 SPDX·법무 재확인.
- 기준 코드: `lingbot_map/` 피드포워드 재구성, `tools/build_coplay.py`의 `place_rigid` / `place_registered`, `scan2bim/coarse_match.py` plan-match
- 근거: `docs/scout-bim-mapping-accuracy.md`, `docs/commercial-license-stack-prd.md`, `memory/INDEX.md`
- 라이선스: 아래 §5. 각 저장소 `LICENSE` 원문을 2026-10-07에 조회.

## 0. 결론

카메라 영상을 기준으로 BIM에 올리는 일에서, 이 목록 안에서는 **COLMAP이 가장 유리**하다. 지금 오차의 본체는 “점군을 BIM에 강체로 못 붙인다”가 아니라, **영상에서 나온 궤적 형상이 방향마다 다르게 늘어나 있다**는 점이다 (`s_h≈1.97`, `s_f≈3.53`).

그 다음 자리는 이미 있는 **plan-match + DXF 스케일**이다. Open3D·TEASER++·PCL은 그 후보를 조이는 2단 정밀화다. gsplat/nerfstudio는 정합이 끝난 포즈 위의 시각 증거이고, 매핑을 대신하지 않는다.

사용자 초안의 “BIM 메시 샘플 → Open3D FPFH/FGR + TEASER++ → ICP → `T`를 GS means에 적용”은 라이선스상 안전하다. 다만 그 파이프는 **궤적이 이미 단일 스케일의 닮은꼴**일 때 성립한다. 이 업로드의 측정은 그 전제를 만족하지 않는다. COLMAP(또는 동등한 BA)으로 형상을 먼저 맞춘 뒤에야 그 조합이 보완이 된다.

## 1. 지금 제품이 하는 일

영상과 BIM 사이에는 픽셀 대응이 없다. 재구성은 피드포워드 1패스(pose·depth 회귀, 특징점·BA·loop closure 없음)이고, 배치는 복도폭·천장고·궤적 길이·L턴 방향 같은 기하 통계를 맞춘다.

일은 세 층으로 갈라져 있고, 라이브러리도 그 층에만 해당한다.

| 층 | 질문 | 지금 구현 | 측정된 한계 |
|---|---|---|---|
| A. 궤적 형상 | 걸은 경로의 상대 기하가 맞는가 | LingBot-Map 1패스 | 단안 up-to-scale. 측방·진행 스케일이 약 1.8배. 급회전이 수 초에 걸쳐 퍼짐. 강체로는 내부 형상을 되돌릴 수 없음 |
| B. BIM 배치 | 그 경로가 도면의 어느 복도인가 | `place_rigid` + plan-match. 현장 변경분 T≈25%까지 자동, 그 이상은 HOLD, 확정오답 0 | 반복 기하에서 자동 확정 금지. 절대 스케일은 DXF |
| C. 증거 | 부재가 보였는가, 데모가 설득력 있는가 | coverage 광선 + 점군 데모 렌더 | 포즈가 틀리면 색칠도 틀린 부재에 떨어짐 |

불변식은 그대로다. red에서 observed 자동 생성 금지, 반복 객체에서 영상-only 전역정합 자동 확정 금지, 정합 전 scan을 모델 위치에 표시 금지.

## 2. 층별 비교

점수: 이 저장소의 카메라 영상 → BIM 매핑에 대한 적합. 일반 점군 정합 성능이 아니다.

| 도구 | 라이선스 (원문) | 맡는 층 | 이 프로젝트에서의 이득 | 한계 | 채택 |
|---|---|---|---|---|---|
| **COLMAP** `model_aligner` | BSD. 본체만. 의존성은 별도 (§5) | **A 우선**, B는 제어점이 있을 때 | 특징점·BA·loop로 재투영 오차를 줄여 `s_h`/`s_f` 분리의 원인을 건드림. 형상이 한 스케일로 모이면 DXF 스케일 하나로 충분해질 여지가 생김 | 절대 스케일은 여전히 없음. 텍스처 없는 복도·모션블러·반복 인테리어에서 SIFT가 실패할 수 있음. `model_aligner`는 BIM 매처가 아니라 제어점 Sim3 | **1순위 추가.** 런타임 폴백 또는 피드포워드 포즈의 형상 보정 |
| **plan-match + `place_rigid`** (현행) | 자체, Apache-2.0 | **B** | 현장≠BIM을 outlier로 버리고, 애매하면 HOLD. 복도 그래프를 알아서 라이브러리 중에 이것뿐 | 궤적 내부 스미어는 못 품. `s_f`는 DXF 단일 소스 | **유지.** 라이브러리로 교체하지 않음 |
| **TEASER++** | MIT | B의 전역 Sim3 | 이상치 비율이 높은 대응에서 스케일 1개 + R + t 를 인증 가능한 방식으로 추정. Umeyama/FGR보다 오염에 강함 | **스케일은 등방 1개.** 이방 스미어를 풀지 못함. 대응점(FPFH 등)이 쓰레기면 결과도 쓰레기. 반복 복도에서 그럴듯한 오답이 나옴 | **형상 보정 이후, 후보 생성기.** 자동 확정 금지 |
| **Open3D** FPFH+FGR/RANSAC→ICP | MIT. 렌더 extra에 이미 있음 | B의 조임 | 파이썬 스택에 붙이기 쉬움. 천장·벽 쉘의 국소 ICP에 적합. RANSAC은 `with_scaling=True`일 때만 스케일을 봄 | **FGR 기본 변환은 회전·이동.** 튜토리얼 기본 경로는 스케일을 안 푼다. 부분 겹침·밀도 차·천장 편향 점군(이 업로드는 점의 대부분이 카메라 위)에서 FPFH 전역 해는 약함 | **ICP 정밀화 1순위.** FPFH 전역은 후보만 |
| **PCL** ICP/GICP/NDT | BSD | B의 조임 | GICP는 노이즈 표면, NDT는 밀도 차가 큰 실내 스캔에 강함. Open3D에 없는 실무 선택지 | 강체 국소 정합. 초기값이 있어야 함. C++ 의존이 제품 이미지에 큼. Open3D ICP와 역할이 겹침 | **보류.** Open3D ICP가 쉘에서 실패할 때만 NDT를 재검토 |
| **gsplat** + **nerfstudio** | Apache-2.0 | **C** | 정합된 포즈 위에서 포토리얼 워크스루. 데모 점군 렌더의 대안. 밀도가 올라가면 coverage 광선의 보조 신호는 될 수 있음 | 주어진 포즈에 색·밀도를 맞춘다. 틀린 포즈는 매끈한 오차가 됨. means에 강체 `T`를 곱해도 스미어는 남음. 부재 ID·coverage enum을 만들지 않음. nerfstudio는 의존 트리가 큼 | **선택 렌더.** 매핑 경로에 넣지 않음. 제품에는 gsplat만, nerfstudio는 학습 툴체인 |
| **Point-Cloud-Utils** | MIT | 전처리 | BIM 메시 → 점 샘플, 거리 쿼리 | Open3D 샘플링과 겹침. 정합 능력 없음 | 샘플링 함수가 부족할 때만 |
| **PyTorch3D** | BSD-3 (Meta) | 나중 단계의 2D–3D | BIM 엣지·실루엣을 렌더해 `solvePnP` 대응을 만들 때 (scout 경로 B2) | 점군 정합기가 아님. 지금 오차의 1차 원인이 아님 | 보류 |
| **Kaolin** | Apache-2.0 (NVIDIA) | 메시 보조 | 메시·텐서 연산 | PyTorch3D와 역할 중복. 런타임이 무거움 | 보류 |

## 3. 증상별로 누가 고치는가

“유리하다”를 기능 목록이 아니라 이 저장소에서 이미 측정된 증상으로 보면 다음과 같다.

| 증상 | COLMAP | TEASER++ | Open3D ICP | 현행 plan-match | gsplat |
|---|---|---|---|---|---|
| 측방·진행 스케일이 다름 (`s_h`≠`s_f`) | 재투영으로 형상을 한 스케일에 가깝게 되돌릴 후보 | 등방 스케일 1개만 추정 | 국소 강체라 내부 형상은 유지 | 이방 스케일로 우회 중 | 포즈를 그대로 학습 |
| 급회전 스미어 | BA가 줄일 수 있는 유일한 후보 | 강체라 잔존 | 잔존 | 잔존 (desmear 전제는 이 업로드에 불성립) | 잔존. 화면만 매끈 |
| 절대 미터 | up-to-scale. `model_aligner`는 제어점 필요 | 대응이 성립하면 스케일 1개 | 기본 경로는 스케일 없음 | DXF 실측이 정본 | 없음 |
| 현장이 BIM과 다름 | 모름 | 이상치 비율이 높아도 Sim3는 시도 | 초기값이 맞으면 부분 겹침 refine | **이 층의 정본.** T≈25% 초과는 HOLD | 모름 |
| 반복 복도·MEP 오확정 | 모름 | 그럴듯한 오답이 나옴 | 같은 위험 | 점수차 마진 미달이면 HOLD, `best=None` | 모름 |
| 부재 coverage | 포즈가 좋아지면 간접 개선 | 배치가 좋아지면 간접 | 간접 | 배치를 제공 | **대체 불가.** 광선 대상은 BIM 메시 |
| 데모 설득력 | 희소 점 | — | 색점군 | 현재 점군 데모 | 포토리얼. 포즈가 맞을 때만 |

## 4. 점군·가우시안을 얹으면 보완되는가

### 4.1 점군 정합 (Open3D + TEASER++ + ICP) — 보완된다. 조건이 있다

보완되는 것:

- plan-match가 **이미 고른 후보**를 벽·천장 쉘에 밀착시키는 마지막 수 cm~수십 cm.
- 대응 이상치가 많을 때, 단일 스케일 Sim3를 TEASER++가 Umeyama보다 덜 무너지게 만드는 것.
- coverage 광선이 이웃 부재로 새는 양을, 배치가 조여진 만큼 줄이는 것.

보완되지 않는 것:

- 궤적 내부의 이방성·스미어. ICP와 TEASER의 출력은 한 개의 변환이다.
- “어느 복도인지”와 “확정해도 되는지”. FPFH는 반복 기하에서 여러 해를 같은 점수로 낸다. 자동으로 `T`를 확정하면 불변식 ②를 깬다.
- BIM 메시를 균일 샘플해 영상 점군과 FPFH로 전역 정합하는 경로는, 이 데이터에서 약한 시작점이다. 영상 점군은 카메라가 본 표면(천장 편향)이고, BIM 샘플은 안 본 실·가구·현장 변경분까지 포함한다. 겹침이 부분적이고 스케일이 방향마다 다르다.

제품에 넣는 형태: `--refine open3d` / `--refine teaser` **opt-in**. 기본 `--plan-match off` 출력은 지금과 byte-identical. 점수차가 마진 미만이면 후보 JSON만 저장한다.

PCL은 같은 층의 두 번째 구현이다. Open3D ICP로 쉘 정밀화가 부족할 때 NDT만 다시 보면 된다.

### 4.2 가우시안 (gsplat / nerfstudio) — 증거 층만 보완된다

보완되는 것:

- 포즈가 BIM 프레임에 앉은 **뒤**의 데모 화질. 지금 표준 출력(팔로우 점군 + 버드아이)의 점군 렌더를 포토리얼로 바꿀 수 있다.
- 표면이 비는 구간에서, 렌더 밀도·보조 occupancy.

보완되지 않는 것:

- 매핑. 학습은 입력 포즈를 참으로 둔다. 포즈의 `s_h`/`s_f` 오차는 가우시안 means의 배치 오차가 된다.
- GS PLY에 강체 `T`만 곱하는 보정. `T`가 맞어도 스미어는 각 means의 상대 위치에 남아 있다.
- BIM 프레임에서 gsplat을 다시 학습하는 것. 카메라가 그 프레임에서 이미 맞아야 학습이 성립한다. 스케일 모순을 학습이 풀어 주지 않는다.
- 부재별 coverage. 상태 enum의 정본은 BIM 메시와의 관계다. 가우시안은 부재가 아니다.
- 메시에 가우시안을 묶는 GaMeS급 표현. 그 역할의 공개 구현은 이번 상업용 목록에 없다 (§6).

제품에 넣는 형태: 2단을 통과한 포즈로만 학습하는 **render-demo extra**. 실패해도 coverage API는 그대로 동작. 의존을 작게 유지하려면 런타임은 gsplat, 학습 스크립트만 nerfstudio.

### 4.3 그래서 어떤 조합이 더 유리한가

카메라 영상이 기준인 이상, 이득 순서는 아래가 맞다.

1. **COLMAP** — 영상 안의 기하를 먼저 살린다. 이 프로젝트에서 가장 큰 빈칸.
2. **현행 plan-match + DXF** — BIM의 어느 곳인지, 미터인지, 확정인지 HOLD인지를 결정한다. 목록의 어떤 라이브러리도 이 정책을 포함하지 않는다.
3. **Open3D ICP** — 통과한 후보를 쉘에 밀착. 이미 MIT이고 파이썬에 가깝다.
4. **TEASER++** — 이상치가 큰 Sim3 후보. 등방 스케일 전제(1번 이후)에서만.
5. **gsplat** — 그 포즈로 데모. 매핑 지표에는 넣지 않는다.
6. **PCL, PCU, Kaolin, PyTorch3D** — 1~5로 막히는 측정이 나오기 전에는 추가하지 않는다.

초안 파이프를 이 저장소에 맞게 고치면 다음과 같다.

```text
폰 RGB
  → 형상: LingBot-Map 포즈를 초기값으로 두거나, COLMAP SfM+BA로 재투영 정합
          (절대 스케일은 아직 없음. VGGT·DUSt3R·MASt3R 경로는 사용하지 않음)
  → 배치: 기존 plan-match / place_rigid + DXF
          애매하면 HOLD, 후보는 명시 저장
  → 조임: 채택 후보에 한해 구조 쉘만 Open3D ICP
          이상치가 크면 TEASER++로 Sim3 후보를 하나 더 만들고, 같은 HOLD 게이트를 태움
  → 증거: 기존 coverage 광선 (BIM 메시)
  → 선택: 그 포즈로 gsplat 학습 → 데모 영상
```

Open3D FPFH 전역 해를 1등으로 두고 plan-match를 빼면, 반복 복도에서 자동 오확정이 나오기 쉽다. FPFH는 후보 생성까지만.

## 5. 라이선스 재확인 (2026-10-07)

원문 URL은 저장소 기본 브랜치의 `LICENSE` / `LICENSE.txt` / `LICENSE.md`. 조회가 실패한 항목은 본문에 적었다.

| 저장소 | 확인한 파일 | 결과 |
|---|---|---|
| isl-org/Open3D | `LICENSE` | MIT. Copyright 2018–2026 www.open3d.org |
| MIT-SPARK/TEASER-plusplus | `LICENSE` | MIT. Copyright 2020 MIT |
| PointCloudLibrary/pcl | `LICENSE.txt` | BSD (3-Clause 형식). Willow Garage / Open Perception |
| colmap/colmap | `LICENSE` | new BSD. **본문 주의:** 의존성은 따로 라이선스되며, 그 의존으로 빌드한 결과물의 라이선스가 달라질 수 있음. 제품 빌드의 의존 SBOM이 필요 |
| nerfstudio-project/gsplat | `LICENSE` | Apache-2.0 |
| nerfstudio-project/nerfstudio | `LICENSE` | Apache-2.0. 플러그인·전이 의존은 패키지 SPDX와 별도 |
| fwilliams/point-cloud-utils | `LICENSE` | MIT. Copyright 2023 Francis Williams |
| NVIDIAGameWorks/kaolin | `LICENSE` | Apache-2.0 |
| facebookresearch/pytorch3d | `LICENSE` | BSD. Copyright Meta Platforms |
| graphdeco-inria/gaussian-splatting | `LICENSE.md` | 연구용. 비상업. 상업 이용은 사전 동의 없으면 금지 (4.3, 상업 목적 사용·배포 금지 조항) |
| Anttwo/SuGaR | `LICENSE.md` | 위 Gaussian-Splatting License와 동일 본문 |
| naver/dust3r | `LICENSE` | CC BY-NC-SA 4.0 |
| naver/mast3r | `LICENSE` | CC BY-NC-SA 4.0 |
| NVlabs/InstantSplat | `LICENSE` | Apache-2.0 (**저장소 자체**). README 기본 경로는 MASt3R 체크포인트 다운로드 + DUSt3R 의존. 코어 SPDX만으로 상업 경로가 되지 않음 |
| facebookresearch/vggt | `LICENSE.txt` | Meta Research Materials 계약 (2025-07-29). OSI 라이선스가 아님. 비독점·비양도, 재배포는 동일 계약, AUP 준수. 상업 배포 전 별도 확인 |
| waczjoan/gaussian-mesh-splatting | `main/LICENSE` | **404.** README가 구현 기반으로 Inria 3DGS를 지정. 3DGS 라이선스에 묶인 것으로 보고 제품에서 제외 |
| CloudCompare/CloudCompare | `LICENSE`, `COPYING`, `COPYING.txt` | **이번 조회 404.** 사용자 제공 정보대로 GPL로 두고, 제품 링크·재배포 전에 저장소 라이선스 파일을 다시 연다. 내부 검수 도구와 제품 의존은 구분 |

배포 게이트는 기존 PRD와 같다. 코드 SPDX, 가중치 SPDX, CUDA EULA를 한 줄로 합치지 않는다. COLMAP을 넣더라도 Ceres·Boost·Qt·CUDA 조합의 고지를 `NOTICE`에 남긴다.

## 6. 상업 경로에서 제외

| 이름 | 이유 | 대신 |
|---|---|---|
| Inria 3DGS, SuGaR | 연구·비상업 라이선스 원문 | gsplat (Apache-2.0) |
| GaMeS | 별도 OSI LICENSE 미확인 + 3DGS 기반 | 정렬된 BIM에 GS를 묶지 않음. 포즈를 BIM에 맞춘 뒤 gsplat 재학습, 또는 means에 Sim3 적용 |
| DUSt3R, MASt3R | CC BY-NC-SA 4.0 | COLMAP 또는 가중치 SPDX가 확인된 LingBot-Map |
| InstantSplat 기본 경로 | 코어는 Apache-2.0이나 MASt3R/DUSt3R을 기본으로 받음 | 그 의존을 뺀 구성이 생기기 전에는 제외 |
| VGGT | Research Materials 계약. 제품 이미지에 런타임·가중치 금지 (PRD FR-1.4) | COLMAP 폴백 |
| CloudCompare | GPL로 알려진 상태. 원문 재확인 실패 | 내부 QA 전용. 제품에 링크하지 않음 |

## 7. 도입 순서

측정이 라이선스 문구보다 먼저다.

1. 같은 업로드에서 COLMAP 궤적과 현재 피드포워드 궤적의 축별 스케일 비를 비교한다. `s_h`와 `s_f`가 한 값으로 모이면 Job A가 줄어든 것이다.
2. 그 포즈로 기존 plan-match를 돌린다. HOLD 조건과 `--plan-match off` byte-identical은 유지한다.
3. HOLD가 아닌 후보에만 Open3D point-to-plane ICP를 쉘(벽·천장)에 적용하고, 부재 coverage 이동량을 본다.
4. ICP 초기화가 이상치로 무너지는 케이스에만 TEASER++를 후보 생성으로 추가한다.
5. 2~3이 맞은 포즈로 gsplat 데모를 한 번 렌더해, 점군 데모 대비 검수 가독성만 비교한다. coverage 숫자의 정본은 바꾸지 않는다.
