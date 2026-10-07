# coordinates

> 범위: dtdx·GLB·재구성을 한 프레임으로 어떻게 두는가 · 갱신 2026-10-07 · 근거 handoff: handoff/2026-10-07-commercial-recon.md

## 정본

- 모델 월드는 Z-up 미터. glTF Y-up 정점은 `(x, −z, y)`.
- `.dtdx`는 Babylon 왼손 좌표다. Three로 올릴 때 X를 뒤집지 않으면 좌우가 거울이다. `tools/build_coplay.py`는 디코드 직후 `pos[:, 0] *= -1`을 한다.
- PXX.glb는 내부 피트·Z-up이고 root 행렬이 변환한다. pipe_duct.glb는 내부 미터·Y-up이고 정점에 구워져 있다. accessor min/max Y는 믿지 않고 정점을 디코드한다.
- COLMAP 파서(`scan2bim/colmap_poses.py`)는 재구성 프레임 그대로다. BIM 축 변환을 하지 않는다. 카메라 +Z가 전방, +Y 아래의 반대가 위다.

## 할 일

- 없음.

## 함정

### DTDWebThree도 X를 뒤집지 않는다 (2026-06-16)

그 뷰어와 맞춘 화면은 원본 저작 뷰어와 좌우가 반대다. 정상 방향은 X 반전이다.

## 결정

### 2026-06-16 · coplay 모델은 X 반전 후 표시

2×2 방 블록이 왼쪽, 긴 소화배관이 오른쪽일 때 원본 뷰어와 같다.
