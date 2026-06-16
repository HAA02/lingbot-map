# scan2bim 진척 요약

- 갱신: 2026-06-16
- 브랜치: `feat/scan2bim-auto-progress`
- PRD: `docs/scan2bim-auto-progress-prd.md`

## 구현 완료 (TDD 32/32 GREEN)

| FR | 모듈 | 내용 | 테스트 |
|---|---|---|---|
| 1.1 | `scan2bim/dtdx.py` | `.dtdx` 수집(material 색·mesh·linkMesh·connector·attr→GUID) | 8 |
| 1.1b | `scan2bim/dtdx_geometry.py` | geometry 디코드+인스턴스 배치 → 월드 삼각형(색상별) | 4 |
| 1.2 | `scan2bim/colormap.py` | 채색/무채색 분류 + 분야×색 usable 색맵 | 6 |
| 2.1 | `scan2bim/registration.py` | Sim3 정합(PCA coarse + Umeyama-ICP, Open3D 불필요) | 3 |
| 2.2 | `scan2bim/matching.py` | 색↔계통 시맨틱 매칭(텍스트 없이) | 4 |
| 3 | `scan2bim/progress.py` | 요소 coverage→상태, 일일 원장, day-over-day diff | 5 |
| 2.3 | `scan2bim/detect.py` | OWL-ViT open-vocab 검출 (실프레임: AC 0.79·조명 0.59·기둥 0.32) | — |
| 2.3 | `scan2bim/anchors.py` | 인스턴스별 객체 앵커(중심+footprint)+타입분류 (AC 54·조명 31·기둥 615) | 6 |
| — | `scan2bim/pipeline.py` | 통합: 정합→coverage→실적→색귀속 | 2 |
| 4 | `tools/build_coplay.py` | 영상↔dtdx WebGL co-play(중력정렬·천장고·follow-cam·매핑·**LH→RH X반전**) | 헤드리스 |

검증: `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests` (38 GREEN)

## 좌우반전(handedness) 수정 — 2026-06-16
`.dtdx`는 Babylon(왼손좌표) 저작 → Three(오른손좌표) 미변환 로드 시 좌우 거울반전. 정상뷰어(원본 프로그램, 53.57×48.73m)와 맞추려면 **모델 X 부호반전**. `build_coplay.py`는 디코드 직후 `pos[:,0]*=-1`, **DTDWebThree는 `SceneManager` models그룹 `scale.x=-1`**(전체 Gasan top-down 검증 완료). 상세 [[dtdx-handedness-flip]].

## co-play 품질 레시피 (demo급)
1. `build_demo_map.py --keep 500000` → dense photoreal cloud(원본 색, 500k)
2. `build_coplay.py --demo-html <그HTML> --demo-match 161613 --point-size 0.02`
3. WebGL: 솔리드 불투명 둥근 점 + `촬영자 시점` follow-cam

## 남은 작업
- **FR-1.3 메트릭 스케일(Metric3D)**: Python 3.14용 휠 부재로 보류. 대안 = 기준길이 1점 캘리브 또는 환경 분리. 현재는 scale≈1.0(거의 metric) 가정.
- **FR-2.3 토폴로지/직경 시그니처**: `connector`(609) 디코드해 반복 인스턴스 확정 — 색만으로 안 되는 회색(공조) 보완.
- **완전 photoreal**: WebGL point-splatting/EDL 셰이더(현재는 dense 크리스프 점).
- **정확 위치(XY·yaw)**: *매칭되는 현장 영상(Gasan 7F)* 필요 — 현재 영상은 다른 건물이라 정합 대상 없음. 엔진(FR-2.1)은 준비됨.

## 핵심 의존성
실데이터 end-to-end(물리적으로 정확한 co-play·자동 실적)는 **Gasan 7F를 촬영한 영상**이 들어와야 가능. 모든 엔진/모듈은 그 위에서 바로 동작하도록 TDD로 준비됨.
