# Build vs Buy

## 기능: 영상 기반 시공실적 자동 추적 (목표 A 전체)

**이 목적은 성숙한 상용 카테고리다.** 직접 구현 전 반드시 검토해야 한다.

| 후보 | 방식 | BIM 필요? | 적합도 | 비고 |
|---|---|---|---|---|
| [Doxel](https://doxel.ai/) | 360° 영상 ↔ BIM 대조로 work-in-place 자동 정량화, 일정 연동 | 필요 | **높음** — 목표 A와 사실상 동일 기능 | 헬멧 360캠 착용 후 평소대로 걷기 |
| [Buildots](https://www.planradar.com/us/360-construction-photo-documentation-software/) | 360° 헬멧캠 → BIM 대조, 트레이드·구역별 완료율, 80+ 시공단계 | 필요 | **높음** | 일일 자동 캡처·분석 |
| [OpenSpace](https://www.openspace.ai/products/progress-tracking/) | 360° 캡처 → 도면/BIM 자동 매핑 | **불필요**(도면만으로도 동작) | **높음** | BIM 없이도 배포 가능 |

### 🔑 결정적 관찰: 상용 3사 전부 **360° 카메라**를 쓴다

우리가 이번 세션 내내 싸운 문제(단안 스케일 모호성·전역 정합 실패·반복 패턴)는 **일반 휴대폰 단안 영상**을 쓴다는 취득 선택에서 파생됐다. 360° 캡처는 시야가 전방위라 프레임마다 벽·바닥·천장·복도 양끝이 동시에 들어와 정합 구속이 압도적으로 많다. 즉 **상용 제품들은 이 문제를 알고리즘이 아니라 하드웨어로 회피**한다.

- 우리 현실: "천장 배관만 올려다보는 단안 영상" → focus-of-expansion 저시차, 벽 신호 부족 → 2회 CEILING [사용자 세션 실측]
- 상용 현실: 360° + 평소 보행 → 정합이 애초에 어려운 문제가 아님

### 부분 차용 가능한 OSS

| 라이브러리 | 용도 | 라이선스 | 상태 |
|---|---|---|---|
| TEASER++ | 전역 point cloud 정합(FR-2.1) | MIT | PRD 부록에 이미 지목, 미도입 |
| Open3D | FGR/ICP | MIT | 미도입 |
| IfcOpenShell | IFC 수집 | LGPL | `.dtdx` 직접 디코드로 대체됨 |
| Metric3D v2 / UniDepth | metric depth(FR-1.3) | — | **보류** — Python 3.14 휠 부재 [문서:docs/scan2bim-progress.md:53] |
| hloc / NetVLAD | place recognition(coarse prior) | — | PRD 옵션, 미도입 |
| ARKit / ARCore | **VIO metric pose** — 단안 스케일 모호성을 소스에서 제거 | 무료(플랫폼) | 미검토 — **최우선 후보** |

### 학술적 근거 (사용자 질문 "학습이 필요한가"에 대한 답)

[MDPI: 3D Laser Scanning and BIM-Based Workflow for MEP Pipe Installation Discrepancies](https://www.mdpi.com/2075-5309/16/12/2444) — 스캔 배관 점군을 **BIM 유도 점군**과 coarse-to-fine으로 정합하고 개별 배관 인스턴스를 추출해 대응을 세우는 접근. 즉 **학습 기반 외관 인식이 아니라 기하 대응**이 이 문제의 주류 해법이다. [Markerless BIM Registration for Mobile AR](https://www.researchgate.net/publication/305303353_Markerless_BIM_Registration_for_Mobile_Augmented_Reality_Based_Inspection) 계열도 마커 없이 자동 정렬을 다루되 SLAM/기하 기반.

→ **"현장 학습 데이터 부족"은 원인이 아니다.** 이 프로젝트의 자체 실험도 같은 결론: `scan2bim/localize.py` 객체앵커 PnP는 합성 GT에서 0.003m로 정확했고, 실패 원인은 학습이 아니라 **모델 앵커 밀도/대응 제약**이었다 [문서:docs/scan2bim-progress.md:FR-2.3 stage-2].

---

## 판정

| 대상 | 판정 |
|---|---|
| 목표 A 전체(실적 자동추적 플랫폼) | 🟢 **도입 검토 강력 권장** (Doxel/Buildots/OpenSpace) — 직접 구현 시 상용 제품과 동일 문제를 처음부터 재발명. 단 사내 데이터 정책·비용·`.dtdx` 파이프라인 통합 요구가 있으면 재판단 |
| 취득 방식 | 🟢 **360° 캠 또는 ARKit/ARCore VIO 도입** — 알고리즘으로 못 푼 문제를 취득으로 해소 (상용 3사 전례) |
| 정합 코어 | 🟡 **부분 차용**(TEASER++/Open3D) — 직접 구현 중인 그리드탐색/ICP보다 검증된 전역정합 |
| 반복 해소 시그니처(직경+토폴로지) | 🔴 **직접 구현** — `.dtdx` connector 고유 자산, 대체 OSS 없음 |
| 텍스트·사인 앵커 | ❌ **폐기** — 모델에 대응물 부재(실측 확인), PRD 비목표와도 충돌 |

**근거:** 이 분야는 성숙한 상용 카테고리이며, 세 제품 모두 우리가 막힌 지점(단안 정합)을 360° 취득으로 우회한다. 자체 구현을 계속한다면 최소한 **취득 방식만은 그들과 같은 급**으로 올려야 한다.

Sources:
- [OpenSpace — Construction Progress Tracking](https://www.openspace.ai/products/progress-tracking/)
- [Doxel — Product](https://doxel.ai/product)
- [PlanRadar vs OpenSpace vs Buildots 비교](https://www.planradar.com/us/360-construction-photo-documentation-software/)
- [MDPI Buildings — Scan-BIM MEP pipe discrepancies](https://www.mdpi.com/2075-5309/16/12/2444)
- [Markerless BIM Registration for Mobile AR Inspection](https://www.researchgate.net/publication/305303353_Markerless_BIM_Registration_for_Mobile_Augmented_Reality_Based_Inspection)
