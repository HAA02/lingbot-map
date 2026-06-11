# 모델 기준 영상 Evidence 작업 요약

작성일: 2026-06-11

## 목표

`lingbot-map`에서 영상을 업로드한 뒤, `/models` 기준 모델 위에 촬영된 위치, 이동 경로, 시점별 썸네일, 객체별 촬영 evidence를 웹뷰에서 확인한다.

핵심 판단 기준은 재구성된 포인트클라우드/메시의 외형 품질이 아니라, 기준 모델 좌표계에서 “어느 부분이 촬영됐는지”를 검토 가능한 근거와 함께 표시하는 것이다.

## 현재까지 작업된 내용

- `/coverage.html?model=pxx&upload={upload_id}` 모델 기준 evidence 뷰어 추가.
- `/coverage-report.html?model=pxx&upload={upload_id}` 90% 검수 리포트 페이지 추가.
- `/upload`에서 업로드 완료 후 모델 evidence 뷰와 검수 리포트 링크 제공.
- 모바일 촬영 업로드와 기존 영상 파일 선택을 분리.
- `/api/models`에서 `models/*.glb`와 PAG JSON 기반 manifest 생성/조회.
- GLB GUID 매핑률이 낮은 경우 PAG proxy geometry를 coverage 기준 형상으로 사용.
- PXX 기준 현재 상태:
  - 객체 수: 507개
  - GUID 매핑률: 0%
  - coverage 기준: PAG proxy
  - GLB 역할: 시각 참고용
- 업로드별 scan/pose/thumb/keyframe 조회 API 추가.
- scan-to-model 수동 대응점 정합 API 추가.
  - 최소 4개 대응점 필요.
  - RMSE, leave-one-out RMSE, residual, spread 검증.
  - `green/yellow` 정합만 coverage 분석에 사용.
  - `red` 정합은 observed 판정 금지.
- 정합 후보 API 추가.
  - 반복 객체가 많은 모델은 자동 매핑 위험을 표시.
  - 후보는 저장 전까지 preview로만 사용.
- coverage 분석 API 추가.
  - 거리만 보지 않고 visibility, temporal support, geometry consistency를 함께 사용.
  - `observed`, `likely_observed`, `uncertain`, `not_observed`로 구분.
- 객체별 evidence frame, keyframe thumbnail, 원본 keyframe 링크 표시.
- 객체 evidence와 frame 위치에 대한 수동 검수 저장 API 추가.
- 90% 품질 게이트 추가.
  - 자동 게이트와 수동 검수 게이트를 분리.
  - 최종 90% 통과는 수동 evidence/frame 검수 90% 이상이 필요.
- 정합 전 scan/path를 모델 위치에 임의로 맞춰 보이지 않도록 차단.
  - 정합 전 상태는 `raw_scan_local`로 표시.
  - viewer scene은 BIM 기준 `Z-up`, `XY` ground plane으로 통일.

## 현재 테스트 상태

### `upload_1779439687108` + `pxx`

- scan: LBP2, 380,918 points.
- pose/thumb: 144 / 144.
- scan GLB: 없음. RGB 점군으로 대응점 선택 가능.
- alignment: 없음.
- coordinate state: `raw_scan_local`.
- coverage: 없음.
- report: `blocked`.
- blockers:
  - coverage result missing.
  - green/yellow scan-to-model alignment required.
- 결론: 현재 화면에서 모델, 경로, 촬영 부위가 맞지 않는 것은 정상적인 제한 상태다. 아직 모델 좌표계 정합이 없으므로 촬영 부위를 정확히 판정할 수 없다.

### `upload_1779442357085` + `pxx`

- scan: LBM1, 282,070 points.
- pose/thumb: 80 / 80.
- scan GLB: 있음.
- alignment: `red`, invalid.
- coverage: stale red alignment 결과가 남아 있음.
- report: `blocked`.
- 결론: 기존 coverage는 신뢰하면 안 된다. 대응점을 다시 잡아 green/yellow 정합을 만든 뒤 coverage를 재생성해야 한다.

## 현재 문제의 원인

- 영상 재구성 좌표계와 BIM/PXX 모델 좌표계가 다르다.
- PXX GLB와 PAG 객체 GUID가 직접 연결되지 않아 현재는 PAG proxy 기준으로 coverage를 계산한다.
- PXX는 유사/반복 배관 객체가 많아 최근접 거리 기반 자동 매핑만으로는 오판 가능성이 높다.
- 현재 테스트 업로드에는 green/yellow 품질의 scan-to-model 정합이 없다.
- 따라서 화면에서 모델 축, 위치, 경로가 어긋나 보이고, 촬영된 부분도 정확히 표시되지 않는다.

## 작업해야 할 내용

1. 실제 대응점 정합 완료
   - `upload_1779439687108`에서 scan 점군과 PXX 모델의 같은 물리 위치를 4개 이상 선택.
   - 권장: 서로 멀리 떨어진 6개 이상 대응점.
   - 정합 품질이 green/yellow인지 확인.

2. coverage 재분석
   - 유효 정합 저장 후 coverage 분석 실행.
   - `observed/likely/uncertain/not` 상태가 모델 위에 제대로 표시되는지 확인.
   - report에서 blocker가 사라지는지 확인.

3. 수동 evidence 검수
   - observed 객체의 keyframe 위치가 실제 영상과 맞는지 검토.
   - 객체 evidence와 frame 위치를 accept/reject로 저장.
   - 최종 90% 통과 여부는 수동 검수 결과로 판단.

4. 모델 기준 형상 보강
   - GLB node와 PAG GUID 매핑을 높이는 export/map 파일을 확보.
   - GUID 매핑이 계속 0%라면 PAG proxy geometry를 더 정확한 배관 형상으로 개선.
   - 모델 원점/축/단위가 설계 데이터와 맞는지 manifest를 고정.

5. 자동 매핑 개선
   - 반복 배관 구간은 완전 자동 정합보다 사용자 anchor 또는 marker가 필요.
   - 향후 자동 후보는 geometry descriptor, 카메라 경로 방향, 구간 길이, keyframe 시각 특징을 함께 써야 한다.
   - 후보가 여러 개면 자동 확정하지 말고 검수 후보로만 표시.

6. 영상 처리 산출물 보강
   - LBP/LBM 산출물에 source frame id를 안정적으로 저장.
   - scan GLB가 없는 업로드도 point cloud overlay가 항상 표시되도록 유지.
   - keyframe 원본/crop 추출을 evidence 검수에 맞게 확대.

7. 품질 검증 자동화
   - `/api/models`, `/coverage/status`, `/coverage/report`, `/coverage.html`, `/coverage-report.html` smoke test 유지.
   - red/stale alignment 결과가 observed로 표시되지 않는지 회귀 테스트 추가.
   - 정합 없는 업로드가 coverage 분석을 통과하지 못하는지 테스트 추가.

## 검증 명령

서버가 실행 중일 때 다음 명령으로 모델 coverage 웹 흐름의 핵심 게이트를 확인한다.

```bash
python tools/coverage_web_smoke.py \
  --base-url http://127.0.0.1:8768 \
  --model pxx \
  --upload upload_1779439687108 \
  --stale-upload upload_1779442357085
```

이 smoke test는 다음 조건을 확인한다.

- coverage/report HTML 페이지가 열린다.
- PXX manifest가 유효하고, GUID 매핑률이 낮으면 PAG proxy 기준으로 처리된다.
- 정합이 없는 업로드는 `needs_alignment` 상태이며 90% 통과가 불가능하다.
- 정합 없이 coverage 분석을 요청하면 실패한다.
- red alignment 기반 stale coverage는 차단되고 리포트도 blocked 상태가 된다.

## 바로 확인할 URL

- 업로드: `http://127.0.0.1:8768/upload`
- evidence 뷰: `http://127.0.0.1:8768/coverage.html?model=pxx&upload=upload_1779439687108`
- 리포트: `http://127.0.0.1:8768/coverage-report.html?model=pxx&upload=upload_1779439687108`

## 다음 우선순위

1. `upload_1779439687108`에서 대응점 6개 이상 생성.
2. green/yellow 정합 저장.
3. coverage 분석 재실행.
4. 리포트에서 blocker 제거 여부 확인.
5. observed 객체 evidence를 수동 검수해 90% 통과 여부 판단.
