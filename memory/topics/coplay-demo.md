# coplay-demo

> 범위: coplay PiP에 어떤 영상을 넣는가 · 갱신 2026-10-07 · 근거 handoff: handoff/2026-10-07-commercial-recon.md

## 정본

- PiP는 원본 휴대폰 mp4가 아니라 `tools/render_demo_format.sh`의 H.264 점군 데모다. 원본은 HEVC/4K라 브라우저가 못 연다.
- 업로드 처리 뒤 데모를 만들고, 그 파일을 `--render-video`로 coplay에 넣는다. 데모가 없으면 raw로 떨어진다.
- `/coplay`에 upload가 없으면 가장 최근 coplay로 간다. 포즈와 PiP는 같은 업로드여야 한다.
- 렌더는 `.venv-lbdemo`에서 돈다. 메인 `.venv`와 섞지 않는다.

## 할 일

- `upload_1779442357085` 데모 렌더가 없으면 PiP가 raw로 남는다.

## 함정

### 데모 합성 단계가 끊기면 입력 세 편만 남는다 (2026-06-29)

pointcloud·overview·rgb가 있으면 ffmpeg 합성만 다시 돌린다. 재구성부터 다시 하지 않는다.

## 결정

### 2026-06-29 · 팀 표준 데모식만 PiP에 넣는다

팔로우 점군이 본화면에 오고, 우상단은 버드아이와 데모 영상이다.
