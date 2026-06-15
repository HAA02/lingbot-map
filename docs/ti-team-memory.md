# TI팀 메모리 — 영상→점군→데모식 출력 작업 요약

작성일: 2026-06-15 · 브랜치: `AutoPM-260612-upstream-windowed`

이 세션에서 한 작업과 팀이 알아야 할 컨벤션·환경을 정리한다.

## 1. 멀티윈도우 병합 수정 (회귀 해결)
- 문제: 긴 영상(>300프레임)에서 realtime 워커가 윈도우별 `inference_streaming` + 자작 Procrustes/ICP(`registration.chain_windows`)로 봉합 → 좌표 드리프트/이중벽.
- 수정: `realtime/inference_worker.py`를 **상류 `gct_stream_window.GCTStream.inference_windowed`** 1회 호출로 교체(overlap=다음 윈도우 scale frame 재사용 + 학습된 chunk_transforms 내부 정렬). 자작 봉합 제거.
- 검증: 카메라 궤적이 끊긴 점프 → 연속 호로 복원, 표면 점밀도 향상. (`tools/windowed_recon_test.py`)

## 2. 영상 출력 표준 포맷 (★ 팀 컨벤션)
앞으로 영상 결과물은 **항상 데모 스타일**로 출력한다:
- **메인 화면** = 촬영자 시점(follow chase-cam) 점군 + 카메라 경로(trail)
- **우상단** = 전체 맵(birdeye 미니맵) + 원본 촬영영상
- **코덱** = H.264 (yuv420p, +faststart) — 우분투 기본 플레이어 호환 (OpenCV 기본 mp4v는 Totem에서 안 열림)

**생성:** `tools/render_demo_format.sh <영상> <출력폴더> [keyframe_interval]`
→ `<출력폴더>/<stem>_demo_format.mp4`. (>320프레임이면 keyframe_interval 4 권장)
- config: `demo_render/config/follow_main.yaml`(메인), `birdeye_mini.yaml`(미니맵)
- 동작: 추론 1회(npz 저장) → follow 렌더 + birdeye 렌더(npz 재사용) → ffmpeg 합성 → H.264

## 3. 데모 오프라인 렌더 환경 (`.venv-lbdemo`, gitignore — 각자 재구축)
RTX 5090(Blackwell, **sm_120**)은 CUDA 12.8+ 필요. kaolin 프리빌트는 torch 2.8/cu128까지 지원.
메인 `.venv`(torch 2.10)는 건드리지 않고 별도 격리 env 사용.

재구축 절차(micromamba):
```bash
P=$PWD/.venv-lbdemo
micromamba create -y -p "$P" -c nvidia -c conda-forge python=3.11 pip \
  "cuda-version=12.8" "cuda-toolkit=12.8.*" gxx_linux-64=13 gcc_linux-64=13 sysroot_linux-64
micromamba run -p "$P" pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
micromamba run -p "$P" pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
micromamba run -p "$P" pip install numpy==1.26.4 open3d==0.19.0 opencv-python einops scipy Pillow \
  pyyaml tqdm imageio imageio-ffmpeg viser huggingface_hub safetensors requests matplotlib
# render_cuda_ext 빌드 (sm_120) — 시스템 glibc가 너무 최신이라 conda gcc-13 + sysroot 사용:
cd demo_render/render_cuda_ext
CUDA_HOME="$P" CPATH="$P/targets/x86_64-linux/include" \
CC="$P/bin/x86_64-conda-linux-gnu-gcc" CXX="$P/bin/x86_64-conda-linux-gnu-g++" \
CUDAHOSTCXX="$P/bin/x86_64-conda-linux-gnu-g++" CONDA_BUILD_SYSROOT="$P/x86_64-conda-linux-gnu/sysroot" \
TORCH_CUDA_ARCH_LIST="12.0" "$P/bin/python" setup.py build_ext --inplace
```
- `setup.py`에 `-D_GNU_SOURCE` 추가됨(커밋 포함) — nvcc가 최신 glibc 헤더 함수를 못 찾던 문제 해결.
- 실행 시: `PYTHONPATH=<repo>:<repo>/demo_render/render_cuda_ext`, `LD_LIBRARY_PATH=$P/targets/x86_64-linux/lib`, `--use_sdpa`(flashinfer 없음).

## 4. 정합/Coverage (pipe_duct, 참고)
- `realtime/server.py` 가동 → `/api/upload-video` → auto-place(`alignment/candidates auto_place`) → `coverage/analyze`.
- pipe_duct는 반복객체(repetition_risk=high) → auto-place는 *yellow 후보*(확정 아님). 정확 정합은 대응점 필요(설계 불변식).
- 보조 스크립트: `tools/align_video_to_model.py`, `tools/capture_coverage_html.py`.

## 5. AutoPM vs main 비교
- `reports/autopm_vs_main_comparison.md` 참조. 공유 코어(`lingbot_map/`·`demo.py`) 동일, AutoPM=realtime/coverage 제품, main(upstream)=benchmark 평가 스위트. 리베이스 충돌은 `.gitignore`·`README.md` 둘뿐.

## 6. 정리/주의
- 결과 품질은 **촬영**에 좌우(느리게·궤도로·겹치게·밝게·텍스처). 파이프라인은 상류 demo와 동일함을 공식 데모 데이터로 검증함.
- 모델은 입력을 518px폭으로 다운샘플 → **일반 모바일(1080p)로 충분**, 장비보다 촬영법이 관건.
- 미사용 라이브러리 정리: `.venv-lbdemo`에서 `onnxruntime-gpu`·`trimesh` 제거(try-wrap, 미사용), pip/micromamba 캐시 정리(~50GB 회수).
- gitignore: `.venv-lbdemo/`, `out_*/`, `video/_demo_ref/`(데모 ref 프레임).
