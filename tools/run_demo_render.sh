#!/usr/bin/env bash
# Run the demo_render OFFLINE rendering pipeline (kaolin + render_cuda_ext) using
# the project-local isolated env .venv-lbdemo (torch 2.8 cu128 + kaolin 0.18 +
# cuda 12.8, built for RTX 5090 / sm_120). The main .venv (torch 2.10) is untouched.
#
# Examples:
#   tools/run_demo_render.sh --video_path video/clip.mp4 --output_folder out_clip --keyframe_interval 2
#   tools/run_demo_render.sh --input_folder video/_demo_ref/travel --output_folder out_demo_render --keyframe_interval 2
set -euo pipefail
REPO="/run/media/iaan/1TB-WD/Github/lingbot-map"
PREFIX="$REPO/.venv-lbdemo"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export PYTHONPATH="$REPO:$REPO/demo_render/render_cuda_ext"
export LD_LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib"
exec micromamba run -p "$PREFIX" python "$REPO/demo_render/batch_demo.py" \
  --model_path "$REPO/ckpts/lingbot-map-long.pt" \
  --config "$REPO/demo_render/config/indoor.yaml" \
  --use_sdpa "$@"
