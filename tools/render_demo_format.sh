#!/usr/bin/env bash
# Reference-style output format:
#   MAIN (full frame) = follow / filmer-POV point-cloud view
#   TOP-RIGHT insets  = full-map overview (birdeye) + original footage
#
# Uses the project-local env .venv-lbdemo (torch 2.8 cu128 + kaolin + render_cuda_ext).
# Inference runs once (predictions saved); birdeye overview re-uses them.
#
# Usage: tools/render_demo_format.sh <video> <out_dir> [keyframe_interval]
set -euo pipefail
REPO=/run/media/iaan/1TB-WD/Github/lingbot-map
PREFIX=$REPO/.venv-lbdemo
VIDEO="${1:?usage: render_demo_format.sh <video> <out_dir> [keyframe_interval]}"
OUT="${2:?out_dir required}"
KFI="${3:-4}"
STEM=$(basename "${VIDEO%.*}")
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export PYTHONPATH="$REPO:$REPO/demo_render/render_cuda_ext"
export LD_LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib"
RUN(){ micromamba run -p "$PREFIX" python "$REPO/demo_render/batch_demo.py" "$@"; }
FF=$(micromamba run -p "$PREFIX" python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
probe_dim(){ "$FF" -hide_banner -i "$1" 2>&1 | sed -n 's/.*Video:.*, \([0-9]\{2,\}x[0-9]\{2,\}\).*/\1/p' | head -1; }

mkdir -p "$OUT"
echo "[1/3] MAIN follow render (+save predictions) ..."
RUN --video_path "$VIDEO" --output_folder "$OUT" \
    --config "$REPO/demo_render/config/follow_main.yaml" \
    --model_path "$REPO/ckpts/lingbot-map-long.pt" \
    --use_sdpa --keyframe_interval "$KFI" --save_predictions

echo "[2/3] OVERVIEW birdeye render (re-use predictions) ..."
RUN --load_predictions "$OUT/$STEM" --output_folder "$OUT/overview" \
    --config "$REPO/demo_render/config/birdeye_mini.yaml" --use_sdpa

MAIN=$(ls "$OUT/${STEM}"*pointcloud.mp4 2>/dev/null | grep -viE "rgb|combined" | head -1)
RGB=$(ls "$OUT/${STEM}"*rgb*.mp4 2>/dev/null | head -1)
OVER=$(ls "$OUT/overview/"*.mp4 2>/dev/null | grep -viE "rgb|combined" | head -1)
[ -f "$MAIN" ] && [ -f "$RGB" ] && [ -f "$OVER" ] || { echo "missing inputs: MAIN=$MAIN RGB=$RGB OVER=$OVER"; exit 1; }

echo "[3/3] compositing (main + overview + footage) ..."
IW=460; M=24; GAP=14; BW=3
ov=$(probe_dim "$OVER"); ovw=${ov%x*}; ovh=${ov#*x}
ovsh=$(( IW * ovh / ovw )); ovsh=$(( ovsh - ovsh%2 ))
rgy=$(( M + ovsh + 2*BW + GAP ))
"$FF" -y -hide_banner -loglevel error -i "$MAIN" -i "$OVER" -i "$RGB" -filter_complex "\
[1:v]scale=${IW}:-2,pad=iw+2*${BW}:ih+2*${BW}:${BW}:${BW}:white[ov];\
[2:v]scale=${IW}:-2,pad=iw+2*${BW}:ih+2*${BW}:${BW}:${BW}:white[rg];\
[0:v][ov]overlay=W-w-${M}:${M}[a];\
[a][rg]overlay=W-w-${M}:${rgy}[out]" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 -preset medium -movflags +faststart \
  "$OUT/${STEM}_demo_format.mp4"
echo "DONE -> $OUT/${STEM}_demo_format.mp4"
