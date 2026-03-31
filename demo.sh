#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-predict}" # train | predict | webcam
DATA="${DATA:-datasets/helmet_data.yaml}"
MODEL="${MODEL:-runs/helmet_50ep/weights/best.pt}"
SOURCE="${SOURCE:-results/sample_predictions}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
CONF="${CONF:-0.25}"
RUN_NAME="${RUN_NAME:-demo_run}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON="python"
fi

echo "Mode: ${MODE}"
echo "Python: ${PYTHON}"

case "${MODE}" in
  train)
    "${PYTHON}" train.py \
      --data "${DATA}" \
      --epochs "${EPOCHS}" \
      --imgsz "${IMGSZ}" \
      --batch "${BATCH}" \
      --name "${RUN_NAME}"
    ;;
  predict)
    "${PYTHON}" predict.py \
      --model "${MODEL}" \
      --source "${SOURCE}" \
      --imgsz "${IMGSZ}" \
      --conf "${CONF}" \
      --name "${RUN_NAME}"
    ;;
  webcam)
    "${PYTHON}" predict.py \
      --model "${MODEL}" \
      --source 0 \
      --imgsz "${IMGSZ}" \
      --conf "${CONF}" \
      --name "${RUN_NAME}" \
      --show
    ;;
  *)
    echo "Invalid mode: ${MODE}. Use one of: train, predict, webcam"
    exit 1
    ;;
esac

