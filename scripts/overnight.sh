#!/bin/zsh
# Waits for detector training to finish, then caches fine-tuned detections for the
# validation game (game 4) and the test split and runs + evaluates the full pipeline on both.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f scripts/train_detector.py >/dev/null; do sleep 60; done
[ -f models/detection/fv_yolo26s_1280.pt ] || { echo "no fine-tuned weights"; exit 1; }
VAL=(data/SoccerNet/tracking/train/SNMOT-0{60..77})
TEST=(data/SoccerNet/tracking/test/SNMOT-*)
caffeinate -i $PY scripts/cache_detections.py --model models/detection/fv_yolo26s_1280.pt --tag ft1280 --seqs $VAL $TEST
caffeinate -i $PY scripts/run_suite.py --seqs $VAL --out outputs_val --workers 3
caffeinate -i $PY scripts/run_suite.py --seqs $TEST --out outputs_test --workers 3
echo OVERNIGHT_DONE
