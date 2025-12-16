#!/bin/bash
set -e
set -o pipefail  # CRITICAL: Ensures script fails if Python fails, even with 'tee'

# 1. Define Variables (Fixed missing DONE_FILE)
EXPERIMENTS_DIR="/app/experiments"
LOG_FILE="$EXPERIMENTS_DIR/training_console.log"
DONE_FILE="$EXPERIMENTS_DIR/.done"

echo "--- Initializing Container ---"

# 2. Idempotency Check
if [ -f "$DONE_FILE" ]; then
    echo "--- Found .done file. Experiments completed. Spinning... ---"
    sleep infinity
fi

# 3. Background TensorBoard
echo "--- Starting TensorBoard... ---"
mkdir -p "$EXPERIMENTS_DIR"
nohup tensorboard \
    --logdir "$EXPERIMENTS_DIR" \
    --port 6006 \
    --bind_all \
    --reload_interval 30 \
    > /tmp/tensorboard.log 2>&1 &

# 4. Run Training (Fixed: Added xvfb-run)
echo "--- Starting Training Sequence ---"
python -u train.py 2>&1 | tee -a "$LOG_FILE"

# 5. Mark Done & Shutdown
echo "--- Training Complete. Syncing logs... ---"
touch "$DONE_FILE"
sleep 5

echo "--- Shutting down Pod to stop billing ---"
runpodctl stop pod "$RUNPOD_POD_ID"