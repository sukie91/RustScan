#!/bin/bash
# 监控 RustGS 训练进度

LOG_FILE="output/colmap_sofa/training_full_30k.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "=== RustGS Training Monitor ==="
echo "Log: $LOG_FILE"
echo ""

# 提取最新状态
LATEST=$(tail -5 "$LOG_FILE" | grep "Metal iter" | tail -1)

if [ -z "$LATEST" ]; then
    echo "Training not started or log format unexpected"
    echo ""
    echo "Last 10 lines of log:"
    tail -10 "$LOG_FILE"
    exit 0
fi

# 解析进度（使用awk而不是grep -P）
ITER=$(echo "$LATEST" | awk '{for(i=1;i<=NF;i++) if($i=="iter") print $(i+1)}' | tr -d '/')
TOTAL=30000
FRAME=$(echo "$LATEST" | awk '{for(i=1;i<=NF;i++) if($i=="frame") print $(i+1)}' | cut -d'/' -f1)
FRAMES_TOTAL=346
LOSS=$(echo "$LATEST" | awk '{for(i=1;i<=NF;i++) if($i=="loss") print $(i+1)}')
VISIBLE=$(echo "$LATEST" | awk '{for(i=1;i<=NF;i++) if($i=="visible") print $(i+1)}' | cut -d'/' -f1)
GAUSSIANS=$(echo "$LATEST" | awk '{for(i=1;i<=NF;i++) if($i=="visible") print $(i+1)}' | cut -d'/' -f2)
ELAPSED=$(echo "$LATEST" | awk -F'elapsed=' '{print $2}' | cut -d's' -f1)
STEP_TIME=$(echo "$LATEST" | awk -F'step_time=' '{print $2}' | cut -d's' -f1)

# 计算进度百分比
if [ "$TOTAL" -gt 0 ]; then
    PROGRESS=$(echo "scale=1; $ITER * 100 / $TOTAL" | bc)
else
    PROGRESS="0.0"
fi

# 估算剩余时间
if [ "$ITER" -gt 0 ]; then
    REMAINING=$(echo "scale=0; ($TOTAL - $ITER) * $STEP_TIME" | bc)
    REMAINING_MIN=$(echo "scale=1; $REMAINING / 60" | bc)
else
    REMAINING_MIN="?"
fi

echo "📊 Training Progress"
echo "  Iteration: $ITER / $TOTAL ($PROGRESS%)"
echo "  Frame: $FRAME / $FRAMES_TOTAL"
echo "  Loss: $LOSS"
echo "  Visible Gaussians: $VISIBLE / $GAUSSIANS"
echo ""
echo "⏱️  Timing"
echo "  Step time: ${STEP_TIME}s"
echo "  Elapsed: ${ELAPSED}s ($(echo "scale=1; $ELAPSED / 60" | bc) min)"
echo "  Remaining: ~${REMAINING_MIN} min"
echo ""

# 显示最新几行
echo "📝 Last 3 iterations:"
tail -20 "$LOG_FILE" | grep "Metal iter" | tail -3

echo ""
echo "💡 Use: tail -f $LOG_FILE"