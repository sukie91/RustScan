#!/usr/bin/env python3
"""监控 RustGS 训练进度"""
import re
import sys
from pathlib import Path

log_file = Path("output/colmap_sofa/training_full_30k.log")

if not log_file.exists():
    print(f"Log file not found: {log_file}")
    sys.exit(1)

# 读取最后几行
lines = log_file.read_text().strip().split('\n')[-10:]

# 找到最新的迭代信息
latest_iter = None
for line in reversed(lines):
    if 'Metal iter' in line:
        latest_iter = line
        break

if not latest_iter:
    print("No iteration info found")
    sys.exit(0)

# 解析
match = re.search(r'Metal iter\s+(\d+)/30000.*?frame\s+(\d+)/346.*?visible\s+(\d+)/(\d+).*?loss\s+([\d.]+).*?step_time=([\d.]+)s.*?elapsed=([\d.]+)s', latest_iter)

if not match:
    print(f"Failed to parse: {latest_iter}")
    sys.exit(0)

iter_num = int(match.group(1))
frame = int(match.group(2))
visible = int(match.group(3))
gaussians = int(match.group(4))
loss = float(match.group(5))
step_time = float(match.group(6))
elapsed = float(match.group(7))

# 计算进度
progress = iter_num / 30000 * 100
elapsed_min = elapsed / 60
remaining = (30000 - iter_num) * step_time / 60

print("=" * 60)
print("RustGS Training Progress")
print("=" * 60)
print(f"Iteration:  {iter_num:,} / 30,000 ({progress:.1f}%)")
print(f"Frame:      {frame} / 346")
print(f"Loss:       {loss:.6f}")
print(f"Gaussians:  {visible:,} / {gaussians:,} visible")
print()
print(f"Speed:      {step_time:.2f}s per iteration")
print(f"Elapsed:    {elapsed_min:.1f} minutes")
print(f"Remaining:  ~{remaining:.0f} minutes")
print("=" * 60)
print()
print("Last 3 iterations:")
for line in lines:
    if 'Metal iter' in line:
        # 提取关键信息
        m = re.search(r'Metal iter\s+\d+/30000.*?loss\s+([\d.]+)', line)
        if m:
            l = m.group(1)
            iter_m = re.search(r'Metal iter\s+(\d+)', line)
            if iter_m:
                print(f"  iter {iter_m.group(1):>5}: loss={l}")