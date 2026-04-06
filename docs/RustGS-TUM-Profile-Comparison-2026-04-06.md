# RustGS TUM Profile Comparison 2026-04-06

## 目标

只用真实 `TUM RGB-D` 数据比较 `legacy-metal` 和 `litegs-mac-v1` 的训练行为，避免被 `Nerfstudio` 或临时 fixture 干扰。

这份报告回答四个问题：

1. `LiteGS` 在真实深度监督下是否持续优于 `legacy`。
2. `LiteGS` 的 `SH` 激活是否真的发生。
3. `LiteGS` 的 topology 分支是否只是“代码存在”，还是已经在真实数据上开始工作。
4. 这些收益对应的时间代价是多少。

## 实验设置

- 数据集: `test_data/tum/rgbd_dataset_freiburg1_xyz`
- 采样参数: `max_frames=180`, `frame_stride=30`
- 实际加载帧数: `6`
- `metal_render_scale=0.25`
- `max_initial_gaussians=2000`
- 运行环境: macOS host Metal
- 执行入口: 临时 runner `/tmp/rustgs-smoke-runner`

对比了五个训练预算：

- `48` iterations: 观察早期 profile 分叉
- `120` iterations: 观察 `SH` 激活、topology 和更长一点的收敛趋势
- `240` iterations: 观察 topology 多轮循环后，loss 与高斯数量如何继续演化
- `360` iterations: 观察 LiteGS 的 topology 扩张是否开始收敛，以及渲染质量是否继续兑现
- `5000` iterations: 观察在极长预算下，PSNR 是否继续显著增长，还是已经进入收益递减区

另外补了 `240` 和 `360` iterations 的渲染评估：

- 先把两条 profile 的训练结果导出为 scene PLY
- 再用 `RustGS/examples/evaluate_psnr.rs` 在同一组 `6` 帧上渲染并计算 `PSNR`
- 评估分辨率同样使用 `render_scale=0.25`，即 `160x120`
- 为了保证 `LiteGS` 评估有效，`evaluate_psnr` 已修正为支持 `SH scene`，不再把 `spherical_harmonics` 场景错误地按 `RGB-only` 渲染

## 结果

### 48 Iterations

| Profile | Elapsed ms | Final loss | Final gaussians | Active SH degree | Rotation frozen | Densify | Prune | Opacity reset |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `legacy-metal` | 4865.137 | 0.734603 | 1410 | 0 | `true` | 0 | 0 | 0 |
| `litegs-mac-v1` | 5816.831 | 0.620402 | 1410 | 1 | `false` | 0 | 0 | 0 |

### 120 Iterations

| Profile | Elapsed ms | Final loss | Final gaussians | Active SH degree | Rotation frozen | Densify | Added | Prune | Removed | Opacity reset |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `legacy-metal` | 11694.116 | 0.733092 | 1410 | 0 | `true` | 0 | 0 | 0 | 0 | 0 |
| `litegs-mac-v1` | 14377.709 | 0.618888 | 1461 | 3 | `false` | 2 | 61 | 1 | 10 | 1 |

### 240 Iterations

| Profile | Elapsed ms | Final loss | Final gaussians | Active SH degree | Rotation frozen | Densify | Added | Prune | Removed | Opacity reset |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `legacy-metal` | 23123.372 | 0.728244 | 1410 | 0 | `true` | 0 | 0 | 0 | 0 | 0 |
| `litegs-mac-v1` | 34449.229 | 0.587451 | 2240 | 3 | `false` | 6 | 846 | 2 | 16 | 3 |

### 360 Iterations

| Profile | Elapsed ms | Final loss | Final gaussians | Active SH degree | Rotation frozen | Densify | Added | Prune | Removed | Opacity reset |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `legacy-metal` | 35327.527 | 0.717433 | 1410 | 0 | `true` | 0 | 0 | 0 | 0 | 0 |
| `litegs-mac-v1` | 64268.784 | 0.449221 | 2240 | 3 | `false` | 6 | 846 | 2 | 16 | 4 |

### 5000 Iterations

| Profile | Elapsed ms | Final loss | Final gaussians | Active SH degree | Rotation frozen | Densify | Added | Prune | Removed | Opacity reset |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `litegs-mac-v1` | 2516855.085 | 0.298543 | 2240 | 3 | `false` | 25 | 914 | 21 | 84 | 66 |

### 240 Iterations Render Evaluation

| Profile | PSNR mean dB | Median dB | Min dB | Max dB | Std dB |
|---|---:|---:|---:|---:|---:|
| `legacy-metal` | 4.1810 | 4.0032 | 3.6349 | 4.8401 | 0.4524 |
| `litegs-mac-v1` | 4.8778 | 4.1798 | 3.9907 | 8.1299 | 1.4702 |

### 360 Iterations Render Evaluation

| Profile | PSNR mean dB | Median dB | Min dB | Max dB | Std dB |
|---|---:|---:|---:|---:|---:|
| `legacy-metal` | 4.1608 | 4.0031 | 3.6350 | 4.7505 | 0.4243 |
| `litegs-mac-v1` | 6.9847 | 7.1773 | 5.9628 | 7.8475 | 0.6997 |

### 5000 Iterations Render Evaluation

| Profile | PSNR mean dB | Median dB | Min dB | Max dB | Std dB |
|---|---:|---:|---:|---:|---:|
| `litegs-mac-v1` | 7.1194 | 7.1713 | 5.3315 | 8.6483 | 1.4294 |

## 直接结论

### 1. LiteGS 在 TUM 上不是偶然更好，而是稳定更好

- 在 `48` iterations 时，`LiteGS` 的最终 loss 比 `legacy` 低约 `15.5%`
- 在 `120` iterations 时，`LiteGS` 的最终 loss 仍然低约 `15.6%`
- 在 `240` iterations 时，`LiteGS` 的最终 loss 比 `legacy` 低约 `19.3%`
- 在 `360` iterations 时，`LiteGS` 的最终 loss 比 `legacy` 低约 `37.4%`

也就是说，这不是一次性初始化优势；在更长训练预算下，这个差距不但保持，而且继续拉开。

### 1.1 LiteGS 的更低训练 loss 已经转化成更好的渲染指标

在 `240` iterations 的同帧渲染评估里：

- `LiteGS` 的 `PSNR mean` 比 `legacy` 高 `0.6968 dB`
- `worst-frame` 的最低分也高出 `0.3558 dB`

所以至少在当前这组 `TUM` 子集上，`LiteGS` 的训练优势并不只是 telemetry 更好，而是已经反映到最终渲染输出。

不过这个提升并不均匀：

- 最大单帧提升出现在 `frame_id=0`，增幅约 `3.2898 dB`
- 除去这帧之后，其余 5 帧的平均提升约 `0.1783 dB`
- `frame_id=60` 这帧甚至比 `legacy` 略低约 `0.075 dB`

这说明当前收益是真实存在的，但分布并不稳定，仍然带有明显的场景 / 视角依赖。

到了 `360` iterations，这个结论明显强化了：

- `LiteGS` 的 `PSNR mean` 达到 `6.9847 dB`
- 相比 `legacy` 的 `4.1608 dB`，领先 `2.8239 dB`
- 6 帧上全部都是正提升，单帧提升区间约 `1.4638 dB` 到 `3.8248 dB`

这说明在更长预算下，LiteGS 的质量收益已经不再依赖单个异常帧，而是变成了全帧一致的领先。

但把预算进一步拉到 `5000` iterations 后，趋势开始变得更微妙：

- `PSNR mean` 只从 `6.9847 dB` 提升到 `7.1194 dB`
- 增量只有 `0.1347 dB`
- 训练时间却从 `64.27s` 拉升到 `2516.86s`，也就是约 `41.95` 分钟

这说明在当前这组 `TUM` 子集上，`360 -> 5000` 已经进入非常明显的收益递减区。

而且这次提升并不均匀：

- `frame_id=0` 和 `frame_id=120` 继续变好
- `frame_id=60` 和 `frame_id=90` 反而明显变差
- 最差帧从 `5.9628 dB` 下降到 `5.3315 dB`

所以更准确的判断是：`5000` 次没有让质量崩掉，但也没有带来和时间成本匹配的稳定收益。

### 2. LiteGS 的 SH 激活已经在真实数据上发生

- `48` iterations 时，`active_sh_degree = 1`
- `120` iterations 时，`active_sh_degree = 3`

这证明当前实现不是“挂了 LiteGS 名字但一直在跑 DC-only 颜色”。

### 3. LiteGS 的 topology 分支在 TUM 上也已经开始工作

在 `120` iterations 的真实 TUM 训练里：

- `densify_events = 2`
- `densify_added = 61`
- `prune_events = 1`
- `prune_removed = 10`
- `opacity_reset_events = 1`

最终高斯数从 `1410` 增加到 `1461`，净增 `51`。  
这说明当前 LiteGS 路径不只是学习率和 SH 调度不同，连 topology 更新也开始介入训练。

在 `240` iterations 时，这种差异已经非常明显：

- `densify_events = 6`
- `densify_added = 846`
- `prune_events = 2`
- `prune_removed = 16`
- `opacity_reset_events = 3`

最终高斯数达到 `2240`。这说明 LiteGS 在当前实现里已经不是轻微扰动，而是进入了明显的 topology 扩张阶段。

但 `360` iterations 的结果也补上了一个更细的判断：

- `final_gaussians` 仍然是 `2240`
- `densify_events`、`densify_added`、`prune_events`、`prune_removed` 都没有继续增加
- 只有 `opacity_reset_events` 从 `3` 增加到 `4`

这说明 LiteGS 在 `240 -> 360` 之间已经不是继续无约束扩张，而是进入了一个“拓扑规模基本稳定、参数继续优化”的阶段。

`5000` iterations 则说明，后面 topology 并没有完全静止，只是节奏显著放缓：

- `final_gaussians` 仍然是 `2240`
- 但累计 `densify_events` 增长到 `25`
- `densify_added` 增长到 `914`
- `prune_events` 增长到 `21`
- `prune_removed` 增长到 `84`
- `opacity_reset_events` 增长到 `66`

也就是说，后期不是简单“冻结 topology”，而是进入了一个反复 reset / prune / 少量调整的长尾阶段。

### 4. LiteGS 的收益不是免费的，但代价可接受

- `48` iterations: LiteGS 比 legacy 慢约 `19.6%`
- `120` iterations: LiteGS 比 legacy 慢约 `23.0%`
- `240` iterations: LiteGS 比 legacy 慢约 `49.0%`
- `360` iterations: LiteGS 比 legacy 慢约 `81.9%`
- `5000` iterations: LiteGS 单独一条训练就耗时约 `41.95` 分钟

换来的收益是：

- 更低的最终 loss
- 非冻结的 rotation 学习
- SH 从 `0 -> 1 -> 3` 的渐进激活
- 实际发生的 densify / prune / opacity reset

按目前这组 TUM 对照看，这个 tradeoff 仍然成立，但代价已经很高。  
不过 `360` iterations 的结果也说明，这部分代价并没有白花：

- `legacy` 从 `240 -> 360` 的 `PSNR mean` 还略降了 `0.0202 dB`
- `LiteGS` 从 `240 -> 360` 的 `PSNR mean` 却继续提升了 `2.1069 dB`

也就是说，LiteGS 到这个阶段仍然在把额外训练时间转成实际画质，而 legacy 已经接近停滞。

但 `5000` iterations 把 tradeoff 的另一面也暴露出来了：

- 相比 `360` iterations，多花了约 `40.88` 分钟
- `PSNR mean` 只增加了 `0.1347 dB`

这个比例已经很差。  
所以在当前 `TUM` 小子集上，`5000` 次更像是“验证平台期存在”，而不是一个值得常规采用的训练预算。

### 5. 240 Iterations 说明两条 profile 的中程趋势已经分叉

从 `120 -> 240` iterations：

- `legacy-metal` 的 loss 只再下降约 `0.66%`
- `litegs-mac-v1` 的 loss 再下降约 `5.08%`

这说明在当前 TUM 子集上：

- `legacy-metal` 已经非常接近平缓区
- `litegs-mac-v1` 仍然处在可继续优化的阶段

这也是为什么两条 profile 的 loss 差距会在 `240` iterations 进一步拉开。

### 5.1 360 Iterations 说明 LiteGS 已经进入“继续优化画质，而不是继续长高斯”的阶段

从 `240 -> 360` iterations：

- `legacy-metal` 的 loss 再下降约 `1.48%`
- `litegs-mac-v1` 的 loss 再下降约 `23.53%`

但更关键的是：

- `legacy` 的最终高斯数保持 `1410`
- `LiteGS` 的最终高斯数也保持 `2240`

这意味着 `LiteGS` 在这 120 次迭代里，收益主要来自参数优化和 SH 表达的继续收敛，而不是继续 densify 扩张。

### 5.2 5000 Iterations 说明 LiteGS 已经明显进入收益递减区

从 `360 -> 5000` iterations：

- `final_loss` 继续下降约 `33.54%`
- 但 `PSNR mean` 只增加 `0.1347 dB`

这说明在当前设置下，训练 loss 和最终渲染质量已经不再保持高效同步。

换句话说：

- 训练目标还在继续优化
- 但可见画质收益已经非常有限

如果目标是更高 `PSNR`，那么在这组配置上，`360` 次之后就该优先考虑：

- held-out 评估
- 更合理的 topology / reset 调度
- 更合适的 render scale

而不是机械继续堆迭代数。

### 6. 绝对渲染质量仍然偏低，当前结论应理解为“LiteGS 更好”，不是“现在已经好”

需要特别强调的是，这次渲染评估的绝对 `PSNR` 仍然不高：

- `240` iterations: `legacy = 4.18 dB`, `LiteGS = 4.88 dB`
- `360` iterations: `legacy = 4.16 dB`, `LiteGS = 6.98 dB`

而且这还是在训练时使用的同一组 `6` 帧上做的评估，不是 held-out view。  
这说明在当前 `6` 帧子集 + `0.25x` 分辨率设置下，即使训练拉到 `360` iterations，也还没有到“画质已经足够好”的阶段。

因此当前更准确的判断是：

- `LiteGS` 明确优于 `legacy`
- 但两者都还处在“能比较趋势，还不能交付质量”的阶段

不过在 `360` iterations 时，这个判断可以再细化一点：

- `legacy` 仍然明显不够好
- `LiteGS` 虽然也谈不上高质量，但已经开始从“只能比较趋势”向“能看出阶段性画质提升”移动

## Telemetry 差异

### 48 Iterations Final Learning Rates

| Profile | xyz | sh_0 | sh_rest | opacity | scale | rot |
|---|---:|---:|---:|---:|---:|---:|
| `legacy-metal` | 0.0000016 | 0.0025 | 0.00025 | 0.05 | 0.005 | 0.0 |
| `litegs-mac-v1` | 0.0000016 | 0.0025 | 0.00025 | 0.025 | 0.005 | 0.001 |

### 120 Iterations Final Learning Rates

| Profile | xyz | sh_0 | sh_rest | opacity | scale | rot |
|---|---:|---:|---:|---:|---:|---:|
| `legacy-metal` | 0.0000016 | 0.0025 | 0.00025 | 0.05 | 0.005 | 0.0 |
| `litegs-mac-v1` | 0.0000016 | 0.0025 | 0.00025 | 0.025 | 0.005 | 0.001 |

### 240 Iterations Final Learning Rates

| Profile | xyz | sh_0 | sh_rest | opacity | scale | rot |
|---|---:|---:|---:|---:|---:|---:|
| `legacy-metal` | 0.0000016 | 0.0025 | 0.00025 | 0.05 | 0.005 | 0.0 |
| `litegs-mac-v1` | 0.0000016 | 0.0025 | 0.00025 | 0.025 | 0.005 | 0.001 |

### 360 Iterations Final Learning Rates

| Profile | xyz | sh_0 | sh_rest | opacity | scale | rot |
|---|---:|---:|---:|---:|---:|---:|
| `legacy-metal` | 0.0000016 | 0.0025 | 0.00025 | 0.05 | 0.005 | 0.0 |
| `litegs-mac-v1` | 0.0000016 | 0.0025 | 0.00025 | 0.025 | 0.005 | 0.001 |

这里最值得注意的是两点：

- `LiteGS` 保持 rotation 学习开启，`legacy` 仍是 `rot = 0.0`
- `LiteGS` 使用更低的 opacity 学习率，和它的 topology / opacity reset 语义是一致的

## 渲染评估工件

已导出 worst-frame review 图像，便于手工核查：

- `legacy`: `/tmp/rustgs-tum-render-eval/tum-legacy-240_psnr_review`
- `LiteGS`: `/tmp/rustgs-tum-render-eval/tum-litegs-240_psnr_review`
- `legacy-360`: `/tmp/rustgs-tum-render-eval/tum-legacy-360_psnr_review`
- `LiteGS-360`: `/tmp/rustgs-tum-render-eval/tum-litegs-360_psnr_review`
- `LiteGS-5000`: `/tmp/rustgs-tum-render-eval/tum-litegs-5000_psnr_review`

每个目录都包含：

- `*_gt.png`: ground truth
- `*_render.png`: rendered result
- `*_diff.png`: 差异可视化
- `*_strip.png`: 横向对比拼图
- `summary.tsv`: worst-frame 摘要

## 对 RustGS 当前状态的判断

### 现在可以确认的

1. `RustGS` 已经可以在真实 `TUM RGB-D` 数据上稳定跑两条 profile。
2. `litegs-mac-v1` 不只是“配置集合”，而是已经表现出真实的训练行为差异。
3. 当前 LiteGS 语义里最关键的几项已经能在真实数据上被观察到：
   - SH 渐进激活
   - rotation 学习
   - topology 更新
   - opacity reset
4. 在 `240` iterations 时，LiteGS 和 legacy 的训练动力学已经明显分叉，不再只是“同一路径上的不同超参数”。
5. 在修正 SH 渲染评估之后，LiteGS 的更低训练 loss 已经能映射到更高的渲染 PSNR。
6. 到 `360` iterations 时，LiteGS 的 topology 扩张基本停止，但渲染质量还在继续提升，这是当前最有价值的信号。
7. 到 `5000` iterations 时，LiteGS 仍然能继续压低 loss，但 `PSNR` 增长已经很小，说明当前小子集训练已经进入明显的平台期。

### 现在还不应该夸大的

1. 这仍然是小规模 `TUM` 子集，不是大规模收敛质量结论。
2. 虽然已经补了渲染 `PSNR`，但评估仍然只覆盖训练同集的 `6` 帧，没有 held-out 视角，也没有更强的感知指标。
3. 还不能据此断言 `LiteGS` 在所有真实数据上都优于 `legacy`，只能说在当前 `TUM RGB-D` 对照上结论明确。

## 下一步建议

1. 把渲染评估从“训练帧同集”扩展到 held-out 帧，避免把 in-sample 小幅提升误判为泛化收益。
2. 不建议在当前配置上继续盲目把迭代数拉得更高；更合理的是调整 `LiteGS` 后期 topology / opacity reset 调度。
3. 如果后续要做训练默认值切换，当前证据已经足够支持优先围绕 `litegs-mac-v1` 继续优化，而不是继续把时间花在 `Nerfstudio` 兼容上。
