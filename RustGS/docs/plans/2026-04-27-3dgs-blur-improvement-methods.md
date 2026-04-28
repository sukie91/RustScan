# 3DGS 训练结果发糊的论文方法整理

日期：2026-04-27  
对象：RustGS 训练质量改进  
重点问题：训练结果偏糊、细节不足、杂散 splat / floater 偏多

## 结论先行

2026-04-28 基于 `/Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap` 的长训实测后，当前推荐先不要把 AbsGS / Pixel-GS 设为默认训练策略。它们已经作为实验开关实现，但在 8000-step 长训中没有超过 baseline。当前更可靠的配置是：

| 目标 | 推荐训练参数 | 推荐导出/评估 blur | 实测结果摘要 |
|---|---|---:|---|
| 当前前 180 帧质量优先 | `--iterations 8000 --max-frames 180 --litegs-topology-freeze-after-epoch 18 --raster-cov-blur 0.3` | `0.2` 稳定评估，`0.15` 主观锐化导出 | full-180 PSNR mean `23.0782`，static-162 mean `23.6976`，`91859` splats |
| 当前前 180 帧 PSNR/体积折中 | 上一项基础上加 `--litegs-growth-select-fraction 0.14` | `0.2` 稳定评估，`0.15` 主观锐化导出 | full-180 PSNR mean `23.0013`，static-162 mean `23.6352`，`49648` splats |
| 当前前 180 帧效率优先 | 上一项基础上加 `--loss-l1-weight 0.9 --loss-ssim-weight 0.1` | `0.2` 稳定评估，`0.15` 主观锐化导出 | full-180 PSNR mean `22.9697`，static-162 mean `23.6060`，`41484` splats |
| 完整 798 帧训练折中 | `--iterations 8000 --lr-decay-iterations 8000 --raster-cov-blur 0.3 --litegs-topology-freeze-after-epoch 4` | `0.2` | PSNR mean `21.9221`，训练约 `314s` |

明确不推荐作为当前默认：

- 从训练开始使用 `--raster-cov-blur 0.2`：10k 会崩塌。
- `--loss-robust-delta`：锐度指标会上升，但 PSNR 和主观质量欠拟合。
- 抬高 `--litegs-prune-opacity-threshold 0.01`：几乎剪不掉 floater，质量反而下降。
- `--litegs-profile abs-pixel`：8000-step 长训没有超过 baseline，保留为其它数据集实验开关。
- 硬排除动态帧 `76-93`：完整和静态子集 PSNR 都下降，不建议作为当前训练默认。
- 全轨迹 `--frame-stride 4` 或简单把前 180 帧 oversample 到全轨迹训练中：都没有超过当前 baseline / prefix-180 compact，不建议继续作为主线。
- `L1-only` 搭配较大增长预算，例如 `--loss-l1-weight 1.0 --loss-ssim-weight 0.0 --litegs-growth-select-fraction 0.25`：会出现局部帧严重崩塌。
- `--max-frames 180` 不能作为完整轨迹重建的通用默认；它只适用于目标就是前 180 帧或当前默认评估口径的实验。

这些论文里的方法不能简单全部叠加。它们解决的是不同来源的“糊”：

| 成因 | 最该参考的方法 | 对 RustGS 的优先级 | 是否建议先做 |
|---|---|---:|---|
| 大 Gaussian 覆盖高频纹理但没有被 split，导致过重建/糊 | AbsGS 的 homodirectional gradient | 高 | 是 |
| 初始点云稀疏，大 Gaussian 在很多视角只擦边可见，平均梯度被稀释 | Pixel-GS 的 pixel-aware gradient | 高 | 是 |
| 近相机 floater 被过多梯度放大，越长越多 | Pixel-GS 的 depth-scaled gradient；BSGS 的空间阈值思想 | 中高 | 是，跟密度控制一起做 |
| 多尺度渲染、zoom in/out、投影滤波导致糊或锯齿 | Mip-Splatting 或 Analytic-Splatting | 中 | 先不要和密度控制混做 |
| 输入图片本身有相机运动模糊，或 COLMAP pose 明显偏 | Deblur-GS / BAD-GS / BSGS | 中低 | 只在确认输入模糊后做 |
| 训练输入是低分辨率，希望输出高分辨率细节 | SRGS / 3DGS SR 系列 | 低到中 | 只在低分辨率训练场景做 |

对当前 RustGS，最现实的第一步是做 **AbsGS + Pixel-GS 风格的 densification score**，因为它直接对应当前 topology 逻辑：

- 当前 `Trainer::accumulate_gradients` 只累计每个 splat 的平均梯度强度。
- `analyze_topology_candidates` 只用 `mean2d_grad >= growth_grad_threshold` 决定增长候选。
- 它没有记录 per-pixel 的绝对梯度、覆盖像素数、相机深度归一化权重。

因此，大 Gaussian 在高频区域被梯度抵消或被多视角平均稀释时，RustGS 很容易“不 split”，最终表现为一片糊；同时近相机 floater 如果梯度很大，会被优先增殖，导致杂散 splat。

## 1. 基线公式：3DGS 为什么会糊

3DGS 的一个 splat 用 3D Gaussian 表示：

$$
G_i(\mathbf{x}) =
\exp\left(
-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T
\boldsymbol{\Sigma}_i^{-1}
(\mathbf{x}-\boldsymbol{\mu}_i)
\right)
$$

协方差通常分解为：

$$
\boldsymbol{\Sigma}_i =
\mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^T \mathbf{R}_i^T
$$

投影到屏幕空间后：

$$
\hat{\boldsymbol{\Sigma}}_i =
\mathbf{J}\mathbf{T}
\boldsymbol{\Sigma}_i
\mathbf{T}^T\mathbf{J}^T
$$

像素颜色按前向 alpha blending：

$$
\mathbf{C}(\mathbf{u}) =
\sum_i T_i \alpha_i
G_i^{2D}(\mathbf{u})
\mathbf{c}_i,
\quad
T_i =
\prod_{j<i}\left(1-\alpha_jG_j^{2D}(\mathbf{u})\right)
$$

训练目标常用：

$$
\mathcal{L}
= (1-\lambda)\mathcal{L}_1
+ \lambda \mathcal{L}_{D\text{-}SSIM}
$$

原始 3DGS 的 densification 通常看 view-space / NDC 的 2D mean gradient 平均值：

$$
g_i =
\frac{
\sum_{k=1}^{M_i}
\sqrt{
\left(\frac{\partial L_k}{\partial \mu^{i,k}_{x}}\right)^2
+
\left(\frac{\partial L_k}{\partial \mu^{i,k}_{y}}\right)^2
}
}{M_i}
$$

当：

$$
g_i > \tau_{\mathrm{pos}}
$$

再结合 scale 判断 clone 或 split。

这个策略有两个典型问题：

1. **梯度抵消**：一个大 Gaussian 覆盖很多像素时，不同像素给出的移动方向可能相反，直接求和后梯度很小，导致本该 split 的大 Gaussian 没被 split。
2. **视角平均稀释**：一个大 Gaussian 在很多视角都可见，但很多视角只是边缘擦过，贡献像素少、梯度小，平均后低于阈值。

## 2. AbsGS：用绝对 per-pixel 梯度解决高频区域过重建

来源：[AbsGS: Recovering Fine Details for 3D Gaussian Splatting](https://arxiv.org/abs/2404.10484)

### 要借鉴什么

AbsGS 的核心是把 densification 的梯度统计从“方向相关的总梯度”改成“homodirectional gradient”。也就是先对每个像素贡献的 x/y 梯度分别取绝对值，再求和。这样可以避免高频区域里的梯度互相抵消。

对第 `i` 个 Gaussian，在一个视角里覆盖 `m` 个像素，AbsGS 统计：

$$
\hat{g}_i =
(\hat{g}_{i,x}, \hat{g}_{i,y})
$$

$$
\hat{g}_{i,x}
=
\sum_{j=1}^{m}
\left|
\frac{\partial L_j}{\partial \mu_{i,x}}
\right|,
\quad
\hat{g}_{i,y}
=
\sum_{j=1}^{m}
\left|
\frac{\partial L_j}{\partial \mu_{i,y}}
\right|
$$

用于 densification 的标量可以取：

$$
s_i^{\mathrm{abs}} =
\left\|
\hat{g}_i
\right\|_2
=
\sqrt{
\hat{g}_{i,x}^2
+
\hat{g}_{i,y}^2
}
$$

然后：

$$
s_i^{\mathrm{abs}} > \tau_{\mathrm{abs}}
$$

时将该 Gaussian 标成 split 候选。

### 为什么能改善糊

“糊”的一个主要表现是大 Gaussian 覆盖了本应由多个小 Gaussian 表达的纹理区域。原始梯度可能被抵消，导致这些 splat 不被 split。AbsGS 的绝对梯度保留了“这个区域表达不好”的强度信号，所以更容易把大 Gaussian 拆开，恢复草地、树叶、轮辐、纹理边缘等细节。

### RustGS 落地点

当前 RustGS 的近似位置：

- `RustGS/src/training/engine/trainer.rs`
  - `accumulate_gradients` 当前用 `transforms_grad[.., 0..3].abs().mean_dim(1)` 作为 `grad_2d_accum`。
- `RustGS/src/training/topology/mod.rs`
  - `build_host_snapshot_stats` 把 `grad_2d_accum / observations` 写入 `refine_weight_max`。
  - `analyze_topology_candidates` 用 `mean2d_grad >= growth_grad_threshold` 判断增长候选。

建议改成新增一条 topology 统计，而不是直接替换现有梯度：

$$
s_i^{\mathrm{split}} =
\frac{1}{M_i}
\sum_k
\sqrt{
\left(
\sum_{p \in P_i^k}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,x}^k}
\right|
\right)^2
+
\left(
\sum_{p \in P_i^k}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,y}^k}
\right|
\right)^2
}
$$

只把它用于 **split 大 Gaussian**，clone 仍可沿用当前梯度逻辑：

$$
\mathrm{split}(i)
\iff
s_i^{\mathrm{split}} > \tau_{\mathrm{split}}
\land
\max(\mathbf{s}_i) > \tau_{\mathrm{scale}}
$$

这样不会让小 Gaussian 大量 clone，风险比全量替换小。

### 注意点

AbsGS 的梯度值通常比原始 signed gradient 大很多，`growth_grad_threshold` 不能复用原值，需要重新标定。建议记录 histogram：

- 原始 `mean2d_grad`
- `abs_mean2d_grad`
- split 候选数
- split 后 Gaussian 总数
- PSNR / SSIM / LPIPS

## 3. Pixel-GS：用像素覆盖数加权，解决大 Gaussian 梯度被视角平均稀释

来源：[Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](https://arxiv.org/abs/2403.15530)

### 要借鉴什么

Pixel-GS 认为原始 3DGS 的平均梯度没有考虑一个 Gaussian 在每个视角里实际覆盖了多少像素。对于大 Gaussian，如果某些视角只在边缘擦过，梯度很小，但仍参与平均，会把真正有用视角的梯度稀释掉。

原始条件近似为：

$$
\frac{
\sum_{k=1}^{M_i}
\left\|
\nabla_{\boldsymbol{\mu}^{i,k}_{ndc}} L_k
\right\|_2
}{M_i}
>
\tau_{\mathrm{pos}}
$$

Pixel-GS 改为用覆盖像素数 `m_i^k` 加权：

$$
\frac{
\sum_{k=1}^{M_i}
m_i^k
\left\|
\nabla_{\boldsymbol{\mu}^{i,k}_{ndc}} L_k
\right\|_2
}{
\sum_{k=1}^{M_i}
m_i^k
}
>
\tau_{\mathrm{pos}}
$$

其中：

$$
\left\|
\nabla_{\boldsymbol{\mu}^{i,k}_{ndc}} L_k
\right\|_2
=
\sqrt{
\left(
\frac{\partial L_k}{\partial \mu^{i,k}_{ndc,x}}
\right)^2
+
\left(
\frac{\partial L_k}{\partial \mu^{i,k}_{ndc,y}}
\right)^2
}
$$

一个像素是否计入覆盖，论文给出的条件大意是：

$$
\sqrt{
(pix_x-\mu_{p,x}^{i,k})^2
+
(pix_y-\mu_{p,y}^{i,k})^2
}
< R_i^k
$$

$$
\prod_{j=1}^{i}(1-\alpha_{j}^{k,pix}) \ge 10^{-4},
\quad
\alpha_i^{k,pix} \ge \frac{1}{255}
$$

### 近相机 floater 抑制公式

Pixel-GS 还引入 depth-scaled gradient field，用相机距离缩放近处梯度：

$$
R_{\mathrm{scene}}
=
1.1
\max_i
\left\|
\mathbf{C}_i
-
\frac{1}{N}\sum_{j=1}^{N}\mathbf{C}_j
\right\|_2
$$

$$
f(i,k)
=
\mathrm{clip}
\left(
\left(
\frac{\mu_{c,z}^{i,k}}
{\gamma_{\mathrm{depth}} R_{\mathrm{scene}}}
\right)^2,
0,
1
\right)
$$

最终 densification 条件变成：

$$
\frac{
\sum_{k=1}^{M_i}
m_i^k f(i,k)
\left\|
\nabla_{\boldsymbol{\mu}^{i,k}_{ndc}} L_k
\right\|_2
}{
\sum_{k=1}^{M_i}
m_i^k
}
>
\tau_{\mathrm{pos}}
$$

### 为什么能改善糊和杂散 splat

- 对糊：覆盖像素多的视角权重大，稀疏初始点区域的大 Gaussian 更容易 split / clone。
- 对 floater：靠近相机的 splat 因为投影面积大，天然会得到更多梯度；`f(i,k)` 会降低近处 splat 的增长冲动，减少近相机杂散 splat 增殖。

### RustGS 落地点

建议新增 topology 统计字段：

$$
m_i^k = \#\{p \mid p \text{ receives non-trivial contribution from } G_i\}
$$

先做近似版也可以：

$$
\tilde{m}_i^k
=
\pi r_{x,i}^k r_{y,i}^k
$$

其中 `r_x, r_y` 来自 projected conic / extent。精确版需要在 raster backward 中对每个 splat 做 atomic 计数，成本更高。

推荐 RustGS 的第一版组合分数：

$$
h_i^k =
\sqrt{
\left(
\sum_{p \in P_i^k}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,x}^k}
\right|
\right)^2
+
\left(
\sum_{p \in P_i^k}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,y}^k}
\right|
\right)^2
}
$$

$$
s_i^{\mathrm{abs\_pixel}}
=
\frac{
\sum_k
m_i^k f(i,k) h_i^k
}{
\sum_k m_i^k + \epsilon
}
$$

然后：

$$
\mathrm{split}(i)
\iff
s_i^{\mathrm{abs\_pixel}} > \tau_{\mathrm{split}}
\land
\max(\mathbf{s}_i) > \tau_{\mathrm{scale}}
$$

这个分数把 AbsGS 和 Pixel-GS 融合在同一个“split 大 Gaussian”场景里，是最适合 RustGS 先试的版本。

### 注意点

Pixel-GS 的分母没有乘 `f(i,k)`，这是为了只降低近处梯度的分子贡献，而不是把权重完全归一回来。RustGS 第一版也建议保持这个形式。

## 4. Mip-Splatting：用 3D 平滑和 2D mip filter 解决采样率变化造成的糊/锯齿

来源：[Mip-Splatting: Alias-free 3D Gaussian Splatting](https://arxiv.org/abs/2311.16493)

### 要借鉴什么

Mip-Splatting 解决的不是 densification 不足，而是采样率变化下的 aliasing、dilation、erosion、过平滑等问题。它有两部分：

1. 3D smoothing filter：限制 3D Gaussian 的最高频率。
2. 2D Mip filter：用近似像素面积积分的滤波替换原始屏幕空间 dilation。

### 采样率公式

给定深度 `d` 和像素单位焦距 `f`，世界空间采样间隔为：

$$
\hat{T}
=
\frac{1}{\hat{\nu}}
=
\frac{d}{f}
$$

对第 `k` 个 primitive，取所有可见训练相机中的最大采样频率：

$$
\hat{\nu}_k
=
\max
\left(
\left\{
\mathbb{1}_n(\mathbf{p}_k)
\frac{f_n}{d_n}
\right\}_{n=1}^{N}
\right)
$$

### 3D smoothing filter

对原始 Gaussian 做低通卷积：

$$
G_k(\mathbf{x})_{\mathrm{reg}}
=
(G_k \otimes G_{\mathrm{low}})(\mathbf{x})
$$

两个 Gaussian 卷积后协方差相加：

$$
G_k(\mathbf{x})_{\mathrm{reg}}
=
\sqrt{
\frac{
|\boldsymbol{\Sigma}_k|
}{
|\boldsymbol{\Sigma}_k + \frac{s}{\hat{\nu}_k}\mathbf{I}|
}
}
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\mathbf{p}_k)^T
(\boldsymbol{\Sigma}_k + \frac{s}{\hat{\nu}_k}\mathbf{I})^{-1}
(\mathbf{x}-\mathbf{p}_k)
\right)
$$

### 2D Mip filter

替换 3DGS 的屏幕空间 dilation：

$$
G_k^{2D}(\mathbf{x})_{\mathrm{mip}}
=
\sqrt{
\frac{
|\boldsymbol{\Sigma}_k^{2D}|
}{
|\boldsymbol{\Sigma}_k^{2D}+s\mathbf{I}|
}
}
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\mathbf{p}_k)^T
(\boldsymbol{\Sigma}_k^{2D}+s\mathbf{I})^{-1}
(\mathbf{x}-\mathbf{p}_k)
\right)
$$

### 是否适合 RustGS 现在做

适合，但不是第一优先级。原因：

- 它需要改 forward projection / rasterization 的 covariance 和 opacity compensation。
- 它可能改变当前训练动态，和 densification 调参耦合较强。
- 如果当前问题主要是“训练出来就糊”，而不是“不同分辨率/不同距离渲染出问题”，应先修 densification。

建议等 AbsGS + Pixel-GS 分数稳定后，再单独做 Mip-Splatting 分支。

## 5. Analytic-Splatting：用像素面积解析积分替换中心点采样

来源：[Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](https://arxiv.org/abs/2403.11056)

### 要借鉴什么

Analytic-Splatting 认为 3DGS 对每个像素只在中心点采样，这会对像素 footprint 变化不敏感。它改为计算 Gaussian 在像素窗口内的积分响应。

1D Gaussian CDF：

$$
G(x)=
\int_{-\infty}^{x}
g(t)dt,
\quad
g(x)=
\frac{1}{\sqrt{2\pi}}
\exp\left(-\frac{x^2}{2}\right)
$$

像素窗口积分：

$$
\mathcal{I}_g(u)
=
\int_{u-\frac{1}{2}}^{u+\frac{1}{2}}
g(x)dx
=
G(u+\frac{1}{2}) - G(u-\frac{1}{2})
$$

论文用 logistic 函数近似 Gaussian CDF：

$$
S(x)=
\frac{1}{
1+\exp(-1.6x-0.07x^3)
}
$$

于是：

$$
\mathcal{I}_g(u)
\approx
S(u+\frac{1}{2})
-
S(u-\frac{1}{2})
$$

2D 情况下，把屏幕空间 covariance 做特征分解：

$$
\lambda_1,\lambda_2
\quad
\Rightarrow
\quad
\sigma_1=\sqrt{\lambda_1},
\quad
\sigma_2=\sqrt{\lambda_2}
$$

在旋转后的像素坐标 `\tilde{u}` 中，2D pixel integral 近似为：

$$
\mathcal{I}^{2D}_{g}(\mathbf{u})
\approx
2\pi\sigma_1\sigma_2
\left[
S_{\sigma_1}(\tilde{u}_x+\frac{1}{2})
-
S_{\sigma_1}(\tilde{u}_x-\frac{1}{2})
\right]
\left[
S_{\sigma_2}(\tilde{u}_y+\frac{1}{2})
-
S_{\sigma_2}(\tilde{u}_y-\frac{1}{2})
\right]
$$

最终 alpha blending 替换为：

$$
\mathbf{C}(\mathbf{u})
=
\sum_i
T_i
\mathcal{I}^{2D}_{g_i}(\mathbf{u})
\alpha_i
\mathbf{c}_i
$$

$$
T_i
=
\prod_{j<i}
\left(
1-
\mathcal{I}^{2D}_{g_j}(\mathbf{u})
\alpha_j
\right)
$$

### 与 Mip-Splatting 的关系

这两者都解决 pixel footprint / anti-aliasing，但思路不同：

- Mip-Splatting 是滤波近似，速度更友好，但可能过平滑。
- Analytic-Splatting 更接近像素面积积分，细节保留更好，但 shader 成本更高。

不建议把两者直接叠加。RustGS 后续应该二选一做 A/B：

- 目标是低风险和速度：先做 Mip-Splatting。
- 目标是细节保真和研究质量：做 Analytic-Splatting。

## 6. BSGS / Deblur-GS / BAD-GS：只在输入存在真实运动模糊时使用

主要来源：[BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring](https://arxiv.org/abs/2510.12493)

### 要借鉴什么

这类方法解决的是输入图片本身因相机运动而模糊，不是普通 densification 失败。核心思想是：模糊图像是曝光时间内多个潜在清晰图像的积分或加权和。

连续模型：

$$
\hat{\mathbf{B}}
=
\phi
\int_0^\tau
\mathbf{C}_t dt
$$

离散近似：

$$
\hat{\mathbf{B}}
\approx
\sum_{i=0}^{n-1}
w_i \mathbf{C}_i
$$

相机在曝光时间内的 SE(3) 插值：

$$
T_t
=
T_{\mathrm{start}}
\cdot
\exp
\left(
\frac{t}{\tau}
\log
\left(
T_{\mathrm{start}}^{-1}
T_{\mathrm{end}}
\right)
\right)
$$

训练损失仍可使用：

$$
\mathcal{L}
=
(1-\lambda)\mathcal{L}_1
+
\lambda\mathcal{L}_{D\text{-}SSIM}
$$

BSGS 的 subframe gradient aggregation：

$$
\nabla \mu_{\mathrm{agg}}(x)
=
\max_{i \in \{1,\dots,n\}}
\left|
\nabla \mu_i(x)
\right|
\cdot
\mathrm{sign}
\left(
\nabla \mu_{i_{\max}}(x)
\right)
$$

这里 `i_max` 是对应最大梯度幅值的 subframe。

BSGS 还给 densification threshold 加空间和时间调度：

$$
\tau_s
=
\tau_0
\left(
1+\alpha e^{-\beta d}
\right)
$$

$$
\tau_t(t)
=
\begin{cases}
\tau_0(1-\gamma t),
& t \in \mathrm{Stage\ I}
\\
\tau_0\eta^{t-t_{\mathrm{split}}},
& t \in \mathrm{Stage\ II}
\end{cases}
$$

最终：

$$
\hat{\tau}
=
\tau_s \tau_t(t)
$$

### RustGS 是否该做

默认不建议先做完整 Deblur-GS / BAD-GS / BSGS。原因：

- 需要引入每张训练图的曝光轨迹参数或 pose refinement。
- 当前 RustGS 里 `learnable_viewproj` 等 pose 相关选项还没有完整闭环。
- 如果输入图像本身不是运动模糊，使用 deblur 模型会把问题复杂化，甚至引入虚假细节。

可以先借鉴 BSGS 的一个轻量思想：**动态 densification threshold**。

例如在早期提高近相机 splat 的 split 阈值，减少 floater 过早增殖：

$$
\tau_i(t)
=
\tau_0
\left(1+\alpha e^{-\beta z_i}\right)
\cdot
\rho(t)
$$

其中：

$$
\rho(t)
=
\begin{cases}
\rho_{\mathrm{early}} > 1,
& t < t_{\mathrm{warmup}}
\\
1,
& t \ge t_{\mathrm{warmup}}
\end{cases}
$$

这比引入完整曝光轨迹安全。

## 7. SRGS：低分辨率训练时的高频先验，不是普通糊的首选解

来源：[SRGS: Super-Resolution 3D Gaussian Splatting](https://arxiv.org/abs/2404.10318)

### 要借鉴什么

SRGS 适用于训练图像是低分辨率、但希望输出高分辨率 novel view 的场景。它把 2D SR 模型生成的伪 HR 图像作为监督，同时用 render-and-downsample 约束保证多视角一致。

2D SR prior：

$$
\mathbf{I}_{\mathrm{HR}}^v
=
\mathcal{M}
\left(
\mathbf{I}_{\mathrm{LR}}^v
\right)
$$

统一目标：

$$
\mathcal{L}
=
\lambda_e
\mathcal{L}_{\mathrm{prior}}
+
(1-\lambda_e)
\mathcal{L}_{\mathrm{reg}}
$$

HR prior 监督：

$$
\mathcal{L}_{\mathrm{prior}}
=
(1-\lambda_{\mathrm{tex}})
\mathcal{L}_1
\left(
\tilde{\mathbf{I}}_{\mathrm{HR}}^v,
\hat{\mathbf{I}}^v(s_{\mathrm{HR}})
\right)
+
\lambda_{\mathrm{tex}}
\mathcal{L}_{D\text{-}SSIM}
\left(
\tilde{\mathbf{I}}_{\mathrm{HR}}^v,
\hat{\mathbf{I}}^v(s_{\mathrm{HR}})
\right)
$$

render-and-downsample 一致性：

$$
\mathcal{L}_{\mathrm{reg}}
=
(1-\lambda_{\mathrm{cvc}})
\mathcal{L}_1
\left(
\mathbf{I}_{\mathrm{LR}}^v,
\mathcal{F}
\left(
\hat{\mathbf{I}}^v(s_{\mathrm{HR}})
\right)
\right)
+
\lambda_{\mathrm{cvc}}
\mathcal{L}_{D\text{-}SSIM}
\left(
\mathbf{I}_{\mathrm{LR}}^v,
\mathcal{F}
\left(
\hat{\mathbf{I}}^v(s_{\mathrm{HR}})
\right)
\right)
$$

### RustGS 是否该做

如果训练数据本身是低分辨率、下采样或纹理不足，SRGS 有价值。但对当前“普通训练发糊”问题，它不是第一优先级，因为：

- 需要接入外部 2D SR 模型或预生成 pseudo-HR targets。
- 容易引入 hallucinated texture。
- 对几何/密度控制问题帮助有限。

## 8. 最新但不建议直接搬的方向

arXiv 上 2026 年还有 [MipSLAM: Alias-Free Gaussian Splatting SLAM](https://arxiv.org/abs/2603.06989)。它把 anti-aliasing、pose graph、频域优化放到 SLAM 框架里，适合在线 RGB-D/SLAM 系统。对 RustGS 的离线训练，可借鉴的是“多分辨率/采样率下要显式处理 anti-aliasing 和 pose noise”，但不建议直接搬完整 SLAM pipeline。

## 9. 推荐 RustGS 实施路线

### Phase 1：补充 topology 统计

新增统计目标：

$$
A_{i,x}
=
\sum_{k,p}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,x}^k}
\right|,
\quad
A_{i,y}
=
\sum_{k,p}
\left|
\frac{\partial L_p^k}{\partial \mu_{i,y}^k}
\right|
$$

$$
M_i = \sum_k m_i^k
$$

$$
D_i =
\sum_k
m_i^k f(i,k)
$$

如果一开始不想改 raster backward，可先用 projected extent 估算：

$$
m_i^k \approx \pi r_{x,i}^k r_{y,i}^k
$$

需要动的模块：

- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/topology/bridge.rs`
- `RustGS/src/training/topology/mod.rs`
- 精确版还需要动 raster backward WGSL / CUDA 等价逻辑。

### Phase 2：新增 split-only score

推荐第一版 score：

$$
s_i =
\frac{
\sum_k
m_i^k f(i,k)
\sqrt{A_{i,x,k}^2 + A_{i,y,k}^2}
}{
\sum_k m_i^k + \epsilon
}
$$

split 条件：

$$
s_i > \tau_{\mathrm{split}}
\land
\max(\mathbf{s}_i) > \tau_{\mathrm{scale}}
$$

clone 条件先保持当前逻辑：

$$
g_i^{\mathrm{old}} > \tau_{\mathrm{clone}}
\land
\max(\mathbf{s}_i) \le \tau_{\mathrm{scale}}
$$

这样可以把“恢复细节”和“填空扩点”拆开调参。

### Phase 3：加入近相机 floater 抑制

先用 Pixel-GS 的深度缩放：

$$
f(i,k)
=
\mathrm{clip}
\left(
\left(
\frac{z_i^k}
{\gamma_{\mathrm{depth}}R_{\mathrm{scene}}}
\right)^2,
0,
1
\right)
$$

如果仍有很多近处 floater，再加 BSGS 风格的动态阈值：

$$
\tau_i(t)
=
\tau_0
\left(
1+\alpha e^{-\beta z_i}
\right)
\rho(t)
$$

### Phase 4：等密度控制稳定后，再做 anti-aliasing 分支

先选一个：

- Mip-Splatting：实现成本相对低，更适合工程落地。
- Analytic-Splatting：质量潜力更高，但 shader 成本和数值风险更高。

不要同时做两者，否则结果难以归因。

### Phase 5：只在确认输入运动模糊时做 Deblur 类方法

判断标准：

- 原始训练图有明显拖影。
- COLMAP poses 在模糊帧上不稳定。
- 同一物体边缘在多视角下存在系统性拉伸。

满足这些条件再考虑 BSGS / BAD-GS / Deblur-GS，否则不做。

## 10. 评估方案

每个阶段至少跑以下 ablation：

| 实验 | 改动 |
|---|---|
| baseline | 当前 RustGS |
| AbsGS-score | 只加 absolute / homodirectional split score |
| AbsGS + Pixel coverage | 加覆盖像素数加权 |
| AbsGS + Pixel + depth scale | 再加近相机梯度缩放 |
| + dynamic threshold | 再加近相机/早期 split 阈值调度 |
| Mip 或 Analytic | 单独分支评估 anti-aliasing |

建议记录：

- PSNR / SSIM / LPIPS
- Gaussian 总数
- split / clone / prune 次数
- 大 scale splat 数量：

$$
N_{\mathrm{large}}
=
\#\{i \mid \max(\mathbf{s}_i) > \tau_{\mathrm{scale}}\}
$$

- 低 opacity splat 数量：

$$
N_{\mathrm{low\_opacity}}
=
\#\{i \mid \alpha_i < \tau_{\alpha}\}
$$

- 近相机 floater 数量，可粗略定义为：

$$
N_{\mathrm{near\_floater}}
=
\#\{i \mid z_i < z_{\mathrm{near}} \land \alpha_i < \tau_{\alpha}\}
$$

主观评估要固定 crop：

- 草地/树叶/轮辐/栏杆等高频纹理。
- 背景远处细节。
- 相机近处是否出现漂浮点。
- TUM / COLMAP 数据中边缘是否仍然发糊。

## 11. 推荐先实现的最小闭环

第一版不要碰 renderer anti-aliasing，也不要做 pose deblur。只做 topology：

1. 新增 `abs_mean2d_grad` 或 `split_grad_score`。
2. 新增 `projected_coverage` 近似统计。
3. 用：

$$
s_i^{\mathrm{topology}}
=
\frac{
\sum_k
m_i^k f(i,k)
\left\|
\hat{g}_i^k
\right\|_2
}{
\sum_k m_i^k + \epsilon
}
$$

4. 只把它用于 split 大 Gaussian。
5. 保留现有 clone / prune 逻辑。
6. 加 telemetry histogram，先跑固定场景对比。

这是对“糊”和“杂散 splat”同时最有针对性的改法。

## 12. 接下来要开发的内容

下面按可交付顺序拆任务。核心原则是先把“训练糊”的 densification 原因定位准，再逐步增强；不要一开始就同时改 renderer、pose、loss 和 topology，否则结果无法归因。

### Milestone A：训练诊断和 telemetry 先补齐

目标：先知道糊来自 under-split、floater、pose/数据、还是渲染滤波。

要开发：

1. 在训练 telemetry 中记录 topology 分布：
   - 当前 Gaussian 总数。
   - 每次 topology step 的 split / clone / prune 数。
   - `mean2d_grad` 的 min / p50 / p90 / p99 / max。
   - `max_scale` 的 p50 / p90 / p99 / max。
   - opacity 的 p10 / p50 / p90。
   - prune 候选数量和实际 prune 数。

2. 新增固定 crop 评估输出：
   - 每 N 次保存同一组训练视角和测试视角 crop。
   - 至少包含高频纹理区域、远景区域、近相机区域。
   - 文件名带 iteration 和 Gaussian count，便于横向比较。

3. 新增“可疑糊类型”统计：

$$
R_{\mathrm{large\_lowgrad}}
=
\frac{
\#\{i \mid \max(\mathbf{s}_i)>\tau_{\mathrm{scale}}
\land g_i < \tau_{\mathrm{grad}}\}
}{
\#\{i \mid \max(\mathbf{s}_i)>\tau_{\mathrm{scale}}\}+\epsilon
}
$$

这个比率高，说明大 splat 多但梯度触发弱，符合 AbsGS 要解决的过重建。

验收标准：

- 不改变训练结果，只增加日志和 report。
- baseline 的 PSNR / SSIM 与当前一致。
- 能从 report 里看到 topology step 前后的 Gaussian count 和梯度分布。

涉及文件：

- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/metrics.rs`
- `RustGS/src/bin/rustgs/train_command.rs`
- `RustGS/experiments/*.md`

### Milestone B：实现 split-only AbsGS 统计

目标：解决高频区域大 Gaussian 不 split 导致的糊。

关键要求：**AbsGS 必须在 per-pixel 梯度贡献累加前取绝对值**。不能对当前已经聚合后的 `transforms_grad.abs()` 直接当作 AbsGS，因为那已经错过了梯度抵消发生的位置。

需要新增的核心统计：

$$
A^k_{i,x}
=
\sum_{p \in P_i^k}
\left|
\frac{\partial L^k_p}{\partial \mu^k_{i,x}}
\right|,
\quad
A^k_{i,y}
=
\sum_{p \in P_i^k}
\left|
\frac{\partial L^k_p}{\partial \mu^k_{i,y}}
\right|
$$

$$
h_i^k
=
\sqrt{
(A^k_{i,x})^2
+
(A^k_{i,y})^2
}
$$

第一版可以先不做 pixel coverage，只做：

$$
s_i^{\mathrm{abs}}
=
\frac{
\sum_k h_i^k
}{
M_i+\epsilon
}
$$

要开发：

1. 在 raster backward shader 增加 per-splat `abs_mean2d_grad_accum`。
2. Rust 侧新增 tensor accumulator：
   - `abs_grad_2d_accum`
   - `abs_grad_observations` 或复用有效可见计数。
3. `TopologySnapshot` 增加 `abs_grad_2d_accum`。
4. `MetalGaussianStats` 增加 `abs_mean2d_grad` 或 `split_grad_score`。
5. `TopologyCandidateInfo` 增加：
   - `split_score`
   - `split_candidate`
   - `clone_candidate`
6. `analyze_topology_candidates` 改为：
   - 大 scale 使用 `split_score` 判断 split。
   - 小 scale 保持旧 `mean2d_grad` 判断 clone。

建议条件：

$$
\mathrm{split}(i)
\iff
s_i^{\mathrm{abs}} > \tau_{\mathrm{split}}
\land
\max(\mathbf{s}_i) > \tau_{\mathrm{scale}}
\land
\alpha_i > \tau_{\alpha,\mathrm{min}}
$$

$$
\mathrm{clone}(i)
\iff
g_i^{\mathrm{old}} > \tau_{\mathrm{clone}}
\land
\max(\mathbf{s}_i) \le \tau_{\mathrm{scale}}
$$

验收标准：

- 增加一个开关，例如 `--litegs-split-score abs`，默认可先关闭。
- 关闭时 baseline bit-level 或数值近似一致。
- 打开时高频区域 split 数上升，但 clone 数不应暴涨。
- `cargo test -p rustgs --all-targets` 通过。

涉及文件：

- `RustGS/src/training/backward/rasterize_bwd.rs`
- `RustGS/src/training/shaders/rasterize_backwards.wgsl`
- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/topology/bridge.rs`
- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/bin/rustgs/train_command.rs`

### Milestone C：加入 Pixel-GS 覆盖像素权重

目标：解决大 Gaussian 在多视角平均中梯度被稀释的问题。

推荐先做近似覆盖面积，不要第一版就做精确 per-pixel atomic：

$$
\tilde{m}_i^k
=
\pi r^k_{x,i}r^k_{y,i}
$$

其中 `r_x, r_y` 来自 projected covariance / conic 的屏幕空间半径估算。这个近似不能替代论文精确定义，但足够验证 Pixel-GS 思路是否对 RustGS 有收益。

加权分数：

$$
s_i^{\mathrm{abs\_pixel}}
=
\frac{
\sum_k
\tilde{m}_i^k h_i^k
}{
\sum_k \tilde{m}_i^k + \epsilon
}
$$

要开发：

1. 从 forward projection 输出或 projected splat buffer 中得到屏幕空间 coverage 近似。
2. 训练循环新增 `coverage_accum`。
3. topology readback 新增 `coverage_accum`。
4. `split_score` 切换为 coverage weighted score。
5. telemetry 记录 `coverage` 分布，避免极端大 splat 直接支配训练。

验收标准：

- 大 Gaussian 的 split 候选更集中在稀疏点云和高频纹理区域。
- Gaussian 总数增长可控，不超过预算。
- 近相机区域没有因为覆盖面积大而明显产生更多 floater。

涉及文件：

- `RustGS/src/training/forward/projection.rs`
- `RustGS/src/training/shaders/project_forward.wgsl`
- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/topology/bridge.rs`
- `RustGS/src/training/topology/mod.rs`

### Milestone D：加入 depth-scaled floater 抑制

目标：降低近相机 splat 因投影面积过大而被优先增殖的问题。

先实现 Pixel-GS 的缩放：

$$
f(i,k)
=
\mathrm{clip}
\left(
\left(
\frac{z_i^k}
{\gamma_{\mathrm{depth}}R_{\mathrm{scene}}}
\right)^2,
0,
1
\right)
$$

RustGS 的 score：

$$
s_i^{\mathrm{final}}
=
\frac{
\sum_k
\tilde{m}_i^k f(i,k) h_i^k
}{
\sum_k \tilde{m}_i^k + \epsilon
}
$$

要开发：

1. 计算 `R_scene`，使用训练相机中心半径，保持和 Pixel-GS 一致。
2. 在 projection 或 trainer 中记录 per-splat 当前视角 camera-space depth。
3. 增加 `--litegs-depth-grad-scale` 和 `--litegs-depth-scale-gamma`。
4. 默认 `gamma_depth` 从 `0.37` 附近开始，但必须作为可调参数。

验收标准：

- 近相机低透明度 splat 数量下降。
- 远景和中景高频区域 split 不明显被压制。
- 对 bounded indoor 和 unbounded outdoor 至少各跑一个场景。

### Milestone E：调参和策略固化

目标：把可用的策略变成默认 profile 或可复现实验 profile。

要开发：

1. 增加 profile：
   - `baseline`
   - `abs-split`
   - `abs-pixel`
   - `abs-pixel-depth`
2. report 中记录全部关键超参。
3. 写固定实验文档，包含命令、数据集、结果表、crop。
4. 根据实验确定默认值：
   - `tau_split`
   - `tau_clone`
   - `tau_scale`
   - `gamma_depth`
   - `growth_select_fraction`

验收标准：

- 至少一个当前发糊场景有清晰改善。
- Gaussian 总数和训练时间没有不可接受增长。
- floater 主观数量不增加，最好下降。
- 结果能用同一命令复现。

### Milestone F：再评估 Mip-Splatting / Analytic-Splatting

只有在 topology 改完后仍存在“缩放视角下糊/锯齿/亮度扩散”时再做。

建议先做 Mip-Splatting，因为工程风险低于 Analytic-Splatting：

1. 在 projected covariance 上替换当前 dilation / compensation。
2. 加 `--render-filter baseline|mip`。
3. 单独跑多尺度测试，不和 densification 改动混在一个实验里。

验收标准：

- 多分辨率/多距离测试下 aliasing 降低。
- 正常分辨率下不明显过平滑。
- renderer 性能下降可接受。

## 13. 三维视觉专家视角的问题审查

下面是从 3DGS 训练和多视图重建角度看，这份方案还需要特别注意的问题。

### 13.1 当前 RustGS 的梯度统计可能不是论文里的 mean2d gradient

当前 `Trainer::accumulate_gradients` 取的是 `transforms_grad[.., 0..3]` 的绝对值均值。需要确认这三个量到底是：

- 3D world position gradient；
- camera-space mean gradient；
- 还是 projected 2D mean gradient 经过 `project_bwd` 链式传播后的结果。

AbsGS / Pixel-GS 都要求的是 view-space / screen-space mean gradient，且最好在 raster backward 中拿到每个像素对 projected mean 的贡献。如果只使用已经回传到 3D transform 的梯度，可能会混入相机 Jacobian、深度、scale、rotation 等因素，导致 densification score 和论文定义不一致。

结论：开发前要先画清楚梯度路径：

$$
\frac{\partial L}{\partial \mathbf{u}_i}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{c,i}}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{w,i}}
$$

AbsGS 应优先统计左侧的：

$$
\frac{\partial L}{\partial \mathbf{u}_i}
=
\left(
\frac{\partial L}{\partial u_{i,x}},
\frac{\partial L}{\partial u_{i,y}}
\right)
$$

### 13.2 不能对聚合后的梯度取 abs 冒充 AbsGS

错误做法：

$$
\left|
\sum_p
\frac{\partial L_p}{\partial \mu_{i,x}}
\right|
$$

AbsGS 需要的是：

$$
\sum_p
\left|
\frac{\partial L_p}{\partial \mu_{i,x}}
\right|
$$

这两个在高频纹理区域差异最大，也是 AbsGS 能解决糊的原因。如果实现时只在 `transforms_grad` 聚合后 `abs()`，基本不能解决 gradient collision。

### 13.3 coverage 不能直接用 tile intersection 代替像素覆盖

RustGS 里已经有 `intersect_counts`，但它更像 tile intersection count，不等价于 Pixel-GS 的有效贡献像素数 `m_i^k`。如果直接用 tile 数：

- 大 splat 会被粗略放大；
- tile 边界会带来阶梯误差；
- 很多 alpha 很低的边缘区域也可能被算进去。

推荐第一版用 projected ellipse area 作为近似，并在文档和代码里明确是 approximation：

$$
\tilde{m}_i^k
=
\pi r_x r_y
$$

精确版再按 Pixel-GS 条件统计有效像素：

$$
\alpha_i^{k,pix} \ge \frac{1}{255},
\quad
T_i^{k,pix} \ge 10^{-4}
$$

### 13.4 denominator 要用参与视角，不要用所有训练迭代

Pixel-GS 的 `M_i` 是 Gaussian 参与计算的视角数，当前 RustGS 的 `num_observations` 每次迭代对所有 splat 加 1，会把不可见 splat 的梯度进一步稀释。

如果继续使用：

$$
g_i =
\frac{\mathrm{grad\_accum}_i}{\mathrm{all\_iterations}}
$$

就会和论文定义偏离。更合理的是：

$$
g_i =
\frac{\mathrm{grad\_accum}_i}
{\max(\mathrm{visible\_observations}_i, 1)}
$$

或者在 Pixel-GS 加权后直接用：

$$
g_i =
\frac{
\sum_k m_i^k h_i^k
}{
\sum_k m_i^k+\epsilon
}
$$

### 13.5 split 和 clone 必须分开，否则会把 floater 一起放大

AbsGS 更适合识别“表达能力不足的大 Gaussian”，也就是 split；它不一定适合 clone 小 Gaussian。如果把 AbsGS score 也用于 clone，会把边缘噪声、小 floater、高残差孤立点一起增殖。

建议策略：

- `split_score_abs_pixel` 只控制 split。
- 旧 `mean2d_grad` 或 color residual 控制 clone。
- prune 使用 opacity + visibility + age，不要只用 opacity。

### 13.6 近相机 floater 不能只靠 opacity 判断

很多 floater 训练后可能 opacity 不低，尤其在背景缺监督或 pose 有误时。只用：

$$
\alpha_i < \tau_\alpha
$$

会漏掉高 opacity floater。更稳的判断需要组合：

- 可见次数低；
- 多视图深度不一致；
- 投影覆盖大但 color residual 高；
- 长期不被多个视角支持；
- 位于相机 frustum 近处但远离 SfM / 初始化几何分布。

第一阶段可以先用 visibility window 和 depth scaling 抑制，后续再考虑 depth consistency pruning。

### 13.7 先排除数据和相机问题，否则 topology 改动会背锅

发糊不一定来自 densification。三维视觉里常见的根因还有：

- COLMAP 相机模型或畸变参数解析错误。
- 图像缩放后 intrinsics 没同步缩放。
- pose 坐标系方向或 handedness 有细小错误。
- 训练/评估 renderer 不一致。
- SH degree 保存/读取不一致导致颜色表达被截断。
- 输入本身存在运动模糊、rolling shutter 或曝光不一致。

因此在做论文方法前，必须先固定一个 sanity scene：

1. 用少量训练图 overfit 到接近清晰。
2. 关闭 topology，只验证投影和 backward 是否能降低 loss。
3. 用同一相机 render 训练视角，确认边缘对齐。
4. 再打开 topology 看是否改善。

### 13.8 Mip-Splatting / Analytic-Splatting 不应过早介入

抗混叠方法会改变 rasterization 的信号模型。它可能让图像更稳定，但也可能把 densification 不足掩盖成“滤波后看起来不抖”。如果当前主要问题是大 splat 没 split，那么先改 anti-aliasing 可能会让结果更平滑，反而不利于恢复细节。

建议判断标准：

- 训练视角都糊：优先 topology / data / pose。
- 训练视角清楚，测试视角缩放后糊或锯齿：再做 Mip / Analytic。

### 13.9 指标不能只看 PSNR

PSNR 对“更平滑”有时不敏感，甚至可能偏好平滑结果。高频细节恢复要同时看：

- LPIPS；
- SSIM；
- 固定 crop；
- Gaussian 分布可视化；
- 远景和近景分开统计；
- floater 数量。

最终判断应以固定 crop + LPIPS + floater 数为主，PSNR 只作为辅助。

### 13.10 当前最值得先做的修正

从专家视角，最重要的顺序是：

1. 确认当前 `grad_2d_accum` 是否真的是 projected mean2d gradient。
2. 如果不是，在 raster backward 中新增真正的 screen-space gradient accumulator。
3. 实现 AbsGS 的 per-pixel absolute accumulation。
4. 只用于 split 大 Gaussian。
5. 再加 Pixel-GS coverage weighting。
6. 最后加 depth scaling 抑制近相机 floater。

这个顺序能最大概率改善“糊”，同时避免把 floater 一起增殖。

### 13.11 当前代码审计结论

已对当前 RustGS backward 路径做初步审计，结论如下：

1. `RustGS/src/training/shaders/rasterize_backwards.wgsl` 中已经在 raster backward 阶段计算了 projected 2D mean 的局部梯度：

$$
\frac{\partial L}{\partial \mathbf{u}_i}
\approx
v_{\mathrm{xy}}
$$

对应 shader 里的 `v_xy`，它来自每个像素对 projected splat center 的贡献。

2. `RustGS/src/training/shaders/project_backwards.wgsl` 会把 `v_xy` 继续通过 projection VJP 传回 camera/world transform：

$$
\frac{\partial L}{\partial \mathbf{u}_i}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{c,i}}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{w,i}}
$$

3. 当前 `RustGS/src/training/engine/trainer.rs::accumulate_gradients` 统计的是：

```rust
transforms_grad[.., 0..3].abs().mean_dim(1)
```

也就是 post-projection transform gradient，不是 AbsGS / Pixel-GS 需要的 per-pixel screen-space mean gradient。

4. 因此，后续实现 AbsGS 不能复用当前 `grad_2d_accum` 直接取 abs。正确做法是在 `rasterize_backwards.wgsl` 中，在 per-pixel contribution 写入 `v_splats` 前，同时维护新的 accumulator：

$$
\sum_p
\left|
\frac{\partial L_p}{\partial u_{i,x}}
\right|,
\quad
\sum_p
\left|
\frac{\partial L_p}{\partial u_{i,y}}
\right|
$$

这对应 Epic 2 / Epic 3 的开发内容。

## 14. 已完成开发进度

### 2026-04-27

已完成 Story 1.1 / Story 1.2 / Story 1.3 / Story 2.2 / Story 3.1 / Story 3.2 / Story 3.3 / Story 4.1 / Story 4.2 / Story 6.1 的第一版代码，并完成 Story 5.1 的实验开关版本。

Story 1.1 / Story 1.3：

- 新增 `ParityFloatDistribution` 和 `ParityTopologyStepSample`。
- `ParityTopologyMetrics` 新增 `topology_step_samples`，并通过 `serde(default)` 兼容旧 report。
- 每次 topology planning 后记录：
  - iteration / epoch / Gaussian count。
  - clone / split / prune / growth candidate 数。
  - `mean2d_grad`、`max_scale`、`opacity` 的 min / p10 / p50 / p90 / p99 / max / mean。
  - large splat 数。
  - large-low-grad 数和比例。
- 训练日志新增一行 topology diagnostics。
- 代码注释明确当前 `grad_2d_accum` 不是 AbsGS 所需的 per-pixel screen-space gradient。

Story 1.2：

- post-training evaluation 新增固定 crop 输出：
  - `--eval-crop-output-dir`
  - `--eval-crop-frames`
  - `--eval-crop-rect`
- 每个 frame 输出三张图：
  - target crop
  - rendered crop
  - `diff_x4` crop
- `SplatEvaluationSummary` 新增 `crop_outputs`，`--eval-json` 会记录 crop 路径。

Story 2.2 / Story 3.1：

- `rasterize_backwards.wgsl` 新增 compact visible splat 的 screen-space gradient 统计 buffer。
- 每个 visible splat 记录 `[signed_x, signed_y, abs_x, abs_y]`。
- `abs_x / abs_y` 是在 raster backward 的 per-pixel `v_xy` 贡献写入前取绝对值后 atomic 累加，不是对已经聚合后的 `transforms_grad` 再取绝对值。
- `project_backwards.wgsl` 把 compact visible buffer scatter 回全局 splat buffer，输出 `[num_splats, 7]`。
- `RenderSplatsOutput` 新增 `screen_grad_stats`，通过 Burn Autodiff 的 gradient sink 在 `loss.backward()` 后取回该统计。
- `WgpuTrainer` 新增：
  - `screen_grad_2d_accum`
  - `abs_grad_2d_accum`
- topology snapshot / host stats 中带出：
  - `screen_mean2d_grad`
  - `abs_mean2d_grad`
  - `split_score`

Story 3.2 / Story 3.3：

- 新增 `LiteGsSplitScoreMode`：
  - `baseline`
  - `abs`
  - `abs-pixel`
- 新增 CLI 参数：
  - `--litegs-split-score baseline|abs|abs-pixel`
  - `--litegs-split-grad-threshold`
- 默认仍为 `baseline`，默认训练行为保持旧逻辑。
- `abs` / `abs-pixel` 模式保留 baseline 的大 Gaussian split 判断，同时用 AbsGS / Pixel-GS score 补充捕捉梯度抵消的大 Gaussian；小 Gaussian 的 clone candidate 仍使用旧 `mean2d_grad`，避免把 near-camera floater 或孤立噪声一起放大。
- topology telemetry 新增：
  - `screen_mean2d_grad`
  - `abs_mean2d_grad`
  - `abs_pixel_mean2d_grad`
  - `pixel_coverage`
  - `split_score`

Story 4.1 / Story 4.2：

- 在 `rasterize_backwards.wgsl` 中直接统计有效贡献像素数。每个通过 `sigma >= 0` 且 `alpha >= 1/255` 的 per-pixel contribution 会对 `pixel_coverage` atomic 加 1。
- 这比最初计划的 projected covariance coverage 近似更贴近 Pixel-GS 的有效像素数定义。
- 新增 accumulators：
  - `abs_pixel_grad_2d_accum = \sum_k \tilde{m}_i^k h_i^k`
  - `pixel_coverage_accum = \sum_k \tilde{m}_i^k`
- 新增 split score：

$$
s_i^{\mathrm{abs\_pixel}}
=
\frac{
\sum_k \tilde{m}_i^k h_i^k
}{
\sum_k \tilde{m}_i^k+\epsilon
}
$$

- `abs-pixel` 模式仍然只控制大 Gaussian split，小 Gaussian clone 继续使用 baseline `mean2d_grad`。

Story 5.1：

- 新增 `abs-pixel-depth` 模式，用 Pixel-GS 的 depth-scaled score 抑制 near-camera growth：

$$
f(i,k)
=
\mathrm{clip}
\left(
\left(
\frac{z_i^k}
{\gamma_{\mathrm{depth}}R_{\mathrm{scene}}}
\right)^2,
0,
1
\right)
$$

$$
s_i^{\mathrm{abs\_pixel\_depth}}
=
s_i^{\mathrm{abs\_pixel}} f_i
$$

- 新增 CLI 参数：
  - `--litegs-split-score abs-pixel-depth`
  - `--litegs-depth-scale-gamma`
- `project_backwards.wgsl` 额外记录 `pixel_coverage * camera_depth`，host 端得到 `camera_depth_mean`。
- topology telemetry 新增：
  - `camera_depth`
  - `depth_scale`

Story 6.1：

- 新增 `--litegs-profile`：
  - `baseline`
  - `abs-split`
  - `abs-pixel`
  - `abs-pixel-depth`
- `abs-pixel` profile 固化当前推荐实验参数：
  - `split_score_mode = abs-pixel`
  - `split_grad_threshold = 0.00001`
- profile 会写入 `LiteGsConfig.training_profile`，因此 parity report 可复现实验来源。

验证：

- `cargo test -p rustgs --all-targets` 通过。
- `cargo test -p rustgs --no-default-features --all-targets` 通过；已有 no-default warning 未在本次处理。
- targeted tests：
  - `abs_split_score_only_marks_large_split_candidates`
  - `abs_pixel_split_score_uses_coverage_weighted_score`
  - `test_rasterize_bwd_kernel_writes_output_buffer`
  - `test_render_splats_autodiff`

当前已知风险 / 后续确认：

- `abs` 模式的 `--litegs-split-grad-threshold` 还没有真实场景标定；默认值等于旧 threshold，只适合 smoke test，不应直接当最终推荐值。
- 当前 per-pixel `v_xy` 统计遵循 raster backward 的实际参数梯度路径；当 `color.a * gaussian > 0.999` 触发饱和分支时，该像素不会贡献 AbsGS 统计。这个行为和当前参数梯度一致，但后续可以对比“饱和前统计”的效果。
- `abs_mean2d_grad` 当前按 visible observation 归一化，而旧 `mean2d_grad` 按 total observation 归一化；这是为了避免不可见视角稀释 split score，但需要通过 ablation 验证阈值尺度。

下一步：

- 把 `abs-pixel` 作为当前推荐实验分支，优先用 `--litegs-profile abs-pixel` 做更长 10k / 30k 检查。
- Story 4.3：继续评估当前 per-pixel coverage 统计的训练开销和质量收益，必要时再做近似版作为性能回退。
- `abs-pixel-depth` 暂时保留为 floater 抑制实验开关；当前 TUM 短跑会压低 PSNR，不能作为默认。

## 15. Epic / Story 拆分

本节把 `## 12. 接下来要开发的内容` 拆成可排期的 epic 和 story。当前没有单独 PRD / Architecture 文档，因此这里以本文的开发方案和 RustGS 当前代码结构作为需求来源。

### Epic 1：建立训练质量诊断基线

目标：在不改变训练行为的前提下，把“糊”和“杂散 splat”的类型量化出来，避免后续优化靠主观判断。

#### Story 1.1：记录 topology 分布和训练过程统计

用户故事：作为 RustGS 开发者，我需要在每次 topology step 后看到 Gaussian 数量、梯度、scale、opacity、split/clone/prune 的分布，这样才能判断训练发糊是否来自 under-split 或错误增长。

范围：

- 在 telemetry/report 中记录：
  - Gaussian 总数。
  - split / clone / prune 数。
  - `mean2d_grad` 的 min / p50 / p90 / p99 / max。
  - `max_scale` 的 p50 / p90 / p99 / max。
  - opacity 的 p10 / p50 / p90。
  - prune candidates 和 growth candidates。
- 不改现有训练策略。

验收标准：

- baseline 训练结果和改动前保持一致。
- 每次 topology step 都能在 report 或日志中看到上述统计。
- `cargo test -p rustgs --all-targets` 通过。

涉及文件：

- `RustGS/src/training/metrics.rs`
- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/bin/rustgs/train_command.rs`
- `RustGS/src/training/topology/mod.rs`

优先级：P0  
估算：M  
依赖：无

#### Story 1.2：固定 crop 输出用于主观质量对比

用户故事：作为视觉算法调试者，我需要固定训练视角/测试视角 crop 输出，这样才能可靠比较高频纹理、远景和近相机 floater 的变化。

范围：

- 支持配置固定 camera index 和 crop rectangle。
- 每 N 次或每个评估点保存：
  - full render。
  - fixed crop render。
  - 文件名包含 iteration、Gaussian count、profile。
- 至少覆盖三类区域：
  - 高频纹理。
  - 远景背景。
  - 近相机区域。

验收标准：

- 同一场景、同一配置多次运行输出路径稳定。
- crop 图能和 PSNR/SSIM/LPIPS 结果一起归档。
- 不影响默认训练路径。

涉及文件：

- `RustGS/src/training/evaluation/*`
- `RustGS/src/bin/rustgs/train_command.rs`
- `RustGS/experiments/*.md`

优先级：P0  
估算：M  
依赖：Story 1.1

#### Story 1.3：增加 over-reconstruction 风险指标

用户故事：作为三维视觉开发者，我需要知道有多少大 Gaussian 处于低梯度状态，这样才能确认 AbsGS 是否是正确方向。

范围：

- 增加指标：

$$
R_{\mathrm{large\_lowgrad}}
=
\frac{
\#\{i \mid \max(\mathbf{s}_i)>\tau_{\mathrm{scale}}
\land g_i < \tau_{\mathrm{grad}}\}
}{
\#\{i \mid \max(\mathbf{s}_i)>\tau_{\mathrm{scale}}\}+\epsilon
}
$$

- 在 report 中输出该指标。
- 记录对应的 large splat 数量。

验收标准：

- 指标在 baseline 中可读。
- 指标不参与训练，只用于诊断。
- 能在至少一个发糊场景中看到该指标与主观糊区域相关。

涉及文件：

- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/training/metrics.rs`

优先级：P0  
估算：S  
依赖：Story 1.1

### Epic 2：确认并补齐 screen-space gradient 统计

目标：确保后续 AbsGS/Pixel-GS 使用的是论文定义需要的 projected mean2d gradient，而不是已经混入 3D transform Jacobian 的聚合梯度。

#### Story 2.1：审计当前梯度路径

用户故事：作为算法实现者，我需要确认当前 `grad_2d_accum` 的数学含义，这样才能判断它是否可用于 AbsGS/Pixel-GS。

范围：

- 从 `rasterize_bwd` 到 `project_bwd` 画清楚梯度路径：

$$
\frac{\partial L}{\partial \mathbf{u}_i}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{c,i}}
\rightarrow
\frac{\partial L}{\partial \boldsymbol{\mu}_{w,i}}
$$

- 明确当前 `transforms_grad[.., 0..3]` 对应哪个空间。
- 在文档中写出结论。

验收标准：

- 有明确结论：当前 accumulator 是否能作为 mean2d gradient。
- 如果不能，列出必须从 shader 新增的 buffer。
- 不改训练行为。

涉及文件：

- `RustGS/src/training/backward/rasterize_bwd.rs`
- `RustGS/src/training/backward/project_bwd.rs`
- `RustGS/src/training/shaders/rasterize_backwards.wgsl`
- `RustGS/src/training/shaders/project_backwards.wgsl`

优先级：P0  
估算：M  
依赖：Epic 1 可并行

#### Story 2.2：新增 true screen-space mean gradient accumulator

用户故事：作为 3DGS 训练系统，我需要在 raster backward 阶段记录每个 splat 的 screen-space mean gradient，这样 densification 才能按论文定义工作。

范围：

- 在 raster backward 中新增 per-splat accumulator：
  - signed screen-space x/y gradient。
  - 可选 magnitude accumulator。
- Rust 侧新增 readback tensor。
- topology snapshot 中带出该统计。

验收标准：

- 新增 accumulator 在关闭时不影响 baseline。
- 打开后 tensor shape 和 splat 数量一致。
- 空 visible splat、无贡献 splat 不产生 NaN。
- 有单元测试或 smoke test 覆盖 readback 和 reset。

涉及文件：

- `RustGS/src/training/backward/rasterize_bwd.rs`
- `RustGS/src/training/shaders/rasterize_backwards.wgsl`
- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/topology/bridge.rs`

优先级：P0  
估算：L  
依赖：Story 2.1

### Epic 3：实现 AbsGS split-only densification

目标：让高频区域的大 Gaussian 更容易被 split，直接改善 over-reconstruction 导致的糊。

#### Story 3.1：实现 per-pixel absolute gradient accumulator

用户故事：作为 densification 策略，我需要在像素梯度贡献累加前取绝对值，这样才能避免高频纹理区域的梯度抵消。

范围：

- 在 raster backward 中实现：

$$
A^k_{i,x}
=
\sum_{p \in P_i^k}
\left|
\frac{\partial L^k_p}{\partial \mu^k_{i,x}}
\right|,
\quad
A^k_{i,y}
=
\sum_{p \in P_i^k}
\left|
\frac{\partial L^k_p}{\partial \mu^k_{i,y}}
\right|
$$

- 累计 scalar score：

$$
h_i^k =
\sqrt{(A^k_{i,x})^2+(A^k_{i,y})^2}
$$

- 增加 `abs_grad_2d_accum`。

验收标准：

- 不是对聚合后的 gradient 做 `abs()`。
- 有测试或调试日志证明 `abs_grad >= |signed_grad|`。
- 默认关闭时 baseline 不变。

涉及文件：

- `RustGS/src/training/shaders/rasterize_backwards.wgsl`
- `RustGS/src/training/backward/rasterize_bwd.rs`
- `RustGS/src/training/engine/trainer.rs`

优先级：P0  
估算：L  
依赖：Story 2.2

#### Story 3.2：在 topology 中拆分 split_candidate 和 clone_candidate

用户故事：作为 topology planner，我需要把 split 和 clone 的判断分开，这样 AbsGS 只会拆大 Gaussian，不会放大小 floater。

范围：

- `TopologyCandidateInfo` 增加：
  - `split_score`
  - `split_candidate`
  - `clone_candidate`
- 大 scale 使用 AbsGS score：

$$
\mathrm{split}(i)
\iff
s_i^{\mathrm{abs}} > \tau_{\mathrm{split}}
\land
\max(\mathbf{s}_i) > \tau_{\mathrm{scale}}
\land
\alpha_i > \tau_{\alpha,\mathrm{min}}
$$

- 小 scale 保持旧梯度 clone 逻辑：

$$
\mathrm{clone}(i)
\iff
g_i^{\mathrm{old}} > \tau_{\mathrm{clone}}
\land
\max(\mathbf{s}_i) \le \tau_{\mathrm{scale}}
$$

验收标准：

- split 和 clone 数量在 telemetry 中分开。
- AbsGS score 不用于 clone。
- 现有 topology tests 更新并通过。

涉及文件：

- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/training/topology/apply.rs`
- `RustGS/src/training/topology/bridge.rs`

优先级：P0  
估算：M  
依赖：Story 3.1

#### Story 3.3：增加 AbsGS 配置开关和阈值

用户故事：作为训练用户，我需要用 CLI/profile 控制 AbsGS 策略，这样可以做可复现 ablation。

范围：

- 新增配置：
  - `--litegs-split-score baseline|abs`
  - `--litegs-split-grad-threshold`
  - `--litegs-clone-grad-threshold` 或复用旧 threshold。
- report 记录配置值。

验收标准：

- 默认配置保持当前行为。
- `abs` 模式下只改变 split 选择。
- report 中能区分 baseline 和 abs 实验。

涉及文件：

- `RustGS/src/bin/rustgs/train_command.rs`
- `RustGS/src/training/config` 或 `RustGS/src/training/mod.rs`
- `RustGS/src/training/topology/mod.rs`

优先级：P0  
估算：M  
依赖：Story 3.2

### Epic 4：实现 Pixel-GS coverage-weighted split score

目标：减少多视角平均对大 Gaussian 的梯度稀释，让稀疏点云区域更容易长出细节。

#### Story 4.1：实现 per-pixel coverage 统计

用户故事：作为 Pixel-GS densification 策略，我需要知道每个 splat 在当前视角的有效贡献像素数，这样可以按贡献面积加权梯度。

范围：

- 在 raster backward 中对有效 contribution 做 atomic 统计：

$$
\alpha_i^{k,pix} \ge \frac{1}{255}
$$

- 累计 `coverage_accum`。
- 不直接使用 tile intersection count 作为 pixel coverage。

验收标准：

- coverage 值非负、有限。
- 大 projected splat 的 coverage 大于小 projected splat。
- report 输出 coverage 分布。

涉及文件：

- `RustGS/src/training/forward/projection.rs`
- `RustGS/src/training/shaders/project_forward.wgsl`
- `RustGS/src/training/engine/trainer.rs`

优先级：P1  
估算：M  
依赖：Epic 3

#### Story 4.2：实现 coverage-weighted AbsGS split score

用户故事：作为 topology planner，我需要把 AbsGS split score 按覆盖像素加权，这样真正覆盖大面积图像的 splat 更容易被 split。

范围：

- 实现：

$$
s_i^{\mathrm{abs\_pixel}}
=
\frac{
\sum_k \tilde{m}_i^k h_i^k
}{
\sum_k \tilde{m}_i^k+\epsilon
}
$$

- 增加 mode：
  - `baseline`
  - `abs`
  - `abs-pixel`

验收标准：

- `abs-pixel` 模式下 split 候选和 `abs` 模式可对比。
- Gaussian 总数不超过 budget。
- 高频 crop 不比 `abs` 退化。

涉及文件：

- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/training/topology/bridge.rs`
- `RustGS/src/bin/rustgs/train_command.rs`

优先级：P1  
估算：M  
依赖：Story 4.1

#### Story 4.3：评估是否需要精确 per-pixel coverage

用户故事：作为算法负责人，我需要判断当前 per-pixel coverage 的质量收益是否值得它的 atomic 统计成本，否则再决定是否引入 projected coverage 近似作为性能回退。

范围：

- 对比 `abs` 与 `abs-pixel`：
  - split 分布。
  - 高频 crop。
  - floater 数。
  - 训练时间。
- 如果 per-pixel 统计成本过高，再设计 projected coverage 近似版：

$$
\tilde{m}_i^k = \pi r_{x,i}^k r_{y,i}^k
$$

验收标准：

- 有实验结论：per-pixel coverage 是否足够进入默认 profile。
- 不在本 story 中实现近似回退，除非实验明确需要。

优先级：P1  
估算：S  
依赖：Story 4.2

### Epic 5：抑制近相机 floater 和错误增长

目标：在增强 split 能力的同时，避免近相机杂散 splat 被一起放大。

#### Story 5.1：实现 Pixel-GS depth-scaled gradient

用户故事：作为训练系统，我需要降低近相机 splat 的 densification score，这样可以抑制近处 floater 过早增殖。

范围：

- 计算 scene camera radius：

$$
R_{\mathrm{scene}}
=
1.1
\max_i
\left\|
\mathbf{C}_i
-
\frac{1}{N}\sum_j\mathbf{C}_j
\right\|_2
$$

- 实现：

$$
f(i,k)
=
\mathrm{clip}
\left(
\left(
\frac{z_i^k}
{\gamma_{\mathrm{depth}}R_{\mathrm{scene}}}
\right)^2,
0,
1
\right)
$$

- 应用到 split score 分子：

$$
s_i^{\mathrm{final}}
=
\frac{
\sum_k \tilde{m}_i^k f(i,k)h_i^k
}{
\sum_k \tilde{m}_i^k+\epsilon
}
$$

验收标准：

- 新增 `abs-pixel-depth` 模式。
- 近相机低可信 splat 数下降或不增加。
- 远景高频 split 不明显下降。

涉及文件：

- `RustGS/src/training/engine/trainer.rs`
- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/bin/rustgs/train_command.rs`

优先级：P1  
估算：M  
依赖：Epic 4

#### Story 5.2：完善 visibility / age prune 保护

用户故事：作为 topology 系统，我需要结合 visibility 和 age 做 prune，这样不会只靠 opacity 漏掉高 opacity floater。

范围：

- report 中记录：
  - invisible windows 分布。
  - low visibility splat 数。
  - near low visibility splat 数。
- prune 条件保持保守，不误删新生 splat。
- 明确不同 prune mode 的行为。

验收标准：

- 新生 splat 不会立即被 prune。
- 长期不可见或低支持 splat 能进入候选。
- 现有 prune tests 通过。

涉及文件：

- `RustGS/src/training/topology/mod.rs`
- `RustGS/src/training/engine/trainer.rs`

优先级：P1  
估算：M  
依赖：Story 5.1 可并行

#### Story 5.3：可选动态 split 阈值

用户故事：作为训练系统，我需要在早期或近相机区域提高 split 阈值，这样可以减少模糊/不稳定阶段的错误增长。

范围：

- 实现可关闭的动态阈值：

$$
\tau_i(t)
=
\tau_0
\left(
1+\alpha e^{-\beta z_i}
\right)
\rho(t)
$$

- 默认关闭。
- 只在 `abs-pixel-depth` 仍有明显 floater 时启用实验。

验收标准：

- 默认行为不变。
- 打开后 early-stage split 数可控下降。
- 不明显降低最终细节。

优先级：P2  
估算：M  
依赖：Story 5.1

### Epic 6：固化实验 profile 和评估报告

目标：把算法改动变成可复现实验，而不是一次性调参。

#### Story 6.1：新增训练 profile

用户故事：作为实验用户，我需要通过 profile 快速切换 topology 策略，这样可以稳定复现实验。

范围：

- 新增 profile：
  - `baseline`
  - `abs-split`
  - `abs-pixel`
  - `abs-pixel-depth`
- profile 写入 report。

验收标准：

- 每个 profile 的关键参数完整记录。
- 同一命令可复现实验配置。

优先级：P1  
估算：S  
依赖：Epic 3

#### Story 6.2：建立 ablation 实验文档

用户故事：作为项目维护者，我需要一份固定 ablation 表，这样可以判断每个算法组件的实际贡献。

范围：

- 建立实验表：
  - baseline
  - AbsGS-score
  - AbsGS + coverage
  - AbsGS + coverage + depth
  - dynamic threshold 可选
- 每组记录：
  - PSNR / SSIM / LPIPS。
  - Gaussian count。
  - split / clone / prune。
  - 训练时间。
  - crop 图片路径。

验收标准：

- 至少一个发糊场景完成全表。
- 结果能说明是否继续推进默认配置。

涉及文件：

- `RustGS/experiments/*.md`

优先级：P1  
估算：M  
依赖：Epic 5

#### Story 6.3：确定默认策略和回滚开关

用户故事：作为 RustGS 使用者，我需要一个默认稳定的训练策略，同时保留回滚到 baseline 的开关。

范围：

- 根据 ablation 确定默认值。
- 保留 `baseline` mode。
- README 或实验文档中说明推荐配置。

验收标准：

- 默认策略在目标场景上清晰度改善。
- floater 不增加。
- 用户可以一键切回旧逻辑。

优先级：P1  
估算：S  
依赖：Story 6.2

### Epic 7：后续 anti-aliasing 分支

目标：在 topology 改动稳定后，单独处理多尺度渲染/采样率导致的糊和锯齿。

#### Story 7.1：Mip-Splatting 可行性原型

用户故事：作为 renderer 开发者，我需要一个 Mip filter 原型，这样可以评估多尺度渲染下是否减少 aliasing 且不过度平滑。

范围：

- 增加 `render_filter=baseline|mip`。
- 实现 2D Mip filter 原型：

$$
G_k^{2D}(\mathbf{x})_{\mathrm{mip}}
=
\sqrt{
\frac{
|\boldsymbol{\Sigma}_k^{2D}|
}{
|\boldsymbol{\Sigma}_k^{2D}+s\mathbf{I}|
}
}
\exp
\left(
-\frac{1}{2}
(\mathbf{x}-\mathbf{p}_k)^T
(\boldsymbol{\Sigma}_k^{2D}+s\mathbf{I})^{-1}
(\mathbf{x}-\mathbf{p}_k)
\right)
$$

验收标准：

- 单独分支评估，不混入 topology 实验。
- 多尺度测试 aliasing 降低。
- 正常尺度不过度变糊。

优先级：P2  
估算：L  
依赖：Epic 6

#### Story 7.2：Analytic-Splatting 研究 Spike

用户故事：作为研究实现者，我需要评估 Analytic-Splatting 的 shader 成本和质量收益，这样决定是否值得替换 Mip filter。

范围：

- 只做技术 Spike，不直接进默认训练。
- 评估 logistic CDF 近似和 eigen decomposition 成本。

验收标准：

- 有明确结论：继续实现、放弃、或仅作为高质量离线 renderer。

优先级：P3  
估算：M  
依赖：Story 7.1

### Epic 8：条件性 Deblur / SR 方法

目标：只在数据条件明确需要时，引入去运动模糊或超分监督。

#### Story 8.1：输入运动模糊检测和决策记录

用户故事：作为数据处理者，我需要判断输入是否真实运动模糊，这样不会把 Deblur-GS/BSGS 错用于普通 densification 问题。

范围：

- 建立检查清单：
  - 原图是否有拖影。
  - COLMAP pose 是否不稳定。
  - 多视角边缘是否系统性拉伸。
- 记录是否进入 Deblur 分支。

验收标准：

- 没有证据时不实现 Deblur。
- 有证据时再单独开实现计划。

优先级：P3  
估算：S  
依赖：Epic 6

#### Story 8.2：低分辨率训练数据的 SRGS 决策 Spike

用户故事：作为训练系统，我需要判断当前数据是否低分辨率受限，这样再决定是否引入 SRGS 的 pseudo-HR 监督。

范围：

- 检查训练图分辨率和目标输出分辨率。
- 如果确实是 LR 训练，设计 pseudo-HR 生成和 render-downsample consistency。

验收标准：

- 普通清晰度问题不进入 SRGS。
- 低分辨率问题有单独方案。

优先级：P3  
估算：S  
依赖：Epic 6

### 推荐排期顺序

| 顺序 | Epic / Story | 原因 |
|---:|---|---|
| 1 | Story 1.1, 1.2, 1.3 | 先建立可观测性，避免盲改 |
| 2 | Story 2.1, 2.2 | 确认梯度语义，否则 AbsGS 可能实现错 |
| 3 | Story 3.1, 3.2, 3.3 | 最直接改善高频糊 |
| 4 | Story 4.1, 4.2, 4.3 | 解决大 splat 多视角平均稀释 |
| 5 | Story 5.1, 5.2 | 抑制 near-camera floater |
| 6 | Story 6.1, 6.2, 6.3 | 固化 profile 和实验结论 |
| 7 | Epic 7 | topology 稳定后再做 anti-aliasing |
| 8 | Epic 8 | 只有特定数据条件触发 |

最小可交付 MVP：

- Story 1.1：已完成第一版
- Story 1.2：已完成第一版
- Story 2.1：已完成文档审计
- Story 2.2：已完成第一版
- Story 3.1：已完成第一版
- Story 3.2：已完成第一版
- Story 3.3：已完成第一版
- Story 4.1：已完成第一版
- Story 4.2：已完成第一版
- Story 5.1：已完成实验开关版本
- Story 6.1：已完成第一版

MVP 完成后，就能回答一个关键问题：RustGS 当前的糊，是否主要来自大 Gaussian 没有被正确 split。

## 16. TUM Freiburg1 XYZ 测试记录

数据集：

```text
/Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap
```

数据集信息：

- COLMAP 输入。
- 798 张训练图。
- 分辨率 `640x480`。
- 初始化点数 `8986`。

统一训练设置：

```sh
cargo run --release -p rustgs --bin rustgs -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --iterations 3000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --eval-after-train --eval-json --log-level info
```

输出目录：

```text
RustGS/output/experiments/tum_absgs_20260427/
```

结果：

| 配置 | split score | split threshold | 最终 splats | PSNR mean | PSNR min | densify added | prune removed | first split candidates | last split candidates | last large-low-grad ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_3000` | baseline | `0.00014` | 69437 | 21.2837 | 14.8517 | 60463 | 12 | 160 | 293 | 0.7244 |
| `abs_3000_t0_00014` | abs | `0.00014` | 66836 | 21.0872 | 14.8479 | 57853 | 3 | 4 | 9 | 0.3881 |
| `abs_3000_t0_00001` | abs | `0.00001` | 67805 | 21.0991 | 14.4824 | 58822 | 3 | 149 | 350 | 0.4313 |
| `abs_pixel_3000_t0_00001` | abs-pixel | `0.00001` | 68331 | 21.1867 | 15.1002 | 59348 | 3 | 164 | 466 | 0.4642 |
| `abs_pixel_3000_t0_00002` | abs-pixel | `0.00002` | 67892 | 21.2059 | 14.5895 | 58908 | 2 | 142 | 276 | 0.4256 |
| `abs_pixel_aug_3000_t0_00001` | abs-pixel + baseline split | `0.00001` | 70286 | 21.4642 | 15.7025 | 61306 | 6 | - | 731 | 0.7508 |
| `abs_pixel_aug_3000_t0_00002` | abs-pixel + baseline split | `0.00002` | 69981 | 21.4455 | 15.5897 | 61000 | 5 | - | 459 | 0.7363 |
| `abs_pixel_depth_3000_t0_00001_g0_37` | abs-pixel-depth | `0.00001` | 68598 | 21.2464 | 15.1293 | 59615 | 3 | - | 609 | 0.4920 |
| `abs_pixel_depth_3000_t0_00001_g0_20` | abs-pixel-depth | `0.00001` | 68400 | 21.2532 | 15.0004 | 59414 | 0 | - | 585 | 0.5405 |

关键观察：

- 新增的 AbsGS gradient 统计链路可以正常跑完训练、导出 PLY、生成 parity report，没有 NaN / OOM。
- `--litegs-split-score abs --litegs-split-grad-threshold 0.00014` 阈值明显过高。第一次 topology 中：
  - `abs_mean2d_grad p50 = 9.25e-6`
  - `abs_mean2d_grad p90 = 4.13e-5`
  - `abs_mean2d_grad p99 = 1.03e-4`
  - 因此 `1.4e-4` 高于 p99，只产生极少 split candidates。
- `1e-5` 阈值能把 split candidate 数拉回到接近 baseline 的量级，但 3000-step 的 6-view PSNR 仍低于 baseline。
- `abs` 模式的 large-low-grad ratio 明显低于 baseline，说明它确实改变了大 Gaussian 的 split 分布；但短跑 PSNR 没有收益，不能把当前 AbsGS 配置设为默认。
- `abs-pixel` 比 `abs` 更有信号：
  - `1e-5` 的 mean PSNR 从 abs-only 的 `21.0991` 提高到 `21.1867`。
  - `1e-5` 的 worst PSNR 从 baseline 的 `14.8517` 提高到 `15.1002`。
  - `2e-5` 的 mean PSNR 提高到 `21.2059`，但 worst PSNR 降到 `14.5895`。
- 直接用 `abs-pixel` 替换 baseline split 仍然没有超过 baseline mean PSNR，但它改善了 worst frame，说明 coverage weighting 对动态/困难视角可能有价值。
- 将 `abs-pixel` 改为 **保留 baseline 大 Gaussian split + 额外补充 AbsPixel split** 后，短跑结果明显改善：
  - `1e-5` 的 mean PSNR 从 baseline `21.2837` 提升到 `21.4642`。
  - worst PSNR 从 baseline `14.8517` 提升到 `15.7025`。
  - splat 数从 `69437` 增到 `70286`，增长幅度可控。
- `abs-pixel-depth` 的当前实现能压低 large-low-grad ratio，但在 TUM Freiburg1 XYZ 的 3000-step 短跑中 PSNR 低于 augmented abs-pixel，说明默认 `gamma_depth` 还不能直接启用为推荐配置。

当前结论：

1. 第一版 AbsGS / Pixel-GS coverage 实现路径是可运行的。
2. 当前 `abs` score 的尺度和 baseline `mean2d_grad` 不同，不能复用 baseline 阈值。
3. 单纯替换 AbsGS split score 没有改善短跑 PSNR；必须保留 baseline split 作为下限，再用 AbsPixel 捕捉额外的大 Gaussian under-split。
4. 当前推荐实验配置是 `--litegs-split-score abs-pixel --litegs-split-grad-threshold 0.00001`，但默认策略仍应保持 `baseline`，直到 10k / 30k 和主观 crop 检查通过。
5. `abs-pixel-depth` 已有开关和 telemetry，但当前参数会压低 PSNR，应继续作为 floater 抑制实验，而不是默认质量提升方案。

建议下一轮实验：

| 实验 | 参数 | 目的 |
|---|---|---|
| Abs-pixel threshold sweep | `8e-6`, `1e-5`, `1.5e-5`, `2e-5` | 在 augmented split 策略下找 mean PSNR / splat count 折中 |
| Fixed crop review | frame 0 / 90 / 120 | 看高频细节和 worst frame blur 是否有主观改善 |
| Full 10k check | 最优 threshold | 验证短跑结论是否随训练拉长反转 |
| Pixel-GS depth scale | `abs-pixel-depth`, sweep `gamma=0.10..0.37` | 只在 floater 明显时启用，避免过度压制正常 split |

### 16.1 2026-04-27 后续锐化实验记录

本轮新增开发内容：

- 训练时新增 `--raster-cov-blur`，默认仍为 `0.3`，保持原训练行为不变。
- 训练后评估新增 `--eval-raster-cov-blur`，默认继承 `--raster-cov-blur`，允许“稳定训练 + 锐化评估/导出”分离。
- `render` 命令新增 `--raster-cov-blur`，允许已训练 PLY 在导出时选择更锐的 raster covariance floor。
- 评估指标新增 `sharpness_grad_ratio_mean` 和 `sharpness_lap_ratio_mean`，用于量化渲染图相对 GT 的边缘/高频强度。
- 评估 crop 导出支持固定 frame，例如 `--eval-crop-frames 0,90,120`，并额外导出 target/render/diff 横向拼图 `*_strip.png`。

关键实验结果：

| 配置 | 训练 blur | 评估 blur | iter | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline old | `0.3` | `0.3` | 3000 | 21.2837 | 14.8517 | 0.7900 | 0.3762 | 原始基线偏糊 |
| gradient loss | `0.3` | `0.3` | 3000 | 21.3516 | 14.9610 | 0.7860 | 0.3713 | 不改善锐度，不推荐 |
| direct low blur | `0.2` | `0.2` | 3000 | 21.4014 | 15.8447 | 0.8285 | 0.4648 | 短跑看起来有收益 |
| direct low blur | `0.2` | `0.2` | 10000 | 16.7333 | 8.5736 | 0.6761 | 0.3825 | 长跑崩塌，frame 30/120 明显发绿/糊掉，不推荐 |
| stable train + sharp render | `0.3` | `0.2` | 10000 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | 当前最稳妥推荐 |
| stable train + sharp render | `0.3` | `0.2` | 30000 | 21.7628 | 16.2852 | 0.8567 | 0.5489 | 更锐但 PSNR 略降，不优于 10k |
| stable train + sharper render | `0.3` | `0.15` | 10000 | 21.8116 | 16.3536 | 0.8852 | 0.6349 | 更锐，PSNR 小降，可作为主观导出选项 |
| stable train + sharper render | `0.3` | `0.15` | 30000 | 21.7388 | 16.2459 | 0.8941 | 0.6555 | 更锐但 PSNR 继续下降 |
| stable train + aggressive render | `0.3` | `0.10` | 30000 | 21.6563 | 16.1885 | 0.9394 | 0.8012 | 过度锐化风险更高，不建议默认 |

当前工程结论：

1. **不要把 `raster_cov_blur=0.2` 直接用于训练默认值**。它的 3000-step 指标是正向的，但 10000-step 出现明显退化，属于短跑假阳性。
2. **当前最稳妥的去糊方案是稳定训练、锐化渲染**：训练保持 `--raster-cov-blur 0.3`，评估/导出使用 `--eval-raster-cov-blur 0.2` 或 `render --raster-cov-blur 0.2`。
3. `0.15` 可以进一步提高高频指标，但会牺牲 PSNR；适合作为人工 review 的锐化档，不适合作为默认。
4. 30k 训练没有解决 frame 90 的主要问题。该帧最差项更像几何/遮挡或相机覆盖问题，而不是单纯 raster blur；继续加迭代不会自动消除这种 smear。
5. 目前默认训练策略应保持 baseline split + `raster_cov_blur=0.3`，把 AbsPixel、gradient loss、低 blur 训练都保留为实验开关。

当前推荐命令：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_sharpness_20260427/baseline_10000_eval_blur_0_2.ply \
  --iterations 10000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --eval-after-train \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_sharpness_20260427/crops_baseline_10000_eval_blur_0_2 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

后续开发重点：

| 优先级 | 内容 | 原因 |
|---:|---|---|
| P0 | 将 render/eval blur preset 写入使用文档和训练输出记录 | 当前已验证有效，风险低 |
| P1 | 增加固定 crop 的 side-by-side strip 输出 | 已开发，后续用它做主观质量对比 |
| P1 | 针对 frame 90 做相机/遮挡误差诊断 | 最差帧不是单纯糊，需要看 pose、遮挡和动态物体 |
| P2 | 只在 topology freeze 后测试 blur schedule | 直接低 blur 训练已失败，schedule 必须作为实验开关 |
| P2 | 继续做 AbsPixel 10k/30k threshold sweep | 3000-step 有短期信号，但 10000-step 曾反转，必须长跑验证 |

### 16.2 2026-04-28 鲁棒残差损失实验记录

本方向目标：验证 worst frame 是否主要由动态物体、遮挡变化或少量异常像素驱动。如果是，可以通过降低大残差像素的损失权重减少错误 splat 增殖，同时提升训练效率。

新增工程开关：

- `--loss-robust-delta`，默认 `0.0`，保持原始 L1 行为。
- 当 `delta > 0` 时，L1 残差项改为饱和残差：

$$
\rho_{\delta}(r)
=
\frac{\delta |r|}{|r|+\delta}
$$

其梯度幅度为：

$$
\left|\frac{\partial \rho_{\delta}}{\partial r}\right|
=
\frac{\delta^2}{(|r|+\delta)^2}
$$

因此小残差仍会被优化，大残差会被明显降权。这个公式比 Huber 更激进，因为损失本身有上界，适合作为动态遮挡 outlier 实验，不适合作为默认重建损失。

实验命令摘要：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_sharpness_20260428/robust_delta_0_10_3000.ply \
  --iterations 3000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --loss-robust-delta 0.1 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_sharpness_20260428/crops_robust_delta_0_10_3000 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

结果：

| 配置 | iter | eval blur | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline old | 3000 | `0.3` | 69437 | 21.2837 | 14.8517 | 0.7900 | 0.3762 | - | 原始 3000-step 基线 |
| robust delta `0.1` | 3000 | `0.2` | 54392 | 20.9817 | 14.0259 | 0.8712 | 0.5416 | 86.66s | splat 数显著下降，但 PSNR 和 worst frame 变差 |
| robust delta `0.5` | 3000 | `0.2` | 62495 | 21.0888 | 14.2511 | 0.8576 | 0.5311 | 90.27s | 比 `0.1` 温和，但仍低于 baseline，不能继续长跑 |

主观 crop 观察：

- frame 90 的 target 中有人体动态遮挡，render 中人的区域和显示器边缘被弱化，diff 更集中，说明 `delta=0.1` 把有用边界残差信号也压掉了。
- frame 0 / 120 也出现细节欠拟合：键盘、纸张文字和桌面小物体更锐度化但不准确，属于指标锐度提高、重建质量下降。
- 这说明 TUM Freiburg1 XYZ 的“糊”不是简单由少量 outlier 像素主导；过强的饱和鲁棒损失会减少 densification 和细节学习。

当前结论：

1. `--loss-robust-delta 0.1` 可以作为 outlier 诊断和效率实验开关，但不应推荐为默认质量提升方案。
2. `delta=0.5` 保留了一部分 densification，最终 `62.5k` splats，但 PSNR mean / worst frame 仍低于 baseline；说明问题不是单纯 outlier 像素过拟合。
3. 该方向唯一积极信号是 splat 数从约 `69k` 降到 `54k-62k`，训练效率和模型大小更好；但画质代价过大。
4. 若未来要继续鲁棒化，更合理的方案可能是 **按帧/区域 mask 动态物体**，而不是对全部像素使用饱和损失。
5. 鲁棒残差方向停止继续探索：保留 CLI 开关用于诊断，不进入推荐 profile。

### 16.3 2026-04-28 训练期 raster blur schedule 实验记录

本方向目标：避免 `raster_cov_blur=0.2` 从训练一开始介入导致 10k 长跑崩塌，同时测试 topology 基本成形后再切到低 blur 是否能进一步锐化。

新增工程开关：

- `--raster-cov-blur-final <value>`：训练后期使用的 covariance blur floor，默认关闭。
- `--raster-cov-blur-final-after-epoch <epoch>`：切换 epoch；未显式设置时跟随 `--litegs-topology-freeze-after-epoch`。
- 默认行为不变：不传 `--raster-cov-blur-final` 时，全程使用 `--raster-cov-blur`。

实验命令摘要：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_sharpness_20260428/blur_schedule_0_3_to_0_2_after_e4_10000.ply \
  --iterations 10000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --raster-cov-blur-final 0.2 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_sharpness_20260428/crops_blur_schedule_0_3_to_0_2_after_e4_10000 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

结果：

| 配置 | 训练 blur | final blur | 切换 epoch | iter | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| direct low blur | `0.2` | - | 0 | 10000 | 77156 | 16.7333 | 8.5736 | 0.6761 | 0.3825 | - | 从头低 blur 长跑崩塌 |
| stable train + sharp render | `0.3` | - | - | 10000 | ~77106 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | ~400s | 当前推荐配置 |
| blur schedule | `0.3` | `0.2` | 4 | 10000 | 77064 | 21.8193 | 16.5130 | 0.8342 | 0.4947 | 401.66s | 稳定但不优于推荐配置 |

主观 crop 观察：

- frame 90 不再像 direct low blur 10k 那样崩坏，说明“低 blur 延后到 topology freeze 后”是稳定的。
- 但与 stable train + sharp render 相比，键盘、纸张和显示器边缘没有可见优势，锐度指标也更低。
- worst frame PSNR 从 `16.3885` 小幅到 `16.5130`，但 mean PSNR 和锐度同时下降，不足以替换推荐方案。

当前结论：

1. 训练期 blur schedule 是安全的实验能力，解决了“从头低 blur 长跑崩塌”的风险。
2. 对 TUM Freiburg1 XYZ，目前更好的策略仍然是 **训练全程 `0.3`，评估/导出 `0.2`**。
3. 不继续做 `0.25` 或更晚切换 sweep：预期只会逐渐退化为稳定训练基线，收益空间很小。
4. 该方向停止继续探索：保留开关用于其它数据集，不进入当前推荐 profile。

### 16.4 2026-04-28 growth select fraction 效率实验记录

本方向目标：在不改损失、不改渲染 blur 的前提下，降低 densification 的额外 growth 采样比例，测试能否减少 splat 数和训练耗时，同时保持训练结果不明显变糊。

固定设置：

- 训练全程 `--raster-cov-blur 0.3`。
- 评估/导出 `--eval-raster-cov-blur 0.2`。
- 其余学习率和 `--litegs-topology-freeze-after-epoch 4` 与推荐配置一致。

结果：

| 配置 | growth fraction | iter | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable train + sharp render | `0.25` | 10000 | ~77106 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | ~400s | 当前质量推荐 |
| efficiency sweep | `0.20` | 10000 | 63473 | 21.6433 | 15.4658 | 0.8457 | 0.5240 | 373.16s | splat 数明显下降，但 worst frame 退化过大 |
| efficiency sweep | `0.23` | 10000 | 71792 | 21.7968 | 16.2819 | 0.8469 | 0.5276 | 390.76s | 质量接近推荐配置，可作为效率 profile |

主观 crop 观察：

- `0.20` 对 frame 90 欠拟合更明显，动态人/显示器区域 diff 增大，说明 growth 预算不足会先伤害困难视角。
- `0.23` 的 crop 与推荐配置接近，键盘和纸张细节没有明显额外糊化；主要代价是 worst frame PSNR 小降。

当前结论：

1. `growth_select_fraction=0.20` 不适合作为质量 profile：省下约 17.7% splats，但 mean PSNR 降约 `0.19 dB`，worst frame 降约 `0.92 dB`。
2. `growth_select_fraction=0.23` 可以作为效率 profile：省下约 6.9% splats，mean PSNR 只降约 `0.04 dB`，锐度指标接近。
3. 当前默认仍应保持 `0.25`，因为它是质量优先配置；如果目标是更小 PLY 或稍快训练，可以显式使用 `--litegs-growth-select-fraction 0.23`。
4. 不继续向 `0.24` / `0.22` 做细扫：收益区间已经很窄，主要结论足够明确。

### 16.5 2026-04-28 opacity prune threshold 实验记录

本方向目标：通过提高低 opacity splat 的剪枝阈值，减少杂散 splat / floater，同时保持画质。

固定设置：

- `--litegs-prune-mode weight` 保持默认。
- 只把 `--litegs-prune-opacity-threshold` 从默认 `1/255 ~= 0.00392` 提高到 `0.01`。
- 训练全程 `--raster-cov-blur 0.3`，评估 `--eval-raster-cov-blur 0.2`。

结果：

| 配置 | prune opacity threshold | iter | 最终 splats | prune removed | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable train + sharp render | `0.00392` | 10000 | ~77106 | ~3 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | ~400s | 当前质量推荐 |
| prune threshold sweep | `0.01` | 10000 | 76885 | 78 | 21.8074 | 16.1604 | 0.8363 | 0.5150 | 402.32s | 剪得很少且质量下降 |

当前结论：

1. 当前 TUM Freiburg1 XYZ 的杂散 splat 不是大量低 opacity splat 简单堆积；把阈值提高到 `0.01` 只额外剪掉几十个 splat。
2. 该改动没有减少 PLY 体积或训练时间，mean PSNR、worst PSNR 和锐度都下降。
3. 不继续抬高到 `0.02+`：在 `0.01` 已经没有收益的情况下，更高阈值更可能误删有贡献 splat。
4. 剪枝方向停止继续探索；默认阈值保持 `1/255`。

### 16.6 2026-04-28 训练迭代数 / LR decay horizon 实验记录

本方向目标：确认 10000 iter 是否已经超过当前数据集的有效收益拐点。固定 topology 策略和 blur 策略，只同步缩短 `--iterations` 与 `--lr-decay-iterations`，避免短训时仍沿用 10k 的学习率衰减节奏。

固定设置：

- 训练全程 `--raster-cov-blur 0.3`。
- 评估/导出 `--eval-raster-cov-blur 0.2`。
- `--litegs-topology-freeze-after-epoch 4` 保持不变。
- `--lr-decay-iterations` 与 `--iterations` 一致。

结果：

| 配置 | iter | lr decay iter | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stable train + sharp render | 10000 | 10000 | 77106 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | ~400s | 当前质量推荐 |
| iteration sweep | 8000 | 8000 | 77746 | 21.9221 | 16.4191 | 0.8390 | 0.5059 | 314.21s | 当前最快高质量 profile |
| iteration sweep | 7000 | 7000 | 78128 | 21.7740 | 16.1309 | 0.8332 | 0.4975 | 271.80s | 进一步缩短开始明显退化 |

复核命令：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --features gpu,cli --example evaluate_psnr -- \
  --scene RustGS/output/experiments/tum_sharpness_20260428/baseline_8000_eval_blur_0_2.ply \
  --dataset /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --render-scale 0.25 \
  --raster-cov-blur 0.2 \
  --frame-stride 30 \
  --max-frames 180 \
  --device cpu \
  --json
```

当前结论：

1. `8000` iter 是当前数据集的效率拐点：训练时间比 10k 少约 21%，mean PSNR 和 worst PSNR 反而略高。
2. `8000` iter 的锐度指标低于 10k，说明额外 2k 迭代主要带来边缘锐化和局部纹理贴合，而不是 PSNR 提升。
3. `7000` iter 虽然更快，但 mean PSNR、worst PSNR、锐度同时下降；不作为推荐 profile。
4. 推荐保留两个 profile：
   - 质量 / 主观锐度优先：`10000` iter + train blur `0.3` + eval/render blur `0.2`。
   - 训练效率优先：`8000` iter + train blur `0.3` + eval/render blur `0.2`。
5. 下一个值得探索的组合方向：`8000 iter + growth_select_fraction=0.23`，测试是否可以在 8k 高质量拐点上进一步减少 splat 数和训练时间。

### 16.7 2026-04-28 `8000 iter + growth_select_fraction=0.23` 组合实验记录

本方向目标：把 16.4 的 splat 数压缩收益和 16.6 的 8k 高效率拐点组合起来，测试是否能在更短训练时间下保持画质。

实验命令摘要：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_sharpness_20260428/growth_fraction_0_23_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --litegs-growth-select-fraction 0.23 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_sharpness_20260428/crops_growth_fraction_0_23_8000_eval_blur_0_2 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

结果：

| 配置 | iter | growth fraction | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10000 quality profile | 10000 | `0.25` | 77106 | 21.8377 | 16.3885 | 0.8491 | 0.5327 | ~400s | 主观锐度优先 |
| 8000 PSNR profile | 8000 | `0.25` | 77746 | 21.9221 | 16.4191 | 0.8390 | 0.5059 | 314.21s | PSNR / 时间折中最好 |
| 8000 compact profile | 8000 | `0.23` | 72336 | 21.8116 | 16.4838 | 0.8407 | 0.5060 | 307.95s | 更小模型，质量接近 |

主观 crop 观察：

- frame 90：人和桌面动态/遮挡区域仍是主要误差来源；`0.23` 没有出现 direct-low-blur 那类颜色崩塌。
- frame 120：键盘和纸张边缘与 8k baseline 接近，diff 主要集中在高频边缘和屏幕遮挡边界。
- 相比 `8000 + 0.25`，`0.23` 的 mean PSNR 降约 `0.11 dB`，但 worst frame 反而略高；锐度指标几乎持平。

当前结论：

1. `8000 iter + growth_select_fraction=0.23` 是当前最好的 **compact efficiency profile**：比 10k quality profile 少约 6.2% splats，训练时间少约 23%，画质只小幅下降。
2. 如果只看 PSNR，`8000 + 0.25` 仍然更好；如果要控制 PLY 体积和训练时间，`8000 + 0.23` 更均衡。
3. 不继续做 `8000 + 0.20`：16.4 已证明 `0.20` 会明显伤害困难视角，8k 短训下风险更大。
4. 不继续做 `0.24` 细扫：相对 `0.23` 的 splat 节省会变小，预计只在 `0.23` 与 `0.25` 之间线性折中，不会产生新的策略结论。

### 16.8 2026-04-28 `abs-pixel` 长训验证记录

本方向目标：验证 3000-step 短跑中表现较好的 augmented AbsPixel split，在 8000-step 长训下是否继续改善糊和困难视角。

工程修正：

- 文档和 3000-step 结果都指向 `abs-pixel` 推荐阈值 `1e-5`。
- 代码中 `--litegs-profile abs-pixel` 仍固化为 `2e-5`，已修正为 `1e-5`。
- 对应测试 `litegs_profile_applies_stable_experimental_split_defaults` 已同步改为断言 `0.00001`。

实验命令摘要：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_sharpness_20260428/abs_pixel_profile_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --litegs-profile abs-pixel \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_sharpness_20260428/crops_abs_pixel_profile_8000_eval_blur_0_2 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

结果：

| 配置 | split score | threshold | iter | 最终 splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 训练耗时 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8000 PSNR profile | baseline | - | 8000 | 77746 | 21.9221 | 16.4191 | 0.8390 | 0.5059 | 314.21s | 当前 PSNR / 时间折中最好 |
| 8000 compact profile | baseline | - | 8000 | 72336 | 21.8116 | 16.4838 | 0.8407 | 0.5060 | 307.95s | 当前 compact efficiency profile |
| `abs-pixel` profile | abs-pixel augmented | `0.00001` | 8000 | 78511 | 21.8782 | 16.1888 | 0.8399 | 0.5070 | 318.86s | 没有超过 baseline，worst frame 退化 |

主观 crop 观察：

- frame 90 仍由动态人和遮挡边界主导误差；`abs-pixel` 没有明显减少该处 blur / ghosting。
- 额外 split 后 splat 数略高于 8k baseline，但锐度没有实质提高。
- 与 3000-step 短跑不同，长训后 baseline 已经能靠后半段优化追回大部分高频，AbsPixel 的额外 split 不再带来净收益。

当前结论：

1. `abs-pixel` profile 的阈值配置错误已修正；作为实验开关保留。
2. 对 TUM Freiburg1 XYZ，`abs-pixel` 不进入推荐 profile：mean PSNR 低于 `8000 + 0.25`，worst frame 也低于 `8000 + 0.23`。
3. 不继续跑 `abs-pixel` 10k / 30k：8k 已显示其主要收益没有延续，继续长训大概率只增加时间和 splat 数。
4. 若后续要继续借鉴 AbsGS，优先方向应改为 **动态/遮挡区域 mask** 或更准确的 visibility 统计，而不是继续调 AbsPixel 阈值。

### 16.9 2026-04-28 导出 / 评估 blur 下限复核

本方向目标：不改变训练，只调整导出和评估时的 `raster_cov_blur`，确认“更锐导出”是否可以作为解决主观糊感的低成本选项。

固定设置：

- 训练 profile 使用 `8000 + baseline + train blur 0.3` 或 `8000 + growth_select_fraction=0.23 + train blur 0.3`。
- 只改变评估 / 导出 `--raster-cov-blur`。

结果：

| 训练配置 | eval/render blur | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 8000 PSNR profile | `0.20` | 77746 | 21.9221 | 16.4191 | 0.8390 | 0.5059 | PSNR 最优 |
| 8000 PSNR profile | `0.15` | 77746 | 21.8966 | 16.3908 | 0.8745 | 0.6040 | 更锐，PSNR 小降 |
| 8000 PSNR profile | `0.10` | 77746 | 21.8156 | 16.3456 | 0.9176 | 0.7382 | 过锐风险上升，不建议默认 |
| 8000 compact profile | `0.20` | 72336 | 21.8116 | 16.4838 | 0.8407 | 0.5060 | compact 默认 |
| 8000 compact profile | `0.15` | 72336 | 21.7852 | 16.4577 | 0.8761 | 0.6033 | compact 更锐导出 |

当前结论：

1. 对当前 TUM 数据，`eval/render blur=0.15` 是合理的主观锐化导出选项：PSNR 只降约 `0.03 dB`，lap sharpness 从约 `0.506` 提升到约 `0.604`。
2. `0.10` 虽继续提高锐度指标，但 frame 90 的 lap sharpness ratio 已超过 `1.30`，容易把动态/遮挡边界变成过锐边缘，不作为默认推荐。
3. 推荐区分两个导出口径：
   - 指标 / 稳定比较：`--eval-raster-cov-blur 0.2`。
   - 主观锐度优先：render/export 时使用 `--raster-cov-blur 0.15`。
4. 不继续向 `0.12` / `0.18` 细扫：`0.15` 已经给出清晰收益，`0.10` 给出过锐下界，中间细分不会改变推荐策略。

### 16.10 2026-04-28 当前停止判断与最终推荐

截至本轮，已经探索并停止的方向：

| 方向 | 结论 |
|---|---|
| robust residual loss | 锐度指标提高但欠拟合，PSNR 和主观质量下降 |
| 训练期 blur schedule | 稳定但不优于“训练 0.3 + 导出 0.2/0.15” |
| growth fraction | `0.23` 有效率收益，`0.20` 质量损失过大 |
| opacity prune threshold | `0.01` 几乎剪不掉 splat，质量下降 |
| 迭代数 | `8000` 是当前效率拐点，`7000` 开始退化 |
| `8000 + growth_fraction=0.23` | 当前 compact efficiency profile |
| `abs-pixel` 长训 | 代码阈值已修正，但 TUM 8k 长训不优于 baseline |
| 导出 blur | `0.15` 可作为更锐导出，`0.10` 过锐风险上升 |

推荐命令一：质量 / 主观锐度优先。

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/tum_quality_10000.ply \
  --iterations 10000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 10000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-raster-cov-blur 0.2 \
  --eval-json
```

推荐命令二：PSNR / 训练时间折中。

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/tum_fast_8000.ply \
  --iterations 8000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-raster-cov-blur 0.2 \
  --eval-json
```

推荐命令三：更小 PLY / compact efficiency。

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/tum_compact_8000.ply \
  --iterations 8000 \
  --litegs-topology-freeze-after-epoch 4 \
  --litegs-growth-select-fraction 0.23 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-raster-cov-blur 0.2 \
  --eval-json
```

更锐导出建议：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- render \
  --input RustGS/output/tum_fast_8000.ply \
  --camera <camera.json> \
  --output RustGS/output/tum_fast_8000_sharp.png \
  --raster-cov-blur 0.15
```

当前停止判断：

1. 已测试的训练侧可调方向里，没有新的方向同时超过 `8000 baseline` 的 PSNR、`10000 baseline` 的主观锐度和 `8000 + 0.23` 的效率。
2. TUM frame 90 的主要误差来自动态人 / 遮挡，不是单纯 splat blur；继续调 densification、prune 或 robust loss 的收益空间很小。
3. 若未来继续突破，需要新输入信号而不是继续调现有超参：例如动态区域 mask、训练/评估全帧更密集验证、或 pose / exposure 质量诊断。

验证：

```sh
cargo fmt --manifest-path RustGS/Cargo.toml
cargo check --manifest-path RustGS/Cargo.toml --features gpu,cli
cargo test --manifest-path RustGS/Cargo.toml --features gpu,cli
```

结果：全部通过；项目级测试为 lib `138` 个、bin `22` 个、TUM tests `4` 个、doc-test `1` 个通过，2 个 integration test 保持 ignored。

### 16.11 2026-04-28 动态 / 遮挡帧诊断与硬排除实验

本方向目标：验证最差帧是否来自动态人 / 遮挡导致的不可解释残差，并测试简单排除整段动态帧能否改善静态场景质量。

新增工程开关：

- `--exclude-frame-ranges <ranges>`：训练时按 frame id 排除整帧，格式支持 `76-93,155` 或 `76..93,155`。
- `--eval-exclude-frame-ranges <ranges>`：`rustgs train --eval-after-train` 的评估阶段排除指定 frame id。
- `examples/evaluate_psnr` 增加 `--exclude-frame-ranges`，用于离线逐帧诊断。

逐帧诊断：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --features gpu,cli --example evaluate_psnr -- \
  --scene RustGS/output/experiments/tum_sharpness_20260428/baseline_8000_eval_blur_0_2.ply \
  --dataset /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --render-scale 0.25 \
  --raster-cov-blur 0.2 \
  --frame-stride 1 \
  --max-frames 180 \
  --device cpu \
  --json \
  --export-worst-k 20 \
  --export-dir RustGS/output/experiments/tum_dynamic_diagnostics_20260428/baseline_8000_worst20
```

结果：

| 模型 | 评估帧 | 排除区间 | PSNR mean | PSNR min | worst frames | 结论 |
|---|---:|---|---:|---:|---|---|
| baseline 8k | 180 | 无 | 22.1698 | 13.0054 | `81,82,84,80,83,85,79,86,78,87,...` | 最差帧高度集中在 `76-93` |
| baseline 8k | 162 | `76-93` | 22.8295 | 17.5597 | `73,155,75,122,52` | 排除动态区间后分布正常很多 |

主观观察：

- worst 20 主要集中在 frame `76-93`，并不是随机散布到所有视角。
- strip 显示误差集中在人、显示器遮挡边界、桌面可见性变化；这更像动态遮挡 / 非静态场景误差，不是单纯训练 blur。
- frame 90 只是这段区间的一帧；最低点其实是 frame `81-85`。

硬排除训练实验：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_dynamic_diagnostics_20260428/exclude_76_93_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --exclude-frame-ranges 76-93 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --eval-crop-output-dir RustGS/output/experiments/tum_dynamic_diagnostics_20260428/crops_exclude_76_93_8000_eval_blur_0_2 \
  --eval-crop-frames 0,90,120 \
  --log-level info
```

对比结果：

| 模型 | 训练排除 | 评估排除 | eval frames | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | 无 | 无 | 180 | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | 当前全帧基线 |
| exclude 76-93 8k | `76-93` | 无 | 180 | 78586 | 21.7338 | 9.4940 | 0.8382 | 0.5063 | 动态区间严重退化 |
| baseline 8k | 无 | `76-93` | 162 | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| exclude 76-93 8k | `76-93` | `76-93` | 162 | 78586 | 22.6458 | 16.4745 | 0.8371 | 0.4926 | 静态子集也下降 |

当前结论：

1. 逐帧诊断证明最差 PSNR 主要来自 `76-93` 这段动态 / 遮挡区间。
2. 但硬排除整段动态帧不是好策略：它会损失重要视角约束，导致静态子集 mean PSNR 下降约 `0.18 dB`，完整 180 帧下降约 `0.44 dB`。
3. 这说明“动态问题”应该用 **像素/区域级 mask 或 loss reweighting** 处理，而不是整帧删除。
4. `--exclude-frame-ranges` 保留为诊断和数据清洗工具，不进入当前推荐训练 profile。
5. 下一个如果继续突破，应实现 residual-based 动态 mask：对高残差且跨迭代不稳定的局部区域降权，而保留同一帧中的静态背景约束。

### 16.12 2026-04-28 soft outlier pixel loss 实验

本方向目标：不删除整帧，而是在 L1 重建项里对高残差像素做软降权，验证它是否能压低动态 / 遮挡区域对静态背景的污染。

新增工程开关：

- `--loss-outlier-threshold <t>`：高残差软降权阈值，`0` 表示关闭。
- `--loss-outlier-weight <w_min>`：高残差像素的梯度下限，范围 `[0, 1]`，默认 `1.0`。

当前实现公式：

设单通道残差为：

```text
r = |\hat I - I|
```

当 `loss_robust_delta == 0` 且 `t > 0`、`w_min < 1` 时，L1 项替换为：

```text
w(r) = w_min + (1 - w_min) * t / (r + t)
L_outlier = mean( r * w(r) )
```

对应残差幅值方向的梯度为：

```text
d(r * w(r)) / dr = w_min + (1 - w_min) * t^2 / (r + t)^2
```

因此小残差接近普通 L1，大残差的梯度下限趋近 `w_min`，不会像 saturating robust loss 那样完全压平。

同时修复了一个诊断工具口径问题：

- `examples/evaluate_psnr --exclude-frame-ranges` 现在与 `rustgs train --eval-after-train` 一致：先按 `--max-frames` 取评估前缀，再排除 frame range，最后应用 `--frame-stride`。
- 修复前会“先排除再取前 180 帧”，导致 `76-93` 被排除后由后续 frame 补满 180 帧，静态子集不是同口径。

实验命令模板：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_dynamic_diagnostics_20260428/soft_outlier_t0_5_w0_5_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --litegs-topology-freeze-after-epoch 4 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --loss-outlier-threshold 0.5 \
  --loss-outlier-weight 0.5 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

完整 180 帧评估结果：

| 模型 | outlier `t/w_min` | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | worst frames | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline 8k | off | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | `81,82,84,80,83` | 当前全帧基线 |
| soft outlier 8k | `0.25 / 0.25` | 68570 | 21.9965 | 11.5174 | 0.8422 | 0.5008 | `81,82,84,80,83` | 静态边缘更锐，但动态段明显更差 |
| soft outlier 8k | `0.50 / 0.50` | 73529 | 22.0975 | 12.8245 | 0.8385 | 0.4918 | `81,82,84,83,80` | 最接近 baseline，但仍低于全帧基线 |
| soft outlier 8k | `0.75 / 0.75` | 76049 | 22.0993 | 12.6502 | 0.8299 | 0.4836 | `81,82,84,80,83` | 更温和也没有超过 baseline |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | outlier `t/w_min` | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | off | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| soft outlier 8k | `0.25 / 0.25` | 68570 | 22.9043 | 17.5353 | 0.8423 | 0.4962 | 静态 mean 提升，min 略低 |
| soft outlier 8k | `0.50 / 0.50` | 73529 | 22.8993 | 17.4403 | 0.8368 | 0.4868 | 静态 mean 提升，min 继续低 |
| soft outlier 8k | `0.75 / 0.75` | 76049 | 22.9318 | 17.4736 | 0.8291 | 0.4785 | 静态 mean 最高，但全帧不优 |

当前结论：

1. soft outlier 证明“像素级降权”这个方向有信号：静态子集 mean PSNR 提升约 `0.07-0.10 dB`，且 splat 数可低于 baseline。
2. 但它不是当前可推荐的默认 profile：完整 180 帧 mean PSNR 仍低于 baseline，动态段 worst min 也没有改善。
3. 失败原因更像“高残差不等于动态遮挡”：真实的动态人 / 遮挡边界需要被局部解释；简单按残差降权会减少这段视角的有效监督，使 frame `81-84` 更欠拟合。
4. 该开关保留为实验能力，不进入推荐命令。若应用只关心静态背景重建、且愿意牺牲动态段，可考虑 `0.75 / 0.75` 作为静态优先候选。
5. 继续突破应从 **空间一致的 dynamic mask** 或 **遮挡区域识别** 入手，而不是继续扫全局残差阈值。

### 16.13 2026-04-28 训练帧采样口径与 prefix-180 专项训练

本方向目标：验证当前“训练 798 帧、默认只评估前 180 帧”的口径是否摊薄了每个评估帧的监督，进而造成默认评估结果偏糊。

背景判断：

- 当前 TUM COLMAP 输入共有 `798` 帧。
- 默认 post-train / `evaluate_psnr` 对比常用 `--max-frames 180`。
- `8000` iter 训练全 `798` 帧时，每帧平均只被采样约 `10` 次；若只训练前 `180` 帧，每帧平均约 `44` 次。
- 但 topology freeze 按 epoch 计。如果直接使用 `--max-frames 180 --litegs-topology-freeze-after-epoch 4`，实际约 `720` iter 就冻结，和全量 798 帧训练的 `4 * 798 = 3192` iter 不同。
- 为保持冻结迭代数接近，本实验使用 `--max-frames 180 --litegs-topology-freeze-after-epoch 18`，即 `18 * 180 = 3240` iter。

同时修复一个 CLI 可用性问题：

- `rustgs train --output path/to/scene.ply` 现在会自动创建输出父目录。
- 之前当目录不存在时，训练完整跑完后会在保存 PLY 时失败。

推荐实验命令，质量优先：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_frame_sampling_20260428/prefix180_freeze18_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --max-frames 180 \
  --litegs-topology-freeze-after-epoch 18 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

推荐实验命令，compact / efficiency：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_frame_sampling_20260428/prefix180_freeze18_growth0_14_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --max-frames 180 \
  --litegs-topology-freeze-after-epoch 18 \
  --litegs-growth-select-fraction 0.14 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

完整前 180 帧评估结果：

| 模型 | 训练帧 | freeze epoch | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | 798 | 4 | 0.25 | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | 全量训练基线 |
| prefix-180 quality | 180 | 18 | 0.25 | 91859 | 23.0782 | 16.5743 | 0.8685 | 0.5593 | 当前前 180 帧质量最高 |
| prefix-180 compact-old | 180 | 18 | 0.20 | 74857 | 22.6567 | 16.7428 | 0.8430 | 0.5192 | 早期 compact 候选 |
| prefix-180 compact | 180 | 18 | 0.18 | 66826 | 22.9934 | 16.3018 | 0.8722 | 0.5592 | 比 0.20 更好且更小 |
| prefix-180 compact | 180 | 18 | 0.16 | 58222 | 22.9918 | 16.6616 | 0.8761 | 0.5594 | 质量持平，继续降 splats |
| prefix-180 compact-best | 180 | 18 | 0.14 | 49648 | 23.0013 | 16.3364 | 0.8766 | 0.5551 | 当前效率/质量最优点 |
| prefix-180 underfit | 180 | 18 | 0.12 | 41245 | 22.0128 | 15.4053 | 0.8501 | 0.5147 | 明显欠拟合 |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | 训练帧 | freeze epoch | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | 798 | 4 | 0.25 | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| prefix-180 quality | 180 | 18 | 0.25 | 91859 | 23.6976 | 18.2898 | 0.8667 | 0.5591 | 静态质量最高 |
| prefix-180 compact-old | 180 | 18 | 0.20 | 74857 | 23.2151 | 18.2835 | 0.8393 | 0.5142 | 早期 compact 候选 |
| prefix-180 compact | 180 | 18 | 0.18 | 66826 | 23.6331 | 18.0451 | 0.8707 | 0.5587 | 接近 quality，少 27% splats |
| prefix-180 compact | 180 | 18 | 0.16 | 58222 | 23.6138 | 18.0415 | 0.8734 | 0.5590 | 与 0.18 基本持平 |
| prefix-180 compact-best | 180 | 18 | 0.14 | 49648 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 当前效率/质量最优点 |
| prefix-180 underfit | 180 | 18 | 0.12 | 41245 | 22.6103 | 17.7744 | 0.8473 | 0.5115 | 明显欠拟合 |

6-frame 默认 post-train eval 结果：

| 模型 | splats | PSNR mean | PSNR min(frame 90) | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline 8k | 77746 | 21.9221 | 16.4191 | 0.8390 | 0.5059 | 旧推荐 |
| prefix-180 quality | 91859 | 22.7214 | 17.5571 | 0.8781 | 0.5879 | 质量优先 |
| prefix-180 compact-old | 74857 | 22.3107 | 17.6610 | 0.8551 | 0.5536 | 早期 compact |
| prefix-180 compact-best | 49648 | 22.6355 | 17.5287 | 0.8871 | 0.5822 | 当前推荐 compact |
| prefix-180 underfit | 41245 | 21.9024 | 17.0098 | 0.8667 | 0.5504 | 过度压缩 |

当前结论：

1. 这是目前对“默认前 180 帧评估发糊”最有效的方向：不是调 loss，而是让训练帧集合与评估目标一致。
2. `prefix-180 compact-best` 是当前最实用 profile：`--litegs-growth-select-fraction 0.14` 只有 `49648` splats，比全量 baseline 少 `36%`，但 full-180 mean PSNR 从 `22.1698` 提升到 `23.0013`，static-162 mean 从 `22.8295` 提升到 `23.6352`。
3. `--litegs-growth-select-fraction 0.12` 是明确拐点：splat 数继续降到 `41245`，但 full-180 mean 掉到 `22.0128`，static-162 mean 掉到 `22.6103`，锐度指标也明显下降。继续压低 growth fraction 不值得。
4. `prefix-180 quality` 适合质量优先：full-180 mean PSNR `23.0782` 仍最高，但 splats 增加到 `91859`。和 `0.14` 相比只多约 `0.077 dB`，却多约 `85%` splats，因此默认更建议 `0.14`。
5. 这个结论只适用于“目标就是前 180 帧 / 当前默认评估口径”的场景。若要重建完整 798 帧轨迹或后续视角，仍应训练全量帧，不能把 `--max-frames 180` 当成通用默认。
6. 后续若要兼顾完整 798 帧和局部细节，应做分层/分段采样：例如按时间窗口均匀抽帧、困难帧 oversampling、或按 eval set 显式配置 train/eval split，而不是隐式只评估前缀。

### 16.14 2026-04-28 显式帧范围选择 include / exclude

本方向目标：把 prefix-180 这种隐式实验口径改造成可复现、可组合的显式 train/eval frame range 选择能力，为后续分段采样和困难帧 oversampling 铺路。

新增能力：

- `rustgs train --include-frame-ranges <ranges>`：训练集只保留指定 frame id / inclusive ranges。
- `rustgs train --eval-include-frame-ranges <ranges>`：post-train eval 只保留指定 frame id / inclusive ranges。
- `evaluate_psnr --include-frame-ranges <ranges>`：离线评估只保留指定 frame id / inclusive ranges。
- 已有 `--exclude-frame-ranges` / `--eval-exclude-frame-ranges` 保持可用。

过滤顺序：

1. dataset loader 先应用 `--max-frames` 和 `--frame-stride`。
2. 再应用 include ranges。
3. 最后应用 exclude ranges。

这样可以表达：

```sh
# 等价于“完整 798 帧数据中只评估前 180 帧静态子集”
cargo run --release --manifest-path RustGS/Cargo.toml --features gpu,cli --example evaluate_psnr -- \
  --scene RustGS/output/experiments/tum_frame_sampling_20260428/prefix180_freeze18_growth0_14_8000_eval_blur_0_2.ply \
  --dataset /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --render-scale 0.25 \
  --raster-cov-blur 0.2 \
  --frame-stride 1 \
  --include-frame-ranges 0-179 \
  --exclude-frame-ranges 76-93 \
  --device cpu \
  --json
```

验证结果：

| 验证命令 | frame_count | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `evaluate_psnr --include-frame-ranges 0-179 --exclude-frame-ranges 76-93` | 162 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 与 `--max-frames 180 --exclude-frame-ranges 76-93` 完全一致 |

当前结论：

1. 这个改动本身不是新的优化算法，但解决了实验不可复现的问题：不再只能靠 prefix `--max-frames` 表达评估窗口。
2. 对 798 帧完整轨迹，下一步可以尝试显式分段训练，例如 `0-179,240-419,600-797`，或者先用 `evaluate_psnr` 找 worst frame 后再 oversample 对应窗口。
3. include/exclude 的组合也能稳定表达静态背景子集，避免把动态帧处理和 dataset loader prefix 混在一起。

### 16.15 2026-04-28 全轨迹 uniform stride-4 采样

本方向目标：验证是否可以从完整 `798` 帧中均匀抽取约 `200` 帧训练，以兼顾全轨迹覆盖和每帧训练次数。

实验设置：

- `--frame-stride 4`，从全轨迹加载 `200` 帧。
- `--litegs-topology-freeze-after-epoch 16`，约 `16 * 200 = 3200` iter 后冻结，和 prefix-180 的 freeze18 接近。
- 对比两个 growth budget：
  - compact：`--litegs-growth-select-fraction 0.14`
  - quality：`--litegs-growth-select-fraction 0.25`

前 180 帧完整评估：

| 模型 | 训练帧 | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | 798 | 0.25 | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | 全量训练基线 |
| prefix-180 compact-best | 180 | 0.14 | 49648 | 23.0013 | 16.3364 | 0.8766 | 0.5551 | 当前推荐 |
| full stride-4 compact | 200 | 0.14 | 43914 | 21.9030 | 12.7615 | 0.8081 | 0.4292 | 前 180 明显更糊 |
| full stride-4 quality | 200 | 0.25 | 78131 | 22.0426 | 13.8251 | 0.8059 | 0.4404 | 加 splats 后仍不如 baseline |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | 训练帧 | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | 798 | 0.25 | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| prefix-180 compact-best | 180 | 0.14 | 49648 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 当前推荐 |
| full stride-4 compact | 200 | 0.14 | 43914 | 22.7020 | 17.5344 | 0.8036 | 0.4238 | 质量低于 baseline |
| full stride-4 quality | 200 | 0.25 | 78131 | 22.7724 | 17.7974 | 0.8025 | 0.4347 | 仍低于 baseline 和 prefix |

全轨迹 stride-4 评估，固定评估 `0,4,8,...` 共 200 帧：

| 模型 | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline 8k full-798 | 77746 | 22.9422 | 13.2810 | 0.8626 | 0.5386 | 全量训练基线 |
| full stride-4 compact | 43914 | 22.6655 | 13.0461 | 0.8437 | 0.4959 | 更小，但更糊 |
| full stride-4 quality | 78131 | 22.8354 | 14.1489 | 0.8453 | 0.5116 | min 改善，但 mean / sharpness 仍不如 baseline |

当前结论：

1. uniform stride-4 不是当前值得继续投入的训练策略。它减少了每个窗口的连续视角约束，导致局部几何和纹理监督不足。
2. `growth0.14` 虽然 splat 少，但前 180 和全轨迹 stride-4 都明显更糊。
3. `growth0.25` 把 splat 数提高到 `78131`，接近 baseline，但全轨迹 mean PSNR 仍低 `0.11 dB`，锐度也更低；它没有带来效率收益。
4. 如果目标是完整 798 帧，当前更稳的是全量训练 baseline，而不是 uniform stride 子采样。
5. 后续若要兼顾完整轨迹和局部清晰度，应转向 **显式窗口 curriculum / hard-frame oversampling**，例如保留连续片段而不是均匀抽帧。

### 16.16 2026-04-28 前 180 帧 naive oversampling 到全轨迹训练

本方向目标：验证“完整 798 帧覆盖 + 提高前 180 帧采样概率”能否兼顾全轨迹重建和当前默认前缀评估清晰度。

实验设置：

- 加载完整 `798` 帧。
- `--oversample-frame-ranges 0-179 --oversample-frame-repeat 3`，把前 180 帧在训练采样池中重复到总计 `1158` 个 pose 条目。
- `--litegs-topology-freeze-after-epoch 3`，约 `3 * 1158 = 3474` iter 后冻结，和 prefix-180 的 `18 * 180 = 3240` iter 接近。
- `--litegs-growth-select-fraction 0.14`，沿用当前 compact-best 的增长预算。

训练命令：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_frame_sampling_20260428/full_oversample_prefix180x3_freeze3_growth0_14_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --oversample-frame-ranges 0-179 \
  --oversample-frame-repeat 3 \
  --litegs-topology-freeze-after-epoch 3 \
  --litegs-growth-select-fraction 0.14 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

前 180 帧完整评估：

| 模型 | 训练采样 | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | full 798 | 0.25 | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | 全量训练基线 |
| prefix-180 compact-best | prefix 180 | 0.14 | 49648 | 23.0013 | 16.3364 | 0.8766 | 0.5551 | 当前推荐 |
| full prefix180 x3 oversample | full 798 + prefix x3 | 0.14 | 50449 | 22.3191 | 13.5039 | 0.8389 | 0.4931 | 好于 full baseline mean，但动态 worst 仍差 |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | 训练采样 | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline 8k | full 798 | 0.25 | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| prefix-180 compact-best | prefix 180 | 0.14 | 49648 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 当前推荐 |
| full prefix180 x3 oversample | full 798 + prefix x3 | 0.14 | 50449 | 23.1100 | 17.8249 | 0.8369 | 0.4909 | 好于 baseline，但不如 prefix |

全轨迹 stride-4 评估，固定评估 `0,4,8,...` 共 200 帧：

| 模型 | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline 8k full-798 | 77746 | 22.9422 | 13.2810 | 0.8626 | 0.5386 | 全量训练基线 |
| full stride-4 compact | 43914 | 22.6655 | 13.0461 | 0.8437 | 0.4959 | 更小，但更糊 |
| full prefix180 x3 oversample | 50449 | 22.4969 | 13.7882 | 0.8637 | 0.5384 | sharpness 接近 baseline，但 mean 明显下降 |

当前结论：

1. naive oversampling 比 uniform stride-4 更符合前 180 帧目标，但仍不是当前主线。它的前 180 mean PSNR `22.3191` 高于 full baseline `22.1698`，但明显低于 prefix-180 compact-best `23.0013`。
2. 静态子集上它确实超过 full baseline：`23.1100` vs `22.8295`，说明提高目标窗口采样概率有效；但距离 prefix-180 compact-best 的 `23.6352` 仍差 `0.53 dB`。
3. 全轨迹 stride-4 上 mean PSNR 掉到 `22.4969`，低于 baseline `22.9422`。简单重复前缀会牺牲后段轨迹，不适合作为完整 798 帧重建策略。
4. 该方向停止继续加 repeat 扫描。继续增大 repeat 只会更接近 prefix-only，同时进一步损害完整轨迹；更合理的下一步是显式多窗口覆盖或困难帧窗口采样。

### 16.17 2026-04-28 显式多段连续窗口训练

本方向目标：验证是否可以用少量连续窗口同时保留局部多视角约束和全轨迹覆盖，避免 uniform stride-4 的稀疏相邻视角问题，也避免 prefix-only 只能服务前 180 帧的问题。

实验设置：

- 训练 include ranges：`0-179,240-419,600-797`。
- 总训练 pose 条目：`558`。
- `--litegs-topology-freeze-after-epoch 6`，约 `6 * 558 = 3348` iter 后冻结，和 prefix-180 / full-798 baseline 的 topology 冻结时机接近。
- `--litegs-growth-select-fraction 0.14`，沿用 compact-best 增长预算。

训练命令：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_frame_sampling_20260428/full_multiwindow_0_179_240_419_600_797_freeze6_growth0_14_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --include-frame-ranges 0-179,240-419,600-797 \
  --litegs-topology-freeze-after-epoch 6 \
  --litegs-growth-select-fraction 0.14 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

前 180 帧完整评估：

| 模型 | 训练采样 | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline 8k | full 798 | 77746 | 22.1698 | 13.0054 | 0.8331 | 0.4801 | 全量训练基线 |
| prefix-180 compact-best | prefix 180 | 49648 | 23.0013 | 16.3364 | 0.8766 | 0.5551 | 当前推荐 |
| full multi-window compact | `0-179,240-419,600-797` | 46577 | 21.4110 | 14.1219 | 0.8360 | 0.4795 | 明显更糊 |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | 训练采样 | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline 8k | full 798 | 77746 | 22.8295 | 17.5597 | 0.8313 | 0.4695 | 静态子集基线 |
| prefix-180 compact-best | prefix 180 | 49648 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 当前推荐 |
| full multi-window compact | `0-179,240-419,600-797` | 46577 | 22.0640 | 17.3863 | 0.8353 | 0.4773 | 低于 baseline |

全轨迹 stride-4 评估，固定评估 `0,4,8,...` 共 200 帧：

| 模型 | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline 8k full-798 | 77746 | 22.9422 | 13.2810 | 0.8626 | 0.5386 | 全量训练基线 |
| full stride-4 compact | 43914 | 22.6655 | 13.0461 | 0.8437 | 0.4959 | 更小，但更糊 |
| full multi-window compact | 46577 | 21.4721 | 14.3597 | 0.8623 | 0.5237 | mean 明显下降 |

当前结论：

1. 显式多段连续窗口不是当前值得继续投入的方向。它虽然保留了窗口内部相邻视角，但窗口间未覆盖区域太大，导致全轨迹和前 180 评估同时退化。
2. 和 naive prefix oversampling 相比，多窗口更差：前 180 mean `21.4110` vs `22.3191`，static mean `22.0640` vs `23.1100`。
3. 该结果说明“覆盖完整轨迹”不能靠少数离散窗口近似；如果目标是完整 798 帧，仍应保留全量训练，或者做更细的 active sampling / curriculum，而不是手工选 3 个窗口。
4. 本方向停止继续扫窗口组合。下一步应回到当前最优 prefix-180 compact，测试 loss / render / topology 的局部消融，寻找真正能减糊且不牺牲口径的改动。

### 16.18 2026-04-28 prefix-180 loss 权重消融

本方向目标：在当前最优的 prefix-180 compact 配置上，验证 SSIM 项是否带来过度平滑，以及能否通过提高 L1 权重降低 splat 数、提升 worst frame 或锐度。

固定设置：

- `--max-frames 180`
- `--iterations 8000`
- `--litegs-topology-freeze-after-epoch 18`
- `--litegs-growth-select-fraction 0.14`，除特别说明外
- `--raster-cov-blur 0.3`
- `--eval-raster-cov-blur 0.2`

实验命令模板：

```sh
cargo run --release --manifest-path RustGS/Cargo.toml --bin rustgs --features gpu,cli -- train \
  --input /Users/tfjiang/Projects/RustScan/test_data/tum_freiburg1_xyz_colmap \
  --output RustGS/output/experiments/tum_loss_ablation_20260428/prefix180_l1_0_9_ssim_0_1_freeze18_growth0_14_8000_eval_blur_0_2.ply \
  --iterations 8000 \
  --max-frames 180 \
  --litegs-topology-freeze-after-epoch 18 \
  --litegs-growth-select-fraction 0.14 \
  --lr-decay-iterations 8000 \
  --lr-scale-final 0.0005 \
  --lr-rotation-final 0.0001 \
  --lr-opacity-final 0.005 \
  --lr-color-final 0.00025 \
  --raster-cov-blur 0.3 \
  --loss-l1-weight 0.9 \
  --loss-ssim-weight 0.1 \
  --eval-after-train \
  --eval-json \
  --eval-raster-cov-blur 0.2 \
  --log-level info
```

前 180 帧完整评估：

| 模型 | loss L1/SSIM | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| prefix-180 compact-best | `0.8 / 0.2` | 0.14 | 49648 | 23.0013 | 16.3364 | 0.8766 | 0.5551 | 当前 PSNR/体积折中 |
| L1-only compact | `1.0 / 0.0` | 0.14 | 31238 | 22.9050 | 16.7351 | 0.8270 | 0.5132 | 极小模型，PSNR 小降 |
| L1-only mid | `1.0 / 0.0` | 0.18 | 37760 | 22.9236 | 16.9702 | 0.8283 | 0.5205 | 稳定，但收益很小 |
| L1-only high-growth | `1.0 / 0.0` | 0.25 | 46491 | 19.3124 | 8.4596 | 0.6498 | 0.3819 | 局部严重崩塌，不推荐 |
| L1-heavy efficient | `0.9 / 0.1` | 0.14 | 41484 | 22.9697 | 16.3266 | 0.8585 | 0.5438 | 当前效率优先候选 |
| L1-heavy sharp | `0.85 / 0.15` | 0.14 | 45912 | 22.8763 | 15.7534 | 0.8696 | 0.5508 | 锐度高但 PSNR 降 |

静态子集评估，固定排除 `76-93`，共 162 帧：

| 模型 | loss L1/SSIM | growth fraction | splats | PSNR mean | PSNR min | grad sharpness | lap sharpness | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| prefix-180 compact-best | `0.8 / 0.2` | 0.14 | 49648 | 23.6352 | 17.9498 | 0.8753 | 0.5543 | 当前 PSNR/体积折中 |
| L1-only compact | `1.0 / 0.0` | 0.14 | 31238 | 23.4946 | 18.1123 | 0.8249 | 0.5128 | 极小模型，min 更好但更软 |
| L1-only mid | `1.0 / 0.0` | 0.18 | 37760 | 23.5009 | 18.0459 | 0.8252 | 0.5196 | 比 0.14 仅小幅提升 |
| L1-only high-growth | `1.0 / 0.0` | 0.25 | 46491 | 19.4614 | 8.4596 | 0.6265 | 0.3652 | 局部严重崩塌，不推荐 |
| L1-heavy efficient | `0.9 / 0.1` | 0.14 | 41484 | 23.6060 | 18.1274 | 0.8577 | 0.5441 | 当前效率优先候选 |
| L1-heavy sharp | `0.85 / 0.15` | 0.14 | 45912 | 23.5299 | 18.0254 | 0.8684 | 0.5511 | 锐度接近原配置，PSNR 不优 |

当前结论：

1. `L1=0.9 / SSIM=0.1` 是本轮最有价值的新效率候选：相比 `0.8 / 0.2` compact-best，splats 从 `49648` 降到 `41484`，减少约 `16.4%`；full-180 mean 只下降 `0.0316 dB`，static-162 mean 只下降 `0.0292 dB`。
2. `L1=0.9 / SSIM=0.1` 的静态 min PSNR 从 `17.9498` 提升到 `18.1274`，说明它对 worst 静态帧更稳；但 grad/lap sharpness 低于原 compact-best，因此如果主观目标是最锐，仍保留原 `0.8 / 0.2`。
3. `L1-only` 不是默认质量策略。`growth0.14` 和 `growth0.18` 都稳定、很小，但 sharpness 明显低于原 compact-best；`growth0.25` 会出现 frame `35-39` 一类局部严重崩塌，说明 L1-only 下单纯加 densification 预算不安全。
4. `0.85 / 0.15` 没有形成更好折中：锐度更接近原配置，但 full/static PSNR 都低于 `0.9 / 0.1`，splat 数还更多。
5. 当前推荐拆成两档：
   - PSNR / 主观锐度优先：保留 `--loss-l1-weight 0.8 --loss-ssim-weight 0.2 --litegs-growth-select-fraction 0.14`。
   - 效率优先：使用 `--loss-l1-weight 0.9 --loss-ssim-weight 0.1 --litegs-growth-select-fraction 0.14`。

## 17. 参考资料

- Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting, arXiv 2403.15530. <https://arxiv.org/abs/2403.15530>
- AbsGS: Recovering Fine Details for 3D Gaussian Splatting, arXiv 2404.10484. <https://arxiv.org/abs/2404.10484>
- Mip-Splatting: Alias-free 3D Gaussian Splatting, arXiv 2311.16493. <https://arxiv.org/abs/2311.16493>
- Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration, arXiv 2403.11056. <https://arxiv.org/abs/2403.11056>
- BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring, arXiv 2510.12493. <https://arxiv.org/abs/2510.12493>
- SRGS: Super-Resolution 3D Gaussian Splatting, arXiv 2404.10318. <https://arxiv.org/abs/2404.10318>
- MipSLAM: Alias-Free Gaussian Splatting SLAM, arXiv 2603.06989. <https://arxiv.org/abs/2603.06989>
