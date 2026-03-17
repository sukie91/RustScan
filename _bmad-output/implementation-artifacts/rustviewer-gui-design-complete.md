# RustViewer 完整 GUI 设计文档

**版本：** 2.0
**日期：** 2026-03-07
**设计工具：** Pencil MCP
**目标平台：** macOS (with cross-platform considerations)
**设计语言：** Apple Human Interface Guidelines

---

## 0. 文档说明

### 0.1 文档目的

本文档提供 RustViewer 的完整 GUI 设计规范，包括所有应用状态的可视化设计稿。本文档旨在：

1. 为开发团队提供清晰的视觉参考
2. 定义所有应用状态的 UI 表现
3. 确保设计的一致性和可执行性
4. 指导后续的实现工作

### 0.2 目标读者

- **UI/UX 设计师**：了解设计理念和视觉规范
- **前端开发者**：实现界面和交互逻辑
- **产品经理**：理解用户体验和功能流程
- **测试工程师**：验证界面实现的正确性

### 0.3 设计文件

- **Pencil 设计文件**：`pencil-new.pen`
- **设计稿数量**：5 个独立 screens
- **设计规范参考**：`rustviewer-gui-design-spec.md`

---

## 1. 设计概览

### 1.1 应用状态系统

RustViewer 包含以下 5 个关键应用状态：

| 状态 | 触发条件 | 主要特征 |
|------|---------|---------|
| **空状态** | 应用启动，未加载数据 | 3D 视口显示引导信息 |
| **加载状态** | 用户点击加载按钮 | 侧边栏顶部显示进度条 |
| **数据已加载** | 文件加载成功 | 显示场景内容和统计信息 |
| **错误状态** | 文件加载失败 | 侧边栏顶部显示红色错误提示 |
| **性能降级** | 数据量超过阈值 | 侧边栏顶部显示橙色警告提示 |

### 1.2 设计原则

遵循 Apple Human Interface Guidelines 的三大核心原则：

- **Clarity（清晰）**：内容优先，界面元素服务于功能
- **Deference（尊重）**：UI 让位于 3D 内容，不喧宾夺主
- **Depth（深度）**：通过层次、阴影、透明度营造空间感

---

## 2. Screen 1: 空状态（Empty State）

### 2.1 触发条件

- 应用启动时
- 未加载任何 SLAM 数据

### 2.2 UI 表现

**侧边栏：**
- 所有加载按钮可用（蓝色）
- 图层开关禁用（灰色）
- 统计卡片显示数值
- 自动对焦按钮可用

**3D 视口：**
- 中央显示空状态提示
- 图标：📂（64×64pt，灰色 50% opacity）
- 标题："No SLAM Data Loaded"（20pt Semibold）
- 描述："Load checkpoint, Gaussian, or mesh files to visualize 3D results"（13pt Regular）
- 按钮："Open Files"（Primary Button，120pt 宽）

### 2.3 设计要点

- 引导用户进行首次操作
- 清晰说明支持的文件格式
- 提供明确的行动号召（CTA）

---

## 3. Screen 2: 加载状态（Loading State）

### 3.1 触发条件

- 用户点击加载按钮
- 文件正在后台加载

### 3.2 UI 表现

**侧边栏：**
- 顶部显示加载指示器：
  - 蓝色进度条（3pt 高，System Blue）
  - 文本："Loading checkpoint.json..."（12pt Regular）
- "Load Checkpoint" 按钮禁用（50% opacity）
- 其他控件保持可用

**3D 视口：**
- 保持空状态显示

### 3.3 设计要点

- 提供即时的加载反馈
- 明确显示正在加载的文件名
- 禁用相关按钮防止重复操作

---

## 4. Screen 3: 数据已加载状态（Data Loaded State）

### 4.1 触发条件

- 文件加载成功
- 场景包含有效数据

### 4.2 UI 表现

**侧边栏：**
- 所有按钮可用
- 图层开关全部启用并选中
- 统计卡片显示实际数值：
  - Keyframes: 100
  - Map Points: 15,234
  - Gaussians: 100,000
  - Mesh Vertices: 102,924

**3D 视口：**
- 显示场景内容：
  - 蓝色轨迹线（相机轨迹）
  - 绿色点（地图点）
  - 橙色点（Gaussian 点云）
  - 灰色线框（Mesh）
- 右下角显示坐标轴指示器
- 相机调试信息：
  - yaw: 45.0°
  - pitch: 30.0°
  - dist: 5.23

### 4.3 设计要点

- 清晰展示场景数据
- 提供实时的相机状态信息
- 使用颜色区分不同类型的数据

---

## 5. Screen 4: 错误状态（Error State）

### 5.1 触发条件

- 文件加载失败
- 文件格式错误
- 权限不足等错误

### 5.2 UI 表现

**侧边栏：**
- 顶部显示错误提示框：
  - 背景：System Red (#FF3B30) @ 10% opacity
  - 边框：System Red 1pt
  - 圆角：6pt
  - 内边距：12pt
  - 内容：⚠️ + "文件未找到：checkpoint.json" + 关闭按钮（×）
- 其他元素保持可用

**3D 视口：**
- 保持空状态显示

### 5.3 设计要点

- 清晰传达错误信息
- 提供关闭按钮让用户手动清除错误
- 错误不会清除已加载的数据
- 使用红色强调错误的严重性

### 5.4 错误类型及消息

```
文件不存在：        "文件未找到：{filename}"
文件格式错误：      "文件格式不支持：{filename}"
JSON 解析错误：     "JSON 格式错误：{error_detail}"
OBJ 解析错误：      "OBJ 格式错误：{error_detail}"
PLY 解析错误：      "PLY 格式错误：{error_detail}"
内存不足：          "内存不足，无法加载文件"
文件过大：          "文件过大（>{size}MB），建议使用简化版本"
权限错误：          "无法读取文件：权限不足"
```

---

## 6. Screen 5: 性能降级状态（Performance Degradation）

### 6.1 触发条件

数据量超过阈值：
- Gaussian 数量 > 1,000,000
- Mesh 顶点 > 10,000,000
- Map Points > 1,000,000

### 6.2 UI 表现

**侧边栏：**
- 顶部显示警告提示框：
  - 背景：System Orange (#FF9500) @ 10% opacity
  - 圆角：6pt
  - 内边距：12pt
  - 内容：ℹ️ + "数据量较大，已自动降低渲染质量以保持流畅"
- Gaussians 统计卡片显示：
  - "1,234,567 (显示 617,283)"

**3D 视口：**
- 显示降级后的场景内容
- 保持流畅的交互性能

### 6.3 设计要点

- 主动告知用户性能优化策略
- 显示实际数量和渲染数量
- 使用橙色表示警告（非错误）
- 保持应用的可用性

### 6.4 自动优化策略

- **Gaussian**：降低点大小（4px → 2px）
- **Mesh**：自动切换为线框模式
- **Map Points**：按距离采样（只显示 50%）

---

## 7. 视觉系统规范

### 7.1 色彩系统

**主色调：**
- System Blue (#007AFF) - 主按钮、强调元素
- System Gray (#8E8E93) - 次要文本、禁用状态
- System Green (#34C759) - 成功提示、地图点
- System Orange (#FF9500) - 警告提示、Gaussian
- System Red (#FF3B30) - 错误提示

**背景色：**
- Window Background: #F6F6F6
- Panel Background: #FFFFFF @ 90% opacity
- Card Background: #FFFFFF

**文本色：**
- Primary Text: #000000
- Secondary Text: #8E8E93
- Disabled Text: #000000 @ 50% opacity

### 7.2 排版系统

**字体：** Inter（Pencil 默认，接近 SF Pro）

**字号：**
- 标题：20pt Semibold
- 正文：13pt Regular
- 辅助：11pt Regular
- 小字：10pt Regular

### 7.3 间距系统

**8pt 网格系统：**
- 按钮高度：32pt
- 卡片内边距：12pt
- 区域间距：16pt
- 侧边栏内边距：24pt（水平）、20pt（垂直）

### 7.4 视觉效果

**圆角：**
- 卡片：8pt
- 按钮：6pt
- 提示框：6pt

**透明度：**
- 侧边栏：90%
- 禁用状态：50%
- 错误背景：10%
- 警告背景：10%

---

## 8. 组件库

### 8.1 Primary Button

```
背景：System Blue (#007AFF)
文字：White
高度：32pt
内边距：12pt (H), 6pt (V)
圆角：6pt
字体：13pt Medium
```

### 8.2 图层开关（Layer Toggle）

```
高度：32pt
布局：水平
内容：颜色标识（12×12 圆形）+ 标签 + Checkbox
间距：8pt
对齐：space-between
```

### 8.3 统计卡片（Stat Card）

```
背景：White
圆角：8pt
内边距：12pt
高度：60pt
布局：垂直
内容：标签（11pt Secondary）+ 数值（15pt Semibold）
间距：4pt
```

### 8.4 错误提示框（Error Alert）

```
背景：System Red @ 10% opacity
边框：System Red 1pt
圆角：6pt
内边距：12pt
布局：水平
内容：⚠️ 图标 + 错误消息 + 关闭按钮
间距：8pt
```

### 8.5 警告提示框（Warning Alert）

```
背景：System Orange @ 10% opacity
圆角：6pt
内边距：12pt
布局：水平
内容：ℹ️ 图标 + 警告消息
间距：8pt
```

### 8.6 加载指示器（Loading Indicator）

```
进度条高度：3pt
进度条颜色：System Blue
文本：12pt Regular
布局：垂直
间距：6pt
```

---

## 9. 交互设计

### 9.1 文件加载流程

```
用户操作                    系统响应                        UI 反馈
─────────────────────────────────────────────────────────────────
1. 点击"加载 Checkpoint"   → 打开文件选择对话框           → 按钮变为禁用状态
2. 选择文件并确认           → 后台线程开始加载             → 显示加载指示器
3. 加载成功                 → 更新场景数据                 → 清除加载指示器
                            → 自动适配相机                 → 更新统计卡片
                            → 启用相关图层开关             → 显示场景内容
4. 加载失败                 → 保留已有数据                 → 显示错误提示
                            → 记录错误日志                 → 按钮恢复可用
```

### 9.2 鼠标交互

**3D 视口：**
- 左键拖拽：旋转视角（Arcball rotation）
- 右键拖拽：平移视角（Pan）
- 滚轮：缩放（Zoom in/out）

**UI 元素：**
- 按钮悬停：背景色变化
- 按钮点击：缩放至 0.98
- Checkbox 点击：切换状态

### 9.3 键盘快捷键（建议）

```
Cmd+O              打开文件对话框
Cmd+W              关闭窗口
Cmd+Q              退出应用
Space              切换所有图层开关
F                  自动对焦场景
1-5                切换单个图层
```

---

## 10. 实现指南

### 10.1 技术栈

- **GUI 框架**：egui 0.31
- **渲染后端**：wgpu (via eframe)
- **字体**：Inter（系统默认）
- **平台**：macOS (primary), Windows/Linux (secondary)

### 10.2 文件结构建议

```
RustViewer/src/ui/
├── mod.rs              # UI 模块导出
├── theme.rs            # Apple HIG 主题配置
├── components.rs       # 自定义组件（加载指示器、Toast 等）
├── panel.rs            # 侧边栏 UI 实现
├── viewport.rs         # 视口覆盖层和空状态
└── keyboard.rs         # 键盘快捷键处理
```

### 10.3 状态管理

```rust
pub enum AppState {
    Empty,              // 空状态
    Loading(String),    // 加载状态（文件名）
    Loaded,             // 数据已加载
    Error(String),      // 错误状态（错误消息）
    Degraded,           // 性能降级
}
```

### 10.4 egui 主题配置

```rust
pub fn configure_apple_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // 字体大小
    style.text_styles = [
        (egui::TextStyle::Heading,
         egui::FontId::new(20.0, egui::FontFamily::Proportional)),
        (egui::TextStyle::Body,
         egui::FontId::new(13.0, egui::FontFamily::Proportional)),
        (egui::TextStyle::Small,
         egui::FontId::new(11.0, egui::FontFamily::Proportional)),
    ].into();

    // 间距（8pt 网格）
    style.spacing.item_spacing = egui::Vec2::new(8.0, 8.0);
    style.spacing.button_padding = egui::Vec2::new(12.0, 6.0);

    // 颜色
    let mut visuals = if ctx.style().visuals.dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.widgets.inactive.weak_bg_fill =
        egui::Color32::from_rgb(0, 122, 255);

    style.visuals = visuals;
    ctx.set_style(style);
}
```

---

## 11. 验证清单

### 11.1 设计完整性

- [x] 所有 5 个关键状态都有对应的设计稿
- [x] 每个状态都有明确的触发条件
- [x] 每个状态都有详细的 UI 表现说明
- [x] 所有交互场景都有清晰的流程定义

### 11.2 视觉一致性

- [x] 所有 screens 使用相同的颜色系统
- [x] 所有 screens 使用相同的字体和字号
- [x] 所有 screens 使用相同的间距系统
- [x] 所有组件样式在所有状态下保持一致

### 11.3 可执行性

- [x] 所有设计都有具体的数值（尺寸、颜色、时长）
- [x] 所有交互都有明确的触发条件和响应
- [x] 所有组件都可以用 egui 实现
- [x] 提供了详细的实现指南和代码示例

---

## 12. 下一步行动

### 12.1 设计阶段（已完成）

- [x] 创建 5 个应用状态的设计稿
- [x] 导出所有设计稿的截图
- [x] 编写完整的设计文档

### 12.2 实现阶段（待进行）

1. **P0（必须实现）**
   - 实现空状态 UI
   - 实现数据已加载状态 UI
   - 实现基本错误提示
   - 实现侧边栏布局
   - 实现图层开关交互
   - 实现相机控制交互

2. **P1（应该实现）**
   - 实现加载状态指示器
   - 实现详细错误类型区分
   - 实现键盘快捷键
   - 优化统计卡片样式
   - 优化相机调试信息

3. **P2（可选实现）**
   - 实现性能降级状态
   - 添加设置窗口
   - 添加截图功能
   - 添加动画和过渡效果
   - 增强无障碍支持

---

## 13. 参考资料

### 13.1 设计规范

- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [Material Design](https://material.io/design)

### 13.2 技术文档

- [egui Documentation](https://docs.rs/egui/)
- [wgpu Documentation](https://docs.rs/wgpu/)

### 13.3 项目文档

- `rustviewer-gui-design-spec.md` - 初始设计规范
- `rustviewer-gui-design-review.md` - 设计评审报告
- `pencil-new.pen` - Pencil 设计文件

---

**文档结束**

*本文档由 Claude Code 使用 Pencil MCP 工具创建*
*最后更新：2026-03-07*
