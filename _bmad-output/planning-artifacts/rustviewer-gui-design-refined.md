# RustViewer GUI 精致化设计文档

**项目：** RustScan - RustViewer 3D 可视化工具
**设计风格：** Apple macOS 原生应用级精致度
**创建日期：** 2026-03-06
**版本：** 3.0 (Refined)

---

## 设计哲学

本文档旨在达到 Apple 原生应用（如 Finder、Photos、Music、Xcode）的精致度标准。精致不仅仅是"好看"，而是对每一个像素、每一个间距、每一个过渡动画的精确控制。

### 精致的三个维度

1. **视觉精确性**：精确到 0.5px 的边框，精确到 1px 的间距
2. **交互细腻度**：微妙的状态变化，流畅的过渡动画
3. **系统一致性**：遵循 8pt 网格，统一的设计语言

---

## 1. 间距系统（8pt Grid System）

### 1.1 基础间距单位

```
4px  - 最小间距（元素内部的紧密间距）
8px  - 小间距（相关元素之间）
12px - 中小间距（组内元素）
16px - 标准间距（卡片内边距）
20px - 中等间距（区域内间距）
24px - 大间距（区域之间）
32px - 特大间距（主要区域分隔）
40px - 超大间距（页面级分隔）
48px - 巨大间距（特殊用途）
```

### 1.2 组件内边距规范

```
卡片内边距：16px（所有方向）
按钮内边距：
  - 小按钮：8px 16px
  - 标准按钮：12px 20px
  - 大按钮：16px 24px

列表项内边距：12px 16px
输入框内边距：8px 12px
工具栏内边距：8px
侧边栏内边距：20px
```

### 1.3 元素间距规范

```
标题与内容：12px
段落之间：16px
卡片之间：16px
区域之间：24px
主要区域：32px
```

---

## 2. 精细阴影系统

### 2.1 卡片阴影（三层组合）

```css
/* 静止状态 */
box-shadow:
  0 1px 3px rgba(0, 0, 0, 0.04),   /* 外层：轻微的边缘阴影 */
  0 4px 8px rgba(0, 0, 0, 0.06),   /* 中层：主要深度 */
  0 8px 16px rgba(0, 0, 0, 0.08);  /* 内层：柔和扩散 */

/* 悬停状态 */
box-shadow:
  0 2px 4px rgba(0, 0, 0, 0.06),
  0 8px 16px rgba(0, 0, 0, 0.08),
  0 16px 32px rgba(0, 0, 0, 0.12);

/* 过渡 */
transition: box-shadow 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
```

### 2.2 悬浮元素阴影（工具栏、弹出框）

```css
box-shadow:
  0 2px 8px rgba(0, 0, 0, 0.08),
  0 8px 24px rgba(0, 0, 0, 0.12);
```

### 2.3 按钮阴影

```css
/* 默认 */
box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);

/* 悬停 */
box-shadow: 0 2px 4px rgba(0, 0, 0, 0.16);

/* 按下 */
box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
```

### 2.4 内阴影（用于输入框）

```css
box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.06);
```

---

## 3. 边框系统

### 3.1 边框规范

```css
/* 卡片边框 */
border: 0.5px solid rgba(0, 0, 0, 0.06);

/* 分隔线 */
border: 0.5px solid rgba(0, 0, 0, 0.08);

/* 输入框边框 */
border: 1px solid rgba(0, 0, 0, 0.12);

/* 焦点边框 */
outline: 2px solid #007AFF;
outline-offset: 2px;

/* 悬停边框 */
border: 0.5px solid rgba(0, 0, 0, 0.12);
```

### 3.2 深色模式边框

```css
/* 卡片边框 */
border: 0.5px solid rgba(255, 255, 255, 0.08);

/* 分隔线 */
border: 0.5px solid rgba(255, 255, 255, 0.06);

/* 输入框边框 */
border: 1px solid rgba(255, 255, 255, 0.12);
```

---

## 4. 完整字体系统

### 4.1 字体层级（Typography Scale）

| 名称 | 大小/行高 | 字重 | 用途 | 示例 |
|------|----------|------|------|------|
| Display Large | 34px / 41px | Light 300 | 特大标题 | 欢迎页面主标题 |
| Display | 28px / 34px | Semibold 600 | 大标题 | 页面主标题 |
| Title 1 | 22px / 28px | Semibold 600 | 一级标题 | 区域标题 |
| Title 2 | 17px / 22px | Semibold 600 | 二级标题 | 卡片标题 |
| Title 3 | 15px / 20px | Semibold 600 | 三级标题 | 小节标题 |
| Headline | 13px / 18px | Semibold 600 | 强调文本 | 列表标题 |
| Body | 15px / 20px | Regular 400 | 正文 | 主要内容 |
| Callout | 13px / 18px | Regular 400 | 说明文字 | 辅助说明 |
| Subheadline | 11px / 16px | Regular 400 | 次要文字 | 时间戳 |
| Footnote | 10px / 13px | Regular 400 | 脚注 | 版权信息 |
| Caption 1 | 10px / 13px | Medium 500 | 标注1 | 图片说明 |
| Caption 2 | 10px / 13px | Semibold 600 | 标注2 | 强调标注 |

### 4.2 字体家族

```css
/* 英文 */
font-family: -apple-system, "SF Pro Display", "SF Pro Text", system-ui, sans-serif;

/* 中文 */
font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;

/* 等宽（数据显示）*/
font-family: "SF Mono", "Monaco", "Menlo", monospace;
```

### 4.3 字重使用规范

```
Light 300    - 仅用于超大标题（34px+）
Regular 400  - 正文、说明文字
Medium 500   - 标注、次要强调
Semibold 600 - 标题、按钮、强调
Bold 700     - 特殊强调（谨慎使用）
```

---

## 5. 毛玻璃效果（Vibrancy）

### 5.1 侧边栏毛玻璃

```css
/* 浅色模式 */
background: rgba(246, 246, 246, 0.72);
backdrop-filter: blur(40px) saturate(180%);
border-right: 0.5px solid rgba(0, 0, 0, 0.06);

/* 深色模式 */
background: rgba(28, 28, 30, 0.72);
backdrop-filter: blur(40px) saturate(180%);
border-right: 0.5px solid rgba(255, 255, 255, 0.08);
```

### 5.2 工具栏毛玻璃

```css
/* 浅色模式 */
background: rgba(255, 255, 255, 0.8);
backdrop-filter: blur(20px) saturate(180%);
border: 0.5px solid rgba(0, 0, 0, 0.04);
box-shadow:
  0 2px 8px rgba(0, 0, 0, 0.08),
  0 8px 24px rgba(0, 0, 0, 0.12);

/* 深色模式 */
background: rgba(44, 44, 46, 0.8);
backdrop-filter: blur(20px) saturate(180%);
border: 0.5px solid rgba(255, 255, 255, 0.08);
```

### 5.3 坐标轴指示器毛玻璃

```css
background: rgba(255, 255, 255, 0.9);
backdrop-filter: blur(20px) saturate(180%);
border: 0.5px solid rgba(0, 0, 0, 0.04);
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
```

---

## 6. 完整交互状态

### 6.1 主按钮（Primary Button）

```css
/* Default */
.button-primary {
  background: linear-gradient(180deg, #007AFF 0%, #0051D5 100%);
  border: none;
  border-radius: 8px;
  padding: 12px 20px;
  color: #FFFFFF;
  font-size: 15px;
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
  cursor: pointer;
  transition: all 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Hover */
.button-primary:hover {
  background: linear-gradient(180deg, #0077F0 0%, #004CC8 100%);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.16);
  transform: translateY(-0.5px);
}

/* Active */
.button-primary:active {
  background: linear-gradient(180deg, #0066D6 0%, #0043B0 100%);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  transform: translateY(0px) scale(0.98);
  transition: all 0.1s cubic-bezier(0.55, 0.055, 0.675, 0.19);
}

/* Focus */
.button-primary:focus {
  outline: 2px solid #007AFF;
  outline-offset: 2px;
}

/* Disabled */
.button-primary:disabled {
  background: #F5F5F7;
  color: #D2D2D7;
  opacity: 0.5;
  cursor: not-allowed;
  box-shadow: none;
}
```

### 6.2 次要按钮（Secondary Button）

```css
/* Default */
.button-secondary {
  background: rgba(0, 0, 0, 0.04);
  border: 0.5px solid rgba(0, 0, 0, 0.12);
  border-radius: 8px;
  padding: 12px 20px;
  color: #1D1D1F;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Hover */
.button-secondary:hover {
  background: rgba(0, 0, 0, 0.08);
  border-color: rgba(0, 0, 0, 0.16);
}

/* Active */
.button-secondary:active {
  background: rgba(0, 0, 0, 0.12);
  transform: scale(0.98);
}
```

### 6.3 复选框（Checkbox）

```css
/* Unchecked */
.checkbox {
  width: 18px;
  height: 18px;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  background: #FFFFFF;
  cursor: pointer;
  transition: all 0.15s ease-out;
}

/* Hover */
.checkbox:hover {
  border-color: rgba(0, 0, 0, 0.3);
  box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.08);
}

/* Checked */
.checkbox:checked {
  background: #007AFF;
  border-color: #007AFF;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
}

/* Checked + Hover */
.checkbox:checked:hover {
  background: #0077F0;
  box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.12);
}
```

### 6.4 卡片（Card）

```css
/* Default */
.card {
  background: #FFFFFF;
  border: 0.5px solid rgba(0, 0, 0, 0.06);
  border-radius: 10px;
  padding: 16px;
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.04),
    0 4px 8px rgba(0, 0, 0, 0.06),
    0 8px 16px rgba(0, 0, 0, 0.08);
  transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Hover (可交互卡片) */
.card:hover {
  transform: translateY(-2px);
  border-color: rgba(0, 0, 0, 0.08);
  box-shadow:
    0 2px 4px rgba(0, 0, 0, 0.06),
    0 8px 16px rgba(0, 0, 0, 0.08),
    0 16px 32px rgba(0, 0, 0, 0.12);
}
```

### 6.5 输入框（Input）

```css
/* Default */
.input {
  background: #FFFFFF;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 15px;
  color: #1D1D1F;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.06);
  transition: all 0.15s ease-out;
}

/* Focus */
.input:focus {
  outline: none;
  border-color: #007AFF;
  box-shadow:
    inset 0 1px 2px rgba(0, 0, 0, 0.06),
    0 0 0 3px rgba(0, 122, 255, 0.12);
}

/* Error */
.input.error {
  border-color: #FF3B30;
  box-shadow:
    inset 0 1px 2px rgba(0, 0, 0, 0.06),
    0 0 0 3px rgba(255, 59, 48, 0.12);
}
```

---

## 7. 动画缓动曲线

### 7.1 标准缓动函数

```css
/* 进入动画（元素出现）*/
--ease-out: cubic-bezier(0.25, 0.46, 0.45, 0.94);
transition: all 0.2s var(--ease-out);

/* 退出动画（元素消失）*/
--ease-in: cubic-bezier(0.55, 0.055, 0.675, 0.19);
transition: all 0.15s var(--ease-in);

/* 状态切换 */
--ease-in-out: cubic-bezier(0.645, 0.045, 0.355, 1);
transition: all 0.2s var(--ease-in-out);

/* 弹性动画（按钮按下等）*/
--spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
transition: transform 0.3s var(--spring);
```

### 7.2 动画时长

```
超快：0.1s  - 按钮按下
快速：0.15s - 悬停状态
标准：0.2s  - 一般过渡
中等：0.3s  - 弹性动画
慢速：0.4s  - 大型元素移动
超慢：0.5s  - 页面级过渡
```

---

## 8. SF Symbols 使用规范

### 8.1 图标尺寸

```
Small:   13pt - 用于内联文本
Regular: 17pt - 用于列表项
Medium:  20pt - 用于工具栏
Large:   24pt - 用于主要操作
XLarge:  28pt - 用于特殊强调
```

### 8.2 图标权重

```
Ultralight: 100 - 极细（特殊用途）
Thin:       200 - 细
Light:      300 - 轻
Regular:    400 - 常规（默认）
Medium:     500 - 中等
Semibold:   600 - 半粗
Bold:       700 - 粗
```

### 8.3 图标颜色

```css
/* 主要图标 */
color: #1D1D1F;

/* 次要图标 */
color: #6E6E73;

/* 三级图标 */
color: #98989D;

/* 强调图标 */
color: #007AFF;

/* 成功图标 */
color: #34C759;

/* 警告图标 */
color: #FF9500;

/* 错误图标 */
color: #FF3B30;
```

### 8.4 图标与文字对齐

```css
/* 图标容器 */
.icon-text {
  display: inline-flex;
  align-items: center;
  gap: 6px; /* 图标与文字间距 */
}

/* 图标垂直居中 */
.icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
```

---

## 9. 微交互设计

### 9.1 加载指示器

```css
/* Apple 标准旋转圆环 */
.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(0, 0, 0, 0.1);
  border-top-color: #007AFF;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### 9.2 进度条

```css
.progress-bar {
  height: 2px;
  background: rgba(0, 0, 0, 0.08);
  border-radius: 1px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #007AFF;
  border-radius: 1px;
  transition: width 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
```

### 9.3 工具提示（Tooltip）

```css
.tooltip {
  background: rgba(0, 0, 0, 0.8);
  color: #FFFFFF;
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);

  /* 延迟显示 */
  animation: tooltip-fade-in 0.15s ease-out 0.5s both;
}

@keyframes tooltip-fade-in {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
```

### 9.4 拖拽反馈

```css
/* 拖拽中的元素 */
.dragging {
  opacity: 0.6;
  transform: scale(0.95);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  cursor: grabbing;
}

/* 拖拽目标区域 */
.drop-target {
  background: rgba(0, 122, 255, 0.08);
  border: 2px dashed #007AFF;
  border-radius: 10px;
}
```

---

## 10. 组件精细化设计

### 10.1 侧边栏卡片

```css
.sidebar-card {
  background: #FFFFFF;
  border: 0.5px solid rgba(0, 0, 0, 0.06);
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow:
    0 1px 3px rgba(0, 0, 0, 0.04),
    0 4px 8px rgba(0, 0, 0, 0.06);
}

.sidebar-card-title {
  font-size: 17px;
  font-weight: 600;
  color: #1D1D1F;
  margin-bottom: 12px;
  line-height: 22px;
}

.sidebar-card-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
```

### 10.2 分段控制器（Segmented Control）

```css
.segmented-control {
  display: inline-flex;
  background: #F5F5F7;
  border-radius: 6px;
  padding: 2px;
  gap: 2px;
}

.segment {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 400;
  color: #6E6E73;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease-out;
}

.segment:hover {
  color: #1D1D1F;
}

.segment.active {
  background: #FFFFFF;
  color: #1D1D1F;
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
}
```

### 10.3 列表项

```css
.list-item {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  gap: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s ease-out;
}

.list-item:hover {
  background: rgba(0, 0, 0, 0.04);
}

.list-item:active {
  background: rgba(0, 0, 0, 0.08);
}

.list-item-icon {
  width: 20px;
  height: 20px;
  color: #6E6E73;
}

.list-item-label {
  font-size: 15px;
  color: #1D1D1F;
  flex: 1;
}

.list-item-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
```

### 10.4 统计信息行

```css
.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
}

.stat-label {
  font-size: 13px;
  color: #6E6E73;
}

.stat-value {
  font-size: 13px;
  font-family: "SF Mono", monospace;
  color: #1D1D1F;
  font-weight: 500;
}
```

---

## 11. 深色模式适配

### 11.1 颜色映射

| 浅色模式 | 深色模式 | 用途 |
|---------|---------|------|
| #FFFFFF | #1C1C1E | 窗口背景 |
| #FFFFFF | #2C2C2E | 卡片背景 |
| #F5F5F7 | #000000 | 视口背景 |
| #1D1D1F | #F5F5F7 | 主文字 |
| #6E6E73 | #98989D | 次要文字 |
| #D2D2D7 | #48484A | 禁用文字 |
| rgba(0,0,0,0.04) | rgba(255,255,255,0.08) | 悬停背景 |
| rgba(0,0,0,0.06) | rgba(255,255,255,0.08) | 边框 |

### 11.2 阴影适配

深色模式下阴影需要更明显：

```css
/* 深色模式卡片阴影 */
box-shadow:
  0 1px 3px rgba(0, 0, 0, 0.2),
  0 4px 8px rgba(0, 0, 0, 0.3),
  0 8px 16px rgba(0, 0, 0, 0.4);
```

---

## 12. 响应式断点

```css
/* 最小尺寸 */
@media (min-width: 1280px) and (min-height: 800px) {
  /* 基础布局 */
}

/* 推荐尺寸 */
@media (min-width: 1440px) and (min-height: 900px) {
  /* 优化布局 */
}

/* 大屏幕 */
@media (min-width: 1920px) {
  /* 扩展布局 */
}
```

---

## 13. 辅助功能（Accessibility）

### 13.1 对比度要求

```
正文文字：≥ 4.5:1 (WCAG AA)
大文字（18px+）：≥ 3:1
图标：≥ 3:1
交互元素：≥ 3:1
```

### 13.2 焦点指示器

```css
:focus-visible {
  outline: 2px solid #007AFF;
  outline-offset: 2px;
  border-radius: 4px;
}
```

### 13.3 键盘导航

- Tab: 前进
- Shift+Tab: 后退
- Enter/Space: 激活
- Esc: 取消/关闭

---

## 14. 实现检查清单

### 设计实现时必须检查：

- [ ] 所有间距符合 8pt 网格
- [ ] 所有阴影使用三层组合
- [ ] 所有边框使用 0.5px 半透明
- [ ] 所有过渡使用正确的缓动曲线
- [ ] 所有字体使用正确的字重
- [ ] 所有交互状态都有定义
- [ ] 毛玻璃效果包含 blur 和 saturate
- [ ] 深色模式颜色正确映射
- [ ] 对比度符合 WCAG AA 标准
- [ ] 焦点状态清晰可见

---

**文档版本：** 3.0 (Refined)
**最后更新：** 2026-03-06
**设计标准：** Apple macOS 原生应用级精致度
