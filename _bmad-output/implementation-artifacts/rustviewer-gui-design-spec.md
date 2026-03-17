# RustViewer GUI Design Specification

**Version:** 1.0
**Date:** 2026-03-06
**Target Platform:** macOS (with cross-platform considerations)
**Design Language:** Apple Human Interface Guidelines

---

## 1. Design Philosophy

### 1.1 Core Principles

RustViewer 遵循 Apple Human Interface Guidelines 的三大核心原则：

- **Clarity（清晰）**
  - 内容优先，界面元素服务于功能
  - 文字清晰易读，图标含义明确
  - 功能层次分明，避免视觉混乱

- **Deference（尊重）**
  - UI 让位于 3D 内容，不喧宾夺主
  - 使用微妙的视觉效果引导用户注意力
  - 动画流畅自然，不干扰工作流程

- **Depth（深度）**
  - 通过层次、阴影、透明度营造空间感
  - 视觉层级帮助用户理解界面结构
  - 交互反馈即时且符合物理直觉

### 1.2 Design Goals

1. **专业而不失亲和**：面向技术用户，但保持易用性
2. **信息密度适中**：展示必要数据，避免信息过载
3. **响应迅速**：所有交互在 100ms 内给出视觉反馈
4. **视觉一致性**：与 macOS 原生应用保持一致的视觉语言

---

## 2. Color System

### 2.1 System Colors (macOS)

采用 macOS 系统色彩，自动适配浅色/深色模式：

#### Primary Colors
```
System Blue (Accent)    #007AFF  rgb(0, 122, 255)
System Gray             #8E8E93  rgb(142, 142, 147)
System Green            #34C759  rgb(52, 199, 89)
System Orange           #FF9500  rgb(255, 149, 0)
System Red              #FF3B30  rgb(255, 59, 48)
```

#### Background Hierarchy (Light Mode)
```
Window Background       #F6F6F6  rgb(246, 246, 246)
Panel Background        #FFFFFF  rgb(255, 255, 255) @ 90% opacity
Card Background         #FFFFFF  rgb(255, 255, 255)
Separator               #D1D1D6  rgb(209, 209, 214) @ 20% opacity
```

#### Background Hierarchy (Dark Mode)
```
Window Background       #1E1E1E  rgb(30, 30, 30)
Panel Background        #282828  rgb(40, 40, 40) @ 90% opacity
Card Background         #323232  rgb(50, 50, 50)
Separator               #38383A  rgb(56, 56, 58) @ 20% opacity
```

#### Text Colors (Light Mode)
```
Primary Text            #000000  rgb(0, 0, 0)
Secondary Text          #8E8E93  rgb(142, 142, 147)
Tertiary Text           #C7C7CC  rgb(199, 199, 204)
Disabled Text           #000000  @ 50% opacity
```

#### Text Colors (Dark Mode)
```
Primary Text            #FFFFFF  rgb(255, 255, 255)
Secondary Text          #98989D  rgb(152, 152, 157)
Tertiary Text           #48484A  rgb(72, 72, 74)
Disabled Text           #FFFFFF  @ 50% opacity
```

### 2.2 3D Scene Colors

#### Visualization Elements
```
Camera Trajectory       #007AFF  rgb(0, 122, 255)      System Blue
Map Points (Near)       #34C759  rgb(52, 199, 89)      System Green
Map Points (Far)        #FF9500  rgb(255, 149, 0)      System Orange
Gaussian Points         #FF9500  rgb(255, 149, 0)      System Orange
Mesh Wireframe          #8E8E93  rgb(142, 142, 147)    System Gray
Mesh Solid              Vertex Color + Lambert Lighting
Grid Lines              #D1D1D6  @ 10% opacity
```

#### Scene Background (Light Mode)
```
Gradient Top            #FAFAFA  rgb(250, 250, 250)
Gradient Bottom         #F0F0F5  rgb(240, 240, 245)
```

#### Scene Background (Dark Mode)
```
Gradient Top            #1A1A1A  rgb(26, 26, 26)
Gradient Bottom         #0F0F0F  rgb(15, 15, 15)
```

---

## 3. Typography

### 3.1 Font Family

- **Primary Font**: SF Pro Text (macOS 系统字体)
- **Monospace Font**: SF Mono (用于数值显示)
- **Chinese Fallback**: PingFang SC (苹方)

### 3.2 Type Scale

```
Large Title     20pt / 28pt line height    Semibold (600)
Title           17pt / 22pt line height    Semibold (600)
Headline        15pt / 20pt line height    Semibold (600)
Body            13pt / 18pt line height    Regular (400)
Callout         12pt / 16pt line height    Regular (400)
Subheadline     11pt / 14pt line height    Regular (400)
Footnote        10pt / 13pt line height    Regular (400)
Caption         9pt / 11pt line height     Regular (400)
```

### 3.3 Font Weights

```
Regular         400     正文、标签、描述
Medium          500     按钮、强调文本
Semibold        600     标题、section header
Bold            700     数值、重要信息（谨慎使用）
```

### 3.4 Usage Guidelines

- **标题**：使用 Semibold，颜色为 Primary Text
- **正文**：使用 Regular，颜色为 Primary Text
- **辅助信息**：使用 Regular，颜色为 Secondary Text
- **数值**：使用 SF Mono Medium，颜色为 Primary Text
- **禁用状态**：使用 50% opacity

---

## 4. Spacing & Layout

### 4.1 Grid System

采用 **8pt 网格系统**，所有尺寸和间距为 8 的倍数：

```
XXS     4pt     极小间距（分隔线、图标内边距）
XS      8pt     小间距（图标与文字、checkbox 与 label）
SM      12pt    中小间距（列表项内部元素）
MD      16pt    标准间距（section 之间）
LG      24pt    大间距（panel padding）
XL      32pt    超大间距（空状态、页面级间距）
XXL     48pt    特大间距（空状态垂直间距）
```

### 4.2 Layout Dimensions

#### Window
```
Default Size            1280 × 800 pt
Minimum Size            960 × 600 pt
Title Bar Height        52 pt (macOS 标准)
```

#### Side Panel
```
Width                   280 pt (固定)
Padding (Horizontal)    24 pt
Padding (Vertical)      20 pt
Section Spacing         16 pt
```

#### Central Viewport
```
Minimum Width           680 pt
Padding                 0 pt (全屏显示 3D 内容)
```

#### Component Heights
```
Button (Small)          28 pt
Button (Standard)       32 pt
Button (Large)          40 pt
List Item (Single)      32 pt
List Item (Subtitle)    44 pt
Input Field             32 pt
Checkbox                16 pt
Toggle Switch           24 pt
```

### 4.3 Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  Title Bar (52pt)                                       │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  Side Panel  │         3D Viewport                      │
│   (280pt)    │         (Flexible)                       │
│              │                                          │
│              │                                          │
│              │                                          │
│              │                                          │
│              │                                          │
│              │                                          │
└──────────────┴──────────────────────────────────────────┘
```

---

## 5. Visual Effects

### 5.1 Corner Radius

```
Panel / Card            8pt
Button                  6pt
Input Field             6pt
Badge / Tag             4pt
Checkbox                3pt
```

### 5.2 Shadows

#### Panel Shadow (Floating)
```
Offset      (0, 2)
Blur        8pt
Spread      0pt
Color       rgba(0, 0, 0, 0.10) Light Mode
            rgba(0, 0, 0, 0.30) Dark Mode
```

#### Card Shadow
```
Offset      (0, 1)
Blur        4pt
Spread      0pt
Color       rgba(0, 0, 0, 0.08) Light Mode
            rgba(0, 0, 0, 0.20) Dark Mode
```

#### Button Shadow (Hover)
```
Offset      (0, 1)
Blur        3pt
Spread      0pt
Color       rgba(0, 0, 0, 0.12) Light Mode
            rgba(0, 0, 0, 0.25) Dark Mode
```

### 5.3 Opacity

```
Panel Background        90%
Disabled State          50%
Secondary Text          70%
Tertiary Text           50%
Separator Line          20%
Hover Overlay           10%
```

### 5.4 Blur Effects

由于 egui 不原生支持背景模糊（backdrop-filter），使用以下替代方案：

- **Panel**: 90% opacity 白色/深色背景 + 微妙阴影
- **Modal**: 80% opacity 背景 + 强阴影
- **Tooltip**: 95% opacity 背景 + 轻阴影

---

## 6. Component Library

### 6.1 Buttons

#### Primary Button
```
Background      System Blue (#007AFF)
Text Color      White
Height          32pt
Padding         12pt (horizontal), 6pt (vertical)
Corner Radius   6pt
Font            13pt SF Pro Text Medium

States:
  Hover         Background → #0066CC (darker 10%)
                Shadow → (0, 1, 3, rgba(0,0,0,0.12))
  Active        Background → #0055AA (darker 20%)
                Scale → 0.98
  Disabled      Opacity → 50%
```

#### Secondary Button
```
Background      Transparent
Border          1pt System Gray
Text Color      Primary Text
Height          32pt
Padding         12pt (horizontal), 6pt (vertical)
Corner Radius   6pt
Font            13pt SF Pro Text Medium

States:
  Hover         Background → rgba(0, 122, 255, 0.10)
  Active        Background → rgba(0, 122, 255, 0.20)
  Disabled      Opacity → 50%
```

#### Icon Button
```
Size            32 × 32 pt
Icon Size       16 × 16 pt
Background      Transparent
Corner Radius   6pt

States:
  Hover         Background → rgba(0, 0, 0, 0.05) Light
                Background → rgba(255, 255, 255, 0.10) Dark
  Active        Background → rgba(0, 0, 0, 0.10) Light
                Background → rgba(255, 255, 255, 0.15) Dark
```

### 6.2 Layer Toggle

```
┌─────────────────────────────────┐
│  ●  Camera Trajectory      ☑    │  Height: 32pt
└─────────────────────────────────┘

Components:
  Color Badge     12 × 12 pt circle, left margin 8pt
  Label           13pt SF Pro Text Regular
  Checkbox        16 × 16 pt, right aligned
  Spacing         8pt between badge and label
```

### 6.3 Stat Card

```
┌─────────────────────────────────┐
│  Keyframes                      │  Background: Card Background
│  100                            │  Corner Radius: 8pt
└─────────────────────────────────┘  Padding: 12pt
                                     Shadow: Card Shadow

Components:
  Label           11pt SF Pro Text Regular, Secondary Text
  Value           15pt SF Pro Text Semibold, Primary Text
  Spacing         4pt between label and value
```

### 6.4 Section Header

```
SCENE LAYERS                        Font: 11pt SF Pro Text Semibold
                                    Color: Secondary Text
                                    Letter Spacing: 0.5pt
                                    Transform: Uppercase
                                    Margin Bottom: 8pt
```

### 6.5 Divider

```
Height          1pt
Color           Separator Color (20% opacity)
Margin          16pt (vertical)
```

### 6.6 Empty State

```
┌─────────────────────────────────┐
│                                 │
│         [Icon 64×64]            │  Icon: System Gray @ 50% opacity
│                                 │  Title: 20pt SF Pro Text Semibold
│    No SLAM Data Loaded          │  Description: 13pt Regular, Secondary
│                                 │  Button: Primary Button, 120pt width
│  Load checkpoint, Gaussian,     │
│  or mesh files to visualize     │  Vertical Spacing:
│                                 │    Icon → Title: 16pt
│         [Open Files]            │    Title → Description: 8pt
│                                 │    Description → Button: 24pt
└─────────────────────────────────┘
```

### 6.7 Error Toast

```
┌─────────────────────────────────┐
│  ⚠ Failed to load file          │  Background: System Red @ 95%
└─────────────────────────────────┘  Text: White, 13pt Medium
                                     Corner Radius: 8pt
Position: Top-center, 16pt from top  Padding: 12pt (H), 10pt (V)
Duration: 4 seconds auto-dismiss     Shadow: (0, 4, 12, rgba(0,0,0,0.2))
```

---

## 7. Interface Layout

### 7.1 Side Panel Structure

```
┌────────────────────────────────┐
│  FILE OPERATIONS               │  ← Section Header
│  ┌──────────────────────────┐  │
│  │  📂 Load Checkpoint      │  │  ← Primary Button
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  ✨ Load Gaussians       │  │  ← Primary Button
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  🔷 Load Mesh            │  │  ← Primary Button
│  └──────────────────────────┘  │
│                                │
│  ────────────────────────────  │  ← Divider (16pt margin)
│                                │
│  SCENE LAYERS                  │  ← Section Header
│  ●  Camera Trajectory     ☑   │  ← Layer Toggle
│  ●  Map Points            ☑   │
│  ●  Gaussians             ☑   │
│  ●  Mesh Wireframe        ☐   │
│  ●  Mesh Solid            ☑   │
│                                │
│  ────────────────────────────  │  ← Divider
│                                │
│  SCENE STATISTICS              │  ← Section Header
│  ┌──────────────────────────┐  │
│  │  Keyframes               │  │  ← Stat Card
│  │  100                     │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  Map Points              │  │  ← Stat Card
│  │  15,234                  │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  Gaussians               │  │  ← Stat Card
│  │  100,000                 │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │  Mesh Vertices           │  │  ← Stat Card
│  │  102,924                 │  │
│  └──────────────────────────┘  │
│                                │
│  ────────────────────────────  │  ← Divider
│                                │
│  CAMERA CONTROLS               │  ← Section Header
│  ┌──────────────────────────┐  │
│  │  🎯 Auto Fit Scene       │  │  ← Secondary Button
│  └──────────────────────────┘  │
│                                │
└────────────────────────────────┘
```

### 7.2 3D Viewport

```
┌────────────────────────────────────────────┐
│                                            │
│                                            │
│                                            │
│              3D Scene Content              │
│                                            │
│                                            │
│                                            │
│                                            │
│                                            │
│                                  ┌───────┐ │
│                                  │ X Y Z │ │  ← Axis Indicator
│                                  └───────┘ │     (80×80pt, bottom-right)
└────────────────────────────────────────────┘

Background: Gradient (top to bottom)
Grid: 1pt lines, 10% opacity, 1m spacing
Axis: X=Red, Y=Green, Z=Blue
```

### 7.3 Empty State (No Data Loaded)

```
┌────────────────────────────────────────────┐
│                                            │
│                                            │
│                                            │
│              [Folder Icon 64×64]           │
│                                            │
│           No SLAM Data Loaded              │
│                                            │
│    Load checkpoint, Gaussian, or mesh      │
│       files to visualize 3D results        │
│                                            │
│          ┌──────────────────┐              │
│          │   Open Files     │              │
│          └──────────────────┘              │
│                                            │
│                                            │
│                                            │
└────────────────────────────────────────────┘
```

---

## 8. Interaction Design

### 8.1 Mouse Interactions

#### 3D Viewport
```
Left Click + Drag       Orbit camera (arcball rotation)
Right Click + Drag      Pan camera (translate target)
Scroll Wheel            Zoom in/out (adjust distance)
Double Click            Auto-fit scene to view
```

#### UI Elements
```
Button Hover            Background color change + shadow
Button Click            Scale 0.98 + color darken
Checkbox Click          Toggle with 150ms animation
Layer Toggle            Instant visibility change
```

### 8.2 Keyboard Shortcuts

```
Cmd + O                 Open file dialog
Cmd + W                 Close window
Cmd + Q                 Quit application
Space                   Toggle all layers on/off
F                       Auto-fit scene to view
1-5                     Toggle individual layers
Cmd + ,                 Open preferences (future)
```

### 8.3 State Feedback

#### Loading State
```
Visual: Spinner (20×20pt) + "Loading..." text
Position: Center of viewport
Duration: Until load completes
```

#### Success State
```
Visual: Green checkmark icon + success message
Position: Top-center toast
Duration: 2 seconds auto-dismiss
```

#### Error State
```
Visual: Red warning icon + error message
Position: Top-center toast
Duration: 4 seconds auto-dismiss
Color: System Red @ 95% opacity
```

#### Hover State
```
Transition: 150ms ease-out
Effect: Background color change + shadow
Cursor: Pointer for clickable elements
```

#### Active State
```
Transition: 100ms ease-out
Effect: Scale 0.98 + color darken
Duration: While mouse button pressed
```

#### Disabled State
```
Opacity: 50%
Cursor: not-allowed
Interaction: None
```

### 8.4 Animations

#### Transition Timing
```
Fast            100ms       Button press, checkbox toggle
Standard        150ms       Hover effects, color changes
Slow            250ms       Panel slide, modal fade
```

#### Easing Functions
```
ease-out        Default for most transitions
ease-in-out     Modal open/close
linear          Progress indicators
```

#### Animation Examples
```rust
// Button hover animation
ctx.animate_bool_with_time(
    egui::Id::new("button_hover"),
    response.hovered(),
    0.15  // 150ms
);

// Panel slide animation
ctx.animate_value_with_time(
    egui::Id::new("panel_offset"),
    target_offset,
    0.25  // 250ms
);
```

---

## 9. Accessibility

### 9.1 Color Contrast

遵循 WCAG 2.1 AA 标准：

```
Normal Text (13pt+)     4.5:1 minimum contrast ratio
Large Text (18pt+)      3:1 minimum contrast ratio
UI Components           3:1 minimum contrast ratio
```

### 9.2 Focus Indicators

```
Keyboard Focus          2pt System Blue outline
Focus Offset            2pt from element edge
Corner Radius           Match element + 2pt
```

### 9.3 Text Sizing

```
Minimum Size            11pt (Subheadline)
Body Text               13pt (Body)
Support Dynamic Type    Yes (future enhancement)
```

### 9.4 Screen Reader Support

```
Button Labels           Clear, descriptive text
Icon Buttons            Alt text provided
Status Messages         Announced to screen reader
Error Messages          Announced with alert role
```

---

## 10. Dark Mode

### 10.1 Automatic Switching

- 自动跟随 macOS 系统设置
- 使用 `eframe::Theme::Auto`
- 无需用户手动切换

### 10.2 Color Adjustments

#### Background
```
Light Mode              Bright, subtle gradients
Dark Mode               Deep, reduced contrast
```

#### Text
```
Light Mode              Black on white
Dark Mode               White on dark gray (not pure black)
```

#### Shadows
```
Light Mode              Subtle, 10-20% opacity
Dark Mode               Stronger, 20-30% opacity
```

#### 3D Scene
```
Light Mode              Light gray gradient background
Dark Mode               Dark gray gradient background
Grid Lines              Adjust opacity for visibility
```

---

## 11. Responsive Behavior

### 11.1 Window Resizing

```
Minimum Width           960pt
Minimum Height          600pt
Side Panel              Fixed 280pt width
Viewport                Flexible, fills remaining space
```

### 11.2 Content Adaptation

```
< 1024pt width          Stat cards stack vertically
< 800pt width           Side panel collapses (future)
< 600pt height          Reduce vertical spacing
```

---

## 12. Implementation Notes

### 12.1 egui Configuration

```rust
// 在 ViewerApp::new() 中配置
pub fn configure_apple_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // 字体大小
    style.text_styles = [
        (egui::TextStyle::Heading,
         egui::FontId::new(20.0, egui::FontFamily::Proportional)),
        (egui::TextStyle::Body,
         egui::FontId::new(13.0, egui::FontFamily::Proportional)),
        (egui::TextStyle::Button,
         egui::FontId::new(13.0, egui::FontFamily::Proportional)),
        (egui::TextStyle::Small,
         egui::FontId::new(11.0, egui::FontFamily::Proportional)),
    ].into();

    // 间距（8pt 网格）
    style.spacing.item_spacing = egui::Vec2::new(8.0, 8.0);
    style.spacing.button_padding = egui::Vec2::new(12.0, 6.0);
    style.spacing.indent = 16.0;

    // 视觉效果
    let mut visuals = if ctx.style().visuals.dark_mode {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    // 圆角
    visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
    visuals.widgets.inactive.rounding = egui::Rounding::same(6.0);
    visuals.widgets.hovered.rounding = egui::Rounding::same(6.0);
    visuals.widgets.active.rounding = egui::Rounding::same(6.0);

    // 颜色
    visuals.widgets.inactive.weak_bg_fill =
        egui::Color32::from_rgb(0, 122, 255);
    visuals.widgets.hovered.weak_bg_fill =
        egui::Color32::from_rgb(0, 110, 230);

    // 阴影
    visuals.window_shadow = egui::epaint::Shadow {
        offset: egui::Vec2::new(0.0, 2.0),
        blur: 8.0,
        spread: 0.0,
        color: egui::Color32::from_black_alpha(25),
    };

    style.visuals = visuals;
    ctx.set_style(style);
}
```

### 12.2 Custom Widget Library

创建 `RustViewer/src/ui/apple_widgets.rs` 实现自定义组件。

### 12.3 File Structure

```
RustViewer/src/ui/
├── mod.rs              # UI module exports
├── apple_widgets.rs    # Custom Apple-style widgets
├── panel.rs            # Side panel implementation
├── viewport.rs         # 3D viewport + empty state
└── theme.rs            # Theme colors and utilities
```

---

## 13. Design Assets

### 13.1 Icons

使用 SF Symbols 风格图标（Unicode 或自定义）：

```
📂  Folder              File operations
✨  Sparkles            Gaussian points
🔷  Diamond             Mesh
🎯  Target              Auto-fit camera
⚠  Warning             Error messages
✓  Checkmark           Success messages
```

### 13.2 Color Swatches

提供设计工具色板（Sketch/Figma）：

```
System Blue     #007AFF
System Gray     #8E8E93
System Green    #34C759
System Orange   #FF9500
System Red      #FF3B30
```

---

## 14. Future Enhancements

### Phase 2 Features

- **Collapsible Side Panel**: 窗口宽度 < 800pt 时自动收起
- **Toolbar**: 顶部工具栏，快速访问常用功能
- **Inspector Panel**: 右侧面板，显示选中对象详情
- **Timeline**: 底部时间轴，回放 SLAM 过程
- **Preferences Window**: 设置窗口，自定义颜色、快捷键等
- **Export Dialog**: 导出对话框，保存截图、视频等

---

## 15. Design Checklist

实现时确保：

- [ ] 所有间距为 8pt 的倍数
- [ ] 所有圆角符合规范（8pt/6pt/4pt/3pt）
- [ ] 所有颜色使用系统色或规范中定义的颜色
- [ ] 所有文字大小符合 Type Scale
- [ ] 所有按钮高度符合规范（28pt/32pt/40pt）
- [ ] 所有交互有视觉反馈（hover/active/disabled）
- [ ] 所有动画使用规范的时长和缓动函数
- [ ] 支持浅色/深色模式自动切换
- [ ] 所有文字对比度符合 WCAG AA 标准
- [ ] 所有可交互元素有键盘焦点指示器

---

**End of Document**
