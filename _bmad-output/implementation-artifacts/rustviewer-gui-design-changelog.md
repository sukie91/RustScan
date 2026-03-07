# RustViewer GUI Design Changelog

## 2026-03-07 - Design Refinements (Apple Standards Alignment)

### Overview
Applied four major design improvements to align with Apple Human Interface Guidelines across all 5 application states.

### Changes Implemented

#### 1. Scene Layers Left Alignment
- **Before**: Layer toggles used `justifyContent: "space_between"` with checkboxes on the right
- **After**: Changed to `justifyContent: "flex_start"` for left alignment with checkboxes on the left
- **Rationale**: Follows Apple's standard pattern for list items with leading controls
- **Affected Screens**: All 5 screens (Empty, Loading, Data Loaded, Error, Performance Degradation)

#### 2. Apple-Standard Checkboxes
- **Before**: Text-based checkboxes using Unicode characters (☑/☐)
- **After**: Proper rounded rectangle checkboxes
  - Checked state: Blue rounded rectangle (#007AFF) with white checkmark (✓)
  - Unchecked state: Transparent with gray border (#D1D1D6, 1.5px stroke)
  - Size: 16×16px with 4px corner radius
- **Rationale**: Matches native macOS checkbox design for better visual consistency
- **Affected Screens**: All 5 screens

#### 3. Scene Statistics Consolidation
- **Before**:
  - 4 separate white cards (Keyframes, Map Points, Gaussians, Mesh Vertices)
  - Vertical layout with 8px gaps
  - Total height: ~264px
  - Low information density
- **After**:
  - Consolidated Keyframes, Map Points, and Gaussians into single card
  - Each statistic displayed as horizontal row with label-value pairs
  - Mesh Vertices retained as separate card
  - Reduced vertical space by ~140px
- **Rationale**: Improved information density and reduced visual clutter
- **Affected Screens**: All 5 screens

#### 4. Camera Controls Removal
- **Before**: Dedicated "CAMERA CONTROLS" section with "Auto Fit Scene" button
- **After**: Section completely removed
- **Rationale**: User confirmed the Auto Fit functionality is not needed
- **Affected Screens**: All 5 screens

### Technical Details

**Files Modified**:
- `pencil-new.pen` (Pencil design file - managed by MCP server)

**Node Changes** (Screen 1 example):
- Layer toggles: MPo5V, GZ2uf, VRZvN, FQGoc, Q3Exx
- Checkboxes: y32NN→o9XZT, 0dwYn→KR4a2, wevcQ→eVUKL, NPvTY→l2wVw, oQgw5→lmX53
- Statistics: Deleted xfBmM, 5UvHU, in19G; Created new consolidated card DhaaA
- Camera Controls: Deleted 2cLiL (title), W4Awz (button)

### Visual Impact

**Before**:
- Checkboxes appeared as text characters
- Statistics occupied significant vertical space
- Camera Controls added extra section
- Right-aligned checkboxes felt inconsistent

**After**:
- Native-looking checkboxes with proper styling
- Compact statistics presentation
- Cleaner sidebar with removed Camera Controls
- Left-aligned controls follow Apple patterns

### Consistency

All changes were applied uniformly across all 5 application states:
1. Screen 1 (bi8Au) - Empty State
2. Screen 2 (r9A9P) - Loading State
3. Screen 3 (2Bu0k) - Data Loaded
4. Screen 4 (9HKaP) - Error State
5. Screen 5 (ocEZ5) - Performance Degradation

### Next Steps

Future design improvements to consider:
- Replace emoji icons with SF Symbols-style outline icons
- Add selection state backgrounds (#e0e0e0) for active layer toggles
- Optimize typography hierarchy (increase font weights)
- Adjust spacing/padding for more comfortable feel
