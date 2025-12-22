# 🌌 Ouroboros Design System (ODS) v1.0

> **"秩序源于混沌，美感成于克制。"**

Ouroboros 设计系统旨在为所有“数字超我”子应用提供统一、高端、且具有极强视觉冲击力的审美框架。

---

## 1. 核心设计原则 (Core Principles)

- **深邃感 (Depth)**：通过大面积暗色调、微弱渐变和玻璃拟态（Glassmorphism）营造层次。
- **激进排版 (Radical Typography)**：对比强烈的字重，富有冲突感的排版比例。
- **流动性 (Fluidity)**：细腻的微交互和过渡动画，使应用感觉是“活的”。
- **零构建友好 (Zero-Build Ready)**：所有设计元素必须能通过 CDN 资源（如 Tailwind / Shoelace）闭环实现。

---

## 2. 设计令牌 (Design Tokens)

所有项目必须在根部样式中初始化以下变量：

```css
:root {
  /* 基础调色盘 (Base Palette) */
  --ods-bg: #020617; /* Tailwind Slate 950 */
  --ods-surface: rgba(255, 255, 255, 0.03);
  --ods-border: rgba(255, 255, 255, 0.1);

  /* 品牌色 (Accents) */
  --ods-primary: #6366f1; /* Indigo 500 */
  --ods-secondary: #a855f7; /* Purple 500 */
  --ods-accent: #f43f5e; /* Rose 500 - 用于危险或强调 */

  /* 排版 (Typography) */
  --ods-font-display: 'Outfit', sans-serif;
  --ods-font-body: 'Inter', sans-serif;

  /* 玻璃参数 (Glassmorphism) */
  --ods-blur: blur(12px);
  --ods-glass-hover: rgba(255, 255, 255, 0.08);
}
```

---

## 3. 组件规范 (Component Patterns)

### A. 玻璃卡片 (Glass Card)

```html
<div
  class="bg-[var(--ods-surface)] backdrop-blur-[var(--ods-blur)] border border-[var(--ods-border)] rounded-3xl p-6 hover:bg-[var(--ods-glass-hover)] transition-all"
>
  <!-- 内容 -->
</div>
```

### B. 渐变文字 (Gradient Text)

```html
<h1
  class="bg-gradient-to-r from-white/100 to-white/60 bg-clip-text text-transparent"
>
  标题内容
</h1>
```

---

## 4. 资产标准 (Assets Standard)

- **图标库**：推荐使用 Shoelace 内置的 **Lucide** 图标。
- **图片**：优先使用 `generate_image` 产出的高审美图资产，并使用 Base64 转内联。

---

_Created by gg | The Architect of Ouroboros_
