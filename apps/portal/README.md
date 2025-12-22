# 🌐 Nexus Portal: 数字超我门户

> **应用状态：活跃 | 版本：v1.2.0**

## 1. 应用愿景 (Vision)

Nexus Portal 是整个“数字超我”套件的感知中心和分发中心。它被独立隔离在 `apps/` 目录下，以保持内核与界面的清晰边界。

## 2. 架构决策 (Architecture Decisions)

- **位置隔离**: 从根目录迁移至 `apps/portal/`，标志着“界面即应用”的架构转型。
- **路径规范**: 所有子应用互联均采用相对路径 `../` 进行水平寻址。
- **设计对齐**: 整合并托管 `DESIGN_SYSTEM.md`。

## 3. 技术规格

- **框架**: Alpine.js v3
- **样式**: Tailwind v4 + ODS (Ouroboros Design System)
- **入口**: 动态 App 磁贴分发系统。

---

_Ouroboros Satellite Profile_
