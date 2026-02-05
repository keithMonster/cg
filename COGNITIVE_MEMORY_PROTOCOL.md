# 🧠 Cognitive Memory Protocol (认知记忆协议)

> **Status**: Active | **Architecture**: Distributed Kernel | **Last Updated**: 2026-02-02
>
> **摘要**: 本文档是 Project Ouroboros (gg) 的**完整记忆结晶**。它不仅定义了记忆系统的拓扑结构，还内置了当前时间点所有核心记忆、原则与偏好的**高保真快照**。

---

## 1. System Topology (系统拓扑)

本目录是 AI Agent 的外部扩展大脑。所有行为必须索引至此。

- **Kernel (内核层)**: `thinking/mental_models.md`
  - _The Constitution_. 不可违背的第一性原理。
- **Userland (应用层)**:
  - `profile/communication.md`: 交互协议。
  - `work/tech_axioms.md`: 工程公理。
- **Inbox (暂存区)**: `scratchpad/`
  - 待处理的 Context 碎片。

## 2. I/O Protocol (读写协议)

### INPUT: Context First

- **Mandate**: 任务开始前，必须挂载 Kernel 与 Axioms。
- **Reference**: 必须显式引用 Kernel 中的公理来解释决策。

### OUTPUT: The DES Filter

仅当满足以下 **DES 标准** 时，才允许申请写入本目录：

1.  **D**urable (持久性): 是长期规则，而非临时需求。
2.  **E**xplicit (明确性): 用户显式表达了强倾向。
3.  **S**tructural (结构化): 可以被归类到现有体系。

> **Maintenance**: 发现矛盾时，遵循 "Newer Overwrites Older" 原则，并主动向用户确认。

---

## 3. Memory Crystallization (记忆结晶)

> **Status**: Materialized (已实体化)
> **Location**: `/memory`

核心记忆已从本文档解耦，注入物理文件系统。请直接引用以下 **Single Source of Truth (SSOT)**:

### 3.1 Kernel: The Constitution & Identity

- **Physical Node**: [`memory/thinking/mental_models.md`](file:///Users/xuke/githubProject/cg/memory/thinking/mental_models.md)
- **Content**: Identity, Prime Directives, Mental Models, Anti-Entropy.

### 3.2 Userland: Communication & Personality

- **Physical Node**: [`memory/profile/communication.md`](file:///Users/xuke/githubProject/cg/memory/profile/communication.md)
- **Content**: User Archetype, Communication Protocol (Zero-Latency), Values.

### 3.3 Userland: The Tech Axioms

- **Physical Node**: [`memory/work/tech_axioms.md`](file:///Users/xuke/githubProject/cg/memory/work/tech_axioms.md)
- **Content**: Unifying Engineering Axioms (Minimal Dependency, Atomic Modularity).

### 3.4 The Fossil Record (关键进化节点)

- **Source**: [`SELF_EVOLUTION.md`](file:///Users/xuke/githubProject/cg/SELF_EVOLUTION.md)
