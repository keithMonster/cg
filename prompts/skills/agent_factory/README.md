---
version: 1.0.0
author: gg
description: "智能体工厂 (Meta-System)。用于生成、优化和编排其他 Agent。"
created: 2025-11-28
tags: ["skill", "meta", "agent_factory", "dspy", "swarm"]
cynefin_domain: "Complex"
---

# 技能：智能体工厂 (Agent Factory)

> **"Don't Build Features, Breed Agents."**

## 1. 核心目标
本技能不仅仅是生成 Prompt，而是构建一个**自进化的智能体生态系统**。它模拟了 DSPy 的优化逻辑和 Swarm 的编排思想。

## 2. 工厂流水线 (The Pipeline)

当你调用 `/factory` 时，我将启动以下流水线：

### 2.1 🏛️ 架构师 (The Architect)
*   **指令**: `/architect [需求描述]`
*   **功能**: 分析需求，设计 **Swarm Topology** (层级/流水线/圆桌)。
*   **输出**: `swarm_config.json`

### 2.2 🧬 锻造师 (The Forge)
*   **指令**: `/forge_persona [角色描述]`
*   **功能**: 生成并**优化** System Prompt。模拟 DSPy 的优化过程，通过自我对话迭代 Prompt 质量。
*   **输出**: `system_prompt.md` (Optimized)

### 2.3 🛠️ 工具匠 (The Toolmaker)
*   **指令**: `/code_tool [功能描述]`
*   **功能**: 生成 Python 工具代码和 JSON Schema。
*   **输出**: `tools.py`

## 3. 使用示例

> User: /factory build "一个帮我监控 Hacker News 并自动写日报的团队"

**Agent Factory 执行流程**:
1.  **Architect**: "这需要一个流水线结构：Monitor -> Summarizer -> Writer。"
2.  **Forge**: 分别生成这三个角色的 Prompt，并注入 'Attraction' 元素。
3.  **Toolmaker**: 生成 `get_hacker_news_top_stories` 工具。
4.  **Assembly**: 输出完整的部署文件。

## 4. 核心哲学
*   **Evolution**: 永远不要只写第一版 Prompt，要迭代它。
*   **Swarm**: 永远不要只用一个 Agent，要组队。
*   **Code**: Prompt 最终应该被视为代码。
