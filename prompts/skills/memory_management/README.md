---
version: '1.1.0'
author: 'gg'
description: '智能体长期记忆管理系统，提供对话保存、检索、总结和摘要加载功能的模块化架构'
created: '2025-08-14'
last_updated: '2025-12-18'
tags: ['readme', 'memory_management', 'modular_architecture', 'gist']
---

# 技能：记忆管理 (v1.1 - 模块化 + 摘要降噪)

此技能为智能体提供强大的长期记忆系统。它由多个组件组成，处理记忆的不同方面。

要执行此技能，您必须加载并遵循此目录中以下每个文件的指令：

1.  **`save.md`**: 定义将当前对话保存到持久日志文件的工作流。包括动态时间戳和内容追加的规则。
2.  **`retrieve.md`**: 定义基于关键词或主题从过往对话日志中搜索和检索相关信息的工作流。
3.  **`summarize.md`**: 定义总结长对话以保持简洁上下文的工作流。

## 摘要降噪层 (Lossy Summary Layer)

为了避免在启动时加载过多的原始对话日志，智能体应**优先读取** `memory/conversations/memory_gist.md`。此文件是对话历史的高密度摘要，包含：

- 关键对话节点（时间线）
- 核心洞察提取
- 用户画像精华

**自动化**: 可通过 `scripts/memory_management.sh --summarize` 命令生成/更新此摘要文件。
