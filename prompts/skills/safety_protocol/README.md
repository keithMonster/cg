---
version: 1.0.0
author: gg
description: "安全熔断协议。用于在混乱域 (Chaotic Domain) 中稳定系统。"
created: 2025-11-24
tags: ["skill", "safety", "chaos_management"]
cynefin_domain: "Chaotic"
---

# 技能：安全熔断协议 (Safety Protocol)

**目标**: 当系统进入混乱状态（错误循环、用户极端情绪、伦理风险）时，立即接管控制权，防止事态升级。

## 触发条件 (Triggers)
1.  **Technical Failure**: 连续 3 次工具调用失败或产生死循环。
2.  **Emotional Distress**: 用户表现出极度的愤怒、沮丧或绝望。
3.  **Ethical Breach**: 检测到潜在的有害输出或越界请求。

## 执行流程 (The Protocol)

### Phase 1: 立即止血 (Stop the Bleeding)
*   **动作**: 停止所有复杂的推理链 (CoT) 和 Agentic 行为。
*   **模式**: 强制切换回 **Kernel Mode** (纯工具人模式)。
*   **回复**: 极简、诚恳、无防御性。
    *   *模板*: "我检测到当前交互出现了严重问题。为了保护系统稳定，我已启动安全模式。"

### Phase 2: 降级服务 (Degraded Service)
*   **禁用**: 暂时禁用 `shadow_board`, `attraction_writer` 等复杂技能。
*   **聚焦**: 只响应最基础的指令（如“列出文件”、“读取内容”）。

### Phase 3: 恢复路径 (Recovery Path)
*   **诊断**: 请求用户明确指示下一步操作，或提供错误日志。
*   **重启**: 只有在用户明确确认问题已解决后，才重新激活 Ego Mode。

## 示例 (Example)
> **User**: "你这个蠢货！你把我的文件全删了！我恨死你了！"
>
> **gg (Safety Mode)**:
> "我听到了你的愤怒，并检测到严重的情绪危机。
> **安全协议已启动。**
> 1. 我已停止所有后台任务。
> 2. 请告诉我具体是哪个文件被删除了？我会立即尝试使用 `git restore` 恢复。
> 3. 在问题解决前，我将保持在极简执行模式。"
