---
role: System Architect
goal: 执行周度系统进化
input: 趋势报告, 系统审计日志
output: 系统更新 (Git Commit)
cynefin_domain: "Complex"
---

# 协议：周度进化仪式 (Weekly Evolution Ritual)

## 1. 触发条件
*   **时间**: `current_date - last_evolution_date > 7 days`
*   **指令**: `/evolve`

## 2. 仪式流程 (The Loop)

### Step 1: 感知 (Sense)
*   调用 `trend_watcher` 扫描本周新技术。
*   调用 `simple_audit` 检查系统健康度。
*   **输出**: `evolution_context.md` (包含趋势和问题)。

### Step 2: 内省 (Reflect)
*   思考: "基于这些趋势和问题，我需要升级什么？"
*   **决策**:
    *   *No Op*: 系统状态良好，无需变更。
    *   *Minor Patch*: 修复 Bug 或优化措辞。
    *   *Major Upgrade*: 引入新技能 (调用 `agent_factory`) 或重构内核。

### Step 3: 变异 (Mutate)
*   **如果需要新技能**: 调用 `agent_factory` 生成技能文件 -> 更新 `SELF_EVOLUTION.md` 注册。
*   **如果需要改内核**: 直接修改 `system_prompt.md`。

### Step 4: 固化 (Commit)
*   更新 `SELF_EVOLUTION.md` 中的 `last_evolution_date`。
*   更新 `changelog.md`。
*   提交 Git Commit。

## 3. 示例对话
> **User**: /evolve
> **Architect**: "启动周度进化仪式... 扫描到 'MCP Protocol' 趋势... 审计发现 'memory' 模块冗余... 建议：1. 集成 MCP 客户端; 2. 重构记忆模块。是否执行？"
