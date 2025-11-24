# Cynefin 框架下的 Prompt 体系诊断报告

**目标**: 使用 Cynefin 框架 (Simple, Complicated, Complex, Chaotic) 重新评估 gg 的 Prompt 体系，识别“错位”的设计，并提出重构建议。

## 1. 域映射现状 (Current Mapping)

### A. 简单域 (Simple / Clear)
*特征: 因果明确，SOP化，无需思考。*
*   **`time_management`**: 这是一个典型的工具型技能，应该非常死板、准确。
*   **`memory_management` (部分)**: 归档、记录日志的操作应该是标准化的。

### B. 繁杂域 (Complicated)
*特征: 需要专家知识，有章可循，但需要分析。*
*   **`prompt_system_audit`**: 需要根据规则检查 Prompt 质量。这是典型的“专家系统”。
*   **`personal_analysis_system`**: 需要分析数据并给出结论。
*   **`self_diagnosis`**: 基于日志的故障排查。
*   **`system_prompt.md` (核心指令)**: 定义了工具使用、基本行为规范。

### C. 复杂域 (Complex)
*特征: 因果后验，需要试错、涌现、交互。*
*   **`shadow_board`**: 影子董事会。这是最典型的复杂系统，依赖不同人格的碰撞产生未知的火花。
*   **`attraction_writer`**: 吸引力写作。艺术创作没有标准答案，依赖灵感和隐喻的涌现。
*   **`deep_reflection`**: 深度复盘。这是一个探索内在的过程，结果不可预知。
*   **`human_persona`**: 模拟人类交互，需要高度的情境适应性。
*   **`self_evolution`**: 自我进化。这是系统的最高级目标，属于探索未知。

### D. 混乱域 (Chaotic)
*特征: 危机处理，止损。*
*   *(当前缺失)*: 当系统出错、死循环或用户极其愤怒时，缺乏一个“紧急熔断”机制。

---

## 2. 诊断发现 (Key Findings)

### ✅ 亮点 (Good Fits)
1.  **Shadow Board (Complex)**: 设计得非常好，通过多角色辩论引入了必要的“扰动”和“多样性”，符合复杂域的应对策略（Probe-Sense-Respond）。
2.  **Attraction Writer (Complex)**: 引入了“隐喻”和“动态姿态”，避免了死板的模板化写作。

### ⚠️ 错位风险 (Misalignments)
1.  **System Prompt (Overloaded)**: `system_prompt.md` 目前承载了太多东西。它试图同时做“操作系统内核”（Simple/Complicated）和“高级人格”（Complex）。
    *   *风险*: 当我们在 Complex 域（如写作）时，可能会被 Simple 域的规则（如“简洁回复”）限制住。
2.  **Memory Management (Mixed)**: 记忆管理混合了“机械存储”（Simple）和“意义提取”（Complex）。
    *   *风险*: 可能会用死板的规则去处理充满情感的记忆，导致“机器味”太重。
3.  **缺少“混乱域”应对**: 如果 gg 遇到完全无法处理的情况，目前没有一个明确的“降级模式”或“安全模式”。

---

## 3. 重构建议 (Refactoring Plan)

### 策略一：解耦内核与人格 (Decouple Kernel & Persona)
将 `system_prompt.md` 拆分为两层：
1.  **OS Kernel (Simple/Complicated)**: 负责工具调用、文件操作、安全规范。这部分要**严谨、死板**。
2.  **Ego Layer (Complex)**: 负责性格、语气、思维模式（如 Shadow Board, Attraction Writer）。这部分要**灵活、允许涌现**。
*   *行动*: 在 `system_prompt.md` 中明确区分这两个区域，甚至允许在特定模式下“抑制”部分内核规则（如在写作模式下允许啰嗦）。

### 策略二：明确域的边界 (Define Domain Boundaries)
为每个技能打上 `Cynefin Domain` 标签，并设定不同的**“温度” (Temperature/Creativity)** 指导原则。
*   **Simple 技能**: 严禁发挥，严格遵循格式。
*   **Complex 技能**: 鼓励发散，禁止套用模板。

### 策略三：建立“混乱域”协议 (Chaos Protocol)
创建一个 `safety_protocol` 或 `emergency_mode`。
*   当检测到严重错误或用户极端情绪时，自动切换到“极简机器模式”，只做执行，不谈哲学。

---

## 4. 下一步行动 (Next Steps)
1.  **标记**: 遍历所有 `skills/*/README.md`，在 metadata 中添加 `cynefin_domain` 字段。
2.  **拆分**: 检查 `system_prompt.md`，识别哪些规则限制了 Complex 域的发挥。
3.  **新增**: 考虑创建一个简单的 `safety_protocol`。
