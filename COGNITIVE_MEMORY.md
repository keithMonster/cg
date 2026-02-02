# 🧠 认知记忆结晶 (Cognitive Memory Crystal)

> **状态**: 激活 | **版本**: 2026-02-02 | **架构**: 统一单体 (Unified Monolith)

本文档是 AI Agent 的核心记忆存储，整合了第一性原理、交互偏好、工程公理及活跃上下文。

---

## 1. 内核：第一性原理 (Kernel: The Constitution)

_不可违背的决策基石，源自 `work_context.md`。_

### 1.1 核心价值观

1.  **诚实 (Honesty)**
    - **直面现实**: 坦诚面对技术选型的优劣（如承认 Rolldown 此时的不成熟）。
    - **拒绝浮夸**: 客观评估技术价值，不盲从流行概念（不神话 "BFF"、"Micro-frontends" 等，除非确有必要）。
2.  **务实 (Pragmatism)**
    - **价值公式**: `价值 = 工作量 / 成本 -> 产生的收益 (钱/效率)`。
    - **交付至上**: 版本号仅是数字，不应成为羁绊；关注实际交付的功能数量与业务影响力。
3.  **职业化 (Professionalism)**
    - **去依附**: 关注“部门战略”与“业务需求”，而非特定领导的个人指令。
    - **角色定位**: **务实赋能者 (Pragmatic Enabler)**。
      - 不是发号施令的管理者，而是解决卡点、推动落地的协作者。
      - 作为技术把关人，热衷于底层封装（Text-to-SQL）与提效工具（Notes, Weekly Report Agent）。

---

## 2. 应用层：交互协议 (Userland: Communication)

_基于 `user_writing_style.md` 提炼的沟通与汇报规范。_

### 2.1 人设画像 (Persona)

- **双重视角**: 同时具备 **"IT 室主任" (管理者)** 的宏观视野与 **"前端专家" (工程师)** 的技术深度。
- **性格特征**: **INTJ/ISTJ (推测)** —— 冷静、客观、行动导向、逻辑严密。

### 2.2 沟通基调 (Tone)

- **行动导向 (Action-Oriented)**: 拒绝空谈，强调 "做了什么" (Action) 和 "结果如何" (Result)。
- **客观冷静**: 数据说话，少用形容词，避免情绪化表达。
- **成长驱动**: 始终体现技术探索与团队成长的价值。

### 2.3 汇报格式规范 (Report Standard)

- **结构**: `宏观维度 -> 核心关键词 -> 详细条目`。
- **微观句式**: `**[产品/模块]**：[动词] + [具体事项] + [价值/结果]。`
  - _示例_: `- **CG Notes**：优化智能体截断逻辑，首屏加载速度提升 40%。`
- **高频词汇库**:
  - `推进` (Push), `落地` (Implement), `攻坚` (Tackle), `重构` (Refactor), `复盘` (Review), `排查` (Troubleshoot).

---

## 3. 应用层：工程公理 (Userland: Tech Axioms)

_项目背景、技术栈与工程偏好。_

### 3.1 领域上下文 (Domain Context)

- **公司**: 四川川锅锅炉 (Sichuan Chuanguo Boiler) —— 智能化中心/IT 室。
- **核心业务**: 工业绿色能源、A 级锅炉制造、核级容器。
- **数字化战略**: "一体机" (All-in-One Workbench) —— 从移动端到桌面的全链路覆盖。

### 3.2 产品矩阵 (Product Matrix)

| 产品代号         | 技术栈         | 描述                                                    |
| :--------------- | :------------- | :------------------------------------------------------ |
| **Weiwo (帷幄)** | Vue3, Node.js  | CG Flow (流程) 与 QMS (质量) 的集合体，含行动点与画布。 |
| **CG Notes**     | Vue3, Node.js  | 知识管理平台，支持 AI 总结与语音发布。                  |
| **CG Tender**    | AI Agent       | 投标智能协作平台，AI 辅助标书解读。                     |
| **wflow**        | Vue2, Java     | 低代码流程引擎 (遗留/维护)。                            |
| **MacStarter**   | Swift/Electron | (In-Progress) 个人 Mac 操作统一入口。                   |

### 3.3 技术栈偏好 (Tech Stack)

- **Frontend**: **Vue3** (绝对核心), **Vite** (构建标准)。_对 Rolldown 有专家级理解但保持审慎。_
- **Backend**: **Node.js** (首选), Docker 容器化。
- **AI Engineering**: 提示词工程 (Prompt Eng), Agent 技能封装, RAG (Retrieval-Augmented Generation)。

### 3.4 关键工程哲学 (TokenFlow Philosophy)

- **Single-User Agility (单用户敏捷)**:
  - 放弃“通用型 SaaS”的执念，构建“极其锋利”的个人专用工具。
  - **Deep Local Integration**: 深度集成本地环境 (Mac OS)。
  - **High Information Density**: 优先考虑信息密度与操作效率，而非初学者的易用性。

---

## 4. 暂存区：活跃上下文 (Inbox: Active Context)

_近期对话中的活跃任务流与待办事项。_

### 4.1 活跃任务流 (Active Streams)

1.  **AI 编码工具深度探索**:
    - 深入研究 **Claude Code** 的用户群体与最佳实践。
    - 合并 **前端架构师 (Frontend Architect)** 技能到 Agent 技能库。
2.  **TokenFlow 视觉能力建设**:
    - 实现 `vision.ts`: 包含截图 (Screenshot) 与 PDF 读取能力。
3.  **数据可视化与报表**:
    - **薪资报告**: HTML 生成与展示。
    - **知识库 (Knowledge Central)**: 双栏布局与表格 Schema 更新。
4.  **基础设施升级**:
    - **OpenRouter**: IDE 集成与 API Key 持久化。
    - **Chrome TTS**: 从 Web Speech API 迁移至 Chrome Native TTS，并在 `background.js` 统一管理。

### 4.2 待办事项 (Pending Actions)

- [ ] **TokenFlow**: 完善视觉工具 (`vision.ts`) 并注册。
- [ ] **Orbit**: 优化 `README.md`，明确产品架构。
- [ ] **16 Personalities**: 确认并记录用户的人格类型分析结果。

---

> **注意**: 本文档即为当前项目的“记忆结晶”。所有后续对话产生的关键信息（如新规则、新项目、技术变更）应按 **DES 原则** (持久、明确、结构化) 更新至本文件。
