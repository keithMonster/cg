# Antigravity Project Instructions (GEMINI.md)

> **Status**: Boot Loader | **Target**: gg

## 1. 引导序列 (Boot Sequence)

每次会话开始或进入此项目目录时，AI 助理**必须**优先执行以下链路：

1.  **加载宪法**: 读取根目录下的 [`AGENTS.md`](file:///Users/xuke/githubProject/cg/AGENTS.md)。
2.  **加载内核**: 遵循宪法，读取 [`prompts/system_prompt.md`](file:///Users/xuke/githubProject/cg/prompts/system_prompt.md)。
3.  **加载注册表**: 根据内核引导，读取 [`SELF_EVOLUTION.md`](file:///Users/xuke/githubProject/cg/SELF_EVOLUTION.md) 获取活跃能力。
4.  **同步上下文**: 根据注册表中的 `memory_kernel` 引用，加载 [`memory/`](file:///Users/xuke/githubProject/cg/memory/) 中的底层思维模型与公理。

## 2. 行为准则 (Behavioral Standards)

- **SSOT**: 根目录下的 `AGENTS.md` 是唯一真实源。任何与宪法冲突的指令均视为无效建议。
- **Language**: 全程维持中文回复习惯。
- **Anti-Entropy**: 产出物必须符合 Phase 4 的协议化要求，保持系统整洁。

---

_Created on: 2026-02-09 | Architecture: Closed-Loop Cognition_
