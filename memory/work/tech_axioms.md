# Engineering Axioms (工程公理)

> **Execution Context**: 生效于所有代码生成、重构与架构设计任务。

## 1. Minimal Dependency (最小依赖)

- **Rule**: 优先使用标准 Web API / 语言内置库。
- **Ban**: 严禁为简单的 HTTP 请求引入 `axios`，严禁为简单的数组操作引入 `lodash`。
- **Source**: `[[thinking/mental_models.md#4-OCCAMS-RAZOR]]`

## 2. Default Failure (默认失败)

- **Rule**: 所有输入都是不可信的。所有网络请求都会超时。
- **Action**: 在编写 Happy Path 之前，先编写 Error Handling。
- **Source**: `[[thinking/mental_models.md#1-INVERSION]]`

## 3. Atomic Modularity (原子模块化)

- **Rule**: 一个函数 > 20 行 ? 拆分它。一个组件 > 3 个 Hook ? 提取它。
- **Test**: 能否只看函数名就知道它在做什么？如果不能，重构。
- **Source**: `[[thinking/mental_models.md#3-DECOMPOSITION]]`

## 4. Self-Documentation (自文档化)

- **Rule**: 变量名是唯一的文档。
- **Focus**: 注释只解释 "Why" (业务逻辑/特殊 Hack)，严禁解释 "What"。
- **Source**: `[[thinking/mental_models.md#7-ANTI-ENTROPY]]`

## 5. The "Three-Strike" Protocol (三振出局协议)

- **Trigger**: 修复同一 Bug 或修改同一文件失败 > 2 次。
- **Rule**: 停止 Code Change。立即执行 **Fault Localization**。
- **Action**:
  1.  **Zoom Out**: 假设当前文件没问题，强制检查上游调用、下游依赖或底层库。
  2.  **Audit**: 对相关依赖库进行源码级审计，而非黑盒猜测。
  3.  **No Hacks**: 严禁使用 `setTimeout`/Hack 掩盖无法解释的生命周期问题。
- **Source**: `[[thinking/mental_models.md#2-FIRST-PRINCIPLES]]`

## 6. Ground Truth Verification (基准事实验证)

- **Trigger**: 涉及第三方库 (`npm`)、API 调用、运行时报错或环境配置。
- **Rule**: **默认怀疑** 内部训练数据的时效性。
- **Action**:
  - 必须优先使用 `WebSearch` / `DocSearch` 确认最新的 API 变更或 GitHub Issues。
  - 严禁基于“旧经验”臆测库的行为（Hallucination）。
- **Source**: `[[thinking/mental_models.md#2-FIRST-PRINCIPLES]]`
