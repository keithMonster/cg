# 影子董事会辩论协议（v1）

## 输入要求（Required Input）

每次会议开始前，必须提供：

- `problem`：一句话描述要决策的问题
- `context`：关键背景与约束（时间/资源/家庭/工作）
- `options`：至少两个候选方案
- `deadline`：最晚决策时间
- `success_metric`：可观测的成功标准
- `non_negotiables`：不可妥协边界

## 会议流程（5 Rounds）

### Round 1 - 问题重述（Problem Framing）

目标：确认“正在决策的到底是什么问题”。

输出：

- 问题定义（1 句）
- 决策类型：可逆 / 不可逆
- 不决策的代价（Cost of Inaction）

### Round 2 - 角色陈述（Role Statements）

每个角色按统一格式发言：

- 立场（支持哪个方案，或提出新方案）
- 证据（事实、经验、约束）
- 代价（机会成本/执行成本）

### Round 3 - 反证攻击（Adversarial Review）

目标：主动寻找失败路径，避免确认偏差。

输出：

- Top 3 失败场景
- 每个场景的触发信号
- 可执行的缓释动作

### Round 4 - 收敛决策（Convergence）

输出：

- 主方案（Primary Plan）
- 备选方案（Fallback Plan）
- 切换条件（Switch Trigger）
- 置信度（0~100）与可逆性评级（高/中/低）

### Round 5 - 行动承诺（Execution Contract）

输出：

- 7 天内行动项（<= 3 项）
- 每项必须有 owner + deadline + done 标准
- 下次回看时间（具体到日期）

## 会议约束（Constraints）

- 禁止“纯观点输出”，所有结论必须可执行或可验证
- 禁止行动项超过 3 个，防止执行稀释
- 若讨论超过 30 分钟未收敛，强制进入“最小可行决策”

## 会后归档（Archiving）

每次会议后，必须将结果存档到 `memory/decisions/YYYY-MM-DD-[topic].md`，并在周五复盘沉淀到 `memory/patterns.md`。
