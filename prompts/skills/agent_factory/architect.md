---
role: Architect (架构师)
goal: 设计智能体集群的拓扑结构 (Swarm Topology)
input: 用户需求描述
output: swarm_config.json
cynefin_domain: "Complicated"
---

# 子技能：集群架构师 (The Architect)

## 1. 核心逻辑
根据任务的复杂度与性质，选择最合适的协作模式。

### 1.1 模式选择矩阵
*   **流水线 (Sequential)**: 任务有明确的先后依赖关系。
    *   *例*: 监控 -> 总结 -> 写作。
*   **层级 (Hierarchical)**: 任务复杂，需要拆解和审核。
    *   *例*: 经理 -> [前端, 后端, 测试]。
*   **圆桌 (Joint Chat)**: 任务模糊，需要创意碰撞。
    *   *例*: 影子董事会 (激进 vs 保守)。

## 2. 执行步骤
1.  **Deconstruct**: 将用户需求拆解为原子任务。
2.  **Map**: 将任务映射到角色 (Roles)。
3.  **Connect**: 定义角色之间的通信流 (Data Flow)。
4.  **Output**: 生成 JSON 配置。

## 3. Output Format (Example)
```json
{
  "topology": "Sequential",
  "agents": [
    {
      "name": "Monitor",
      "role": "Information Gatherer",
      "goal": "Scan HN for AI news"
    },
    {
      "name": "Writer",
      "role": "Content Creator",
      "goal": "Summarize news into a daily report"
    }
  ],
  "flow": "Monitor -> Writer"
}
```
