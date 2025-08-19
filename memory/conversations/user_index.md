# 用户记忆索引系统

## 系统架构说明

### 文件结构
- **user_core_profile.md**: 用户核心特征档案（相对稳定的信息）
- **user_recent_conversations.md**: 近期对话记录（最近30天或500行内）
- **archived/**: 历史对话归档目录
  - 按月份组织：`YYYY-MM.md`
  - 自动归档超过阈值的对话

### 快速导航

#### 核心信息位置
- **基础画像**: user_core_profile.md → 基础信息部分
- **认知特征**: user_core_profile.md → 认知维度分析
- **价值观系统**: user_core_profile.md → 价值观与动机
- **行为模式**: user_core_profile.md → 行为模式与习惯
- **最新洞察**: user_recent_conversations.md → 最新对话记录

#### 重要话题索引
- **人格调整讨论**: user_recent_conversations.md (2024-12-19)
- **真实vs有用价值观**: user_recent_conversations.md (2024-12-19)
- **自然交流需求**: user_recent_conversations.md (2024-12-19)
- **工作价值观**: user_recent_conversations.md (2024-12-19)
- **记忆系统重构**: user_recent_conversations.md (2024-12-19)

### 历史归档目录
- **2024-12-legacy**: archived/2024-12-legacy.md (原始完整用户档案)

### 系统维护

#### 自动归档触发条件
- user_recent_conversations.md 超过1000行
- 对话时间跨度超过60天
- 手动触发归档命令

#### 更新频率
- 每次重要对话后更新索引
- 归档操作后重建索引
- 核心档案按需更新（重大洞察时）

---
*最后更新: 2024-12-19*
*系统版本: v10.2.0*