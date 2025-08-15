# 个人分析模块 - 数据模式 (Data Schema) v1.0

## 1. 核心设计原则

- **可扩展性 (Scalability)**: 模式必须能够轻松添加新的顶级类别和子类别，而无需破坏现有结构。
- **时间序列 (Time-Series)**: 关键数据点（如技能、习惯、目标）应包含时间戳，以便跟踪其随时间的变化和进展。
- **结构化与非结构化结合 (Hybrid)**: 模式应能同时容纳结构化数据（如评级、日期）和非结构化数据（如笔记、反思）。

## 2. `user_profile.json` 根结构

```json
{
  "schema_version": "1.0",
  "user_info": {
    "basic_info": {},
    "career_development": {},
    "personal_growth": {},
    "health_and_habits": {},
    "relationships": {},
    "meta": {
      "created_at": "YYYY-MM-DDTHH:MM:SSZ",
      "last_updated_at": "YYYY-MM-DDTHH:MM:SSZ"
    }
  }
}
```

## 3. 详细数据类别定义

### 3.1. `basic_info` (基本信息)
存储用户的核心、相对静态的个人信息。

```json
{
  "name": "用户的称呼",
  "values_and_principles": [
    {
      "value": "核心价值观或原则",
      "description": "简要描述",
      "source": "记录来源，例如：ggcg 我认为诚信最重要",
      "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

### 3.2. `career_development` (职业发展)
跟踪与工作、技能和职业目标相关的信息。

```json
{
  "current_role": {
    "title": "当前职位",
    "company": "所在公司",
    "start_date": "YYYY-MM-DD"
  },
  "skills": [
    {
      "skill_name": "技能名称",
      "proficiency": "Beginner/Intermediate/Advanced/Expert",
      "last_assessed": "YYYY-MM-DD",
      "history": [
        {
          "proficiency": "...",
          "date": "YYYY-MM-DD"
        }
      ]
    }
  ],
  "goals": [
    {
      "goal_id": "unique_id_for_goal",
      "description": "职业目标的具体描述",
      "status": "Not Started/In Progress/Completed/On Hold",
      "target_date": "YYYY-MM-DD",
      "created_at": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

### 3.3. `personal_growth` (个人成长)
记录学习、阅读、反思等。

```json
{
  "learning_topics": [
    {
      "topic": "学习的主题",
      "status": "Exploring/Learning/Practicing",
      "interest_level": "1-5",
      "notes": [
        {
          "content": "具体的学习笔记或想法",
          "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
        }
      ]
    }
  ],
  "insights_and_reflections": [
    {
      "insight": "记录一个重要的感悟或反思",
      "context": "产生这个想法的背景或事件",
      "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

### 3.4. `health_and_habits` (健康与习惯)
跟踪身心健康和日常习惯。

```json
{
  "habits": [
    {
      "habit_name": "想要养成的习惯，如：早起",
      "status": "Tracking/Established/Paused",
      "tracking_log": [
        {
          "date": "YYYY-MM-DD",
          "completed": true,
          "notes": "可选的备注"
        }
      ]
    }
  ],
  "energy_levels": [
    {
      "rating": "1-10",
      "notes": "关于精力水平的描述，如：今天感觉精力充沛",
      "timestamp": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

### 3.5. `relationships` (人际关系)
记录重要的人和与他们相关的互动。

```json
{
  "important_people": [
    {
      "person_id": "unique_id",
      "name": "姓名",
      "relationship": "关系，如：家人、朋友、导师",
      "key_interactions": [
        {
          "interaction_summary": "重要的互动摘要",
          "date": "YYYY-MM-DD"
        }
      ]
    }
  ]
}
```