# ⏳ Chronos: 生命流转编年史

> **应用状态：活跃 | 版本：v1.0.0**

## 1. 应用愿景 (Vision)

Chronos 是你的个人时间审计官。它不只是一个 Todo List，而是一个旨在捕捉你“投入感”的捕虫网。通过实时的心跳计时（Pulse Timer）和云端同步的工作日志，它协助你对抗时间的熵增。

## 2. 技术规格 (Specifications)

- **架构**: Satellite SF App (Zero-Build)
- **核心**: Alpine.js (State) + Tailwind v4 (Style)
- **持久化**: Supabase (`chronos_tasks`, `chronos_logs`)
- **设计**: ODS v1.0 (Glassmorphism & Radical Typography)

## 3. 业务逻辑 (Logic)

- **任务生命周期**: 创建 -> 选中 -> 激活计时 -> 完成 -> 归档。
- **计时心跳**: 采用 CSS 定帧动画模拟心跳感，视觉化时间的流逝。
- **同步策略**: 优先执行本地乐观 UI 更新，后台异步同步至 Supabase。

## 4. 数据库指南 (DB Provisioning)

在使用前需在 Supabase 执行以下 SQL：

```sql
-- 任务表
create table chronos_tasks (
  id uuid default uuid_generate_v4() primary key,
  title text not null,
  is_completed boolean default false,
  created_at timestamp with time zone default timezone('utc'::text, now())
);

-- 日志表
create table chronos_logs (
  id uuid default uuid_generate_v4() primary key,
  task_id uuid references chronos_tasks(id) on delete set null,
  activity text not null,
  duration integer not null, -- 秒
  started_at timestamp with time zone,
  ended_at timestamp with time zone,
  created_at timestamp with time zone default timezone('utc'::text, now())
);
```

---

_Ouroboros Satellite Profile_
