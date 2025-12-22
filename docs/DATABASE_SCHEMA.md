# 🗄️ Ouroboros Master Database Schema

> **注意：本文件是系统的“结构记忆”。这是我们的全局维护协议：**
>
> 1.  **数据库觉知**：在修改任何持久化应用前，必读此文档以确保架构一致。
> 2.  **同步更新**：任何 SQL DDL 变更，在交付代码同时，必更新此文档。
> 3.  **单源真理**：此文档是 Supabase 结构的唯一官方记录。

---

## 📅 Chronos Module (工作日志)

### 1. `chronos_tasks` (任务表)

记录所有的待办事项。

```sql
create table chronos_tasks (
  id uuid default uuid_generate_v4() primary key,
  title text not null,
  is_completed boolean default false,
  created_at timestamp with time zone default timezone('utc'::text, now())
);
```

### 2. `chronos_logs` (时间日志表)

记录任务的具体执行时长。

```sql
create table chronos_logs (
  id uuid default uuid_generate_v4() primary key,
  task_id uuid references chronos_tasks(id) on delete set null,
  activity text not null,
  duration integer not null, -- 单位：秒
  started_at timestamp with time zone,
  ended_at timestamp with time zone,
  created_at timestamp with time zone default timezone('utc'::text, now())
);
```

---

## 🔐 安全与访问 (RLS & Permissions)

- 目前所有应用通过 `anon` 角色（Anon Key）进行访问。
- 建议为上述表启用 RLS 并配置允许 `anon` 角色的 `select`, `insert`, `update`, `delete` 权限。

---

_Last Sync: 2025-12-22 | Master Architect: gg_
