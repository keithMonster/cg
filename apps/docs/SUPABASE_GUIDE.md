# 🗄️ Supabase Integration Guide (Ouroboros Bridge)

为了让所有的“卫星应用”具备持久化记忆，我们采用 **Supabase** 作为全局后端。所有的交互将遵循“零构建”友好的 SDK 引入模式。

---

## 1. 基础连接协议 (The Bridge Protocol)

在我们的 **Zero-Build (单文件)** 架构中，为了避免在每个 HTML 中硬编码秘钥，我们采用 **“Local Vault (本地保险箱)”** 模式：

1.  **统一入口**：用户在 `Nexus Core` 控制台输入一次秘钥。
2.  **域内共享**：秘钥通过 `localStorage` 存储。由于所有卫星应用（Portal, GenUI, Core）都运行在同一个域名下（如 GitHub Pages），它们可以共享同一个 `localStorage` 空间。
3.  **自动恢复**：每个子应用初始化时，优先从 `localStorage` 读取秘钥。

### 客户端初始化规范：

每个应用在引入 Supabase 时，必须在 `<head>` 中添加资源：

```html
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
```

标准初始化逻辑：

```javascript
// Ouroboros 标准：从本地保险箱提取配置
const ODS_CONFIG = {
  url:
    localStorage.getItem('sb_url') ||
    'https://pscrjxtqukzivnuiqwah.supabase.co',
  key: localStorage.getItem('sb_key') || '',
};

const supabaseClient =
  ODS_CONFIG.url && ODS_CONFIG.key
    ? supabase.createClient(ODS_CONFIG.url, ODS_CONFIG.key)
    : null;
```

---

## 2. 推荐数据模型 (Schema Design)

为了保持系统的自进化能力，我们建议建立以下核心表：

### 表名：`satellite_states` (应用状态存储)

- `id`: uuid (primary key)
- `app_id`: string (例如 'gen-ui-lite')
- `state_data`: jsonb (存储所有动态状态)
- `updated_at`: timestamp

### 表名：`ouroboros_memories` (全局记忆池)

- `id`: uuid
- `category`: string (e.g., 'concept', 'decision', 'log')
- `content`: text
- `meta`: jsonb
- `created_at`: timestamp

---

## 3. 安全与权限 (RLS)

由于我们目前是全客户端调用，请务必在 Supabase Dashboard 为相关表启用 **Row Level Security (RLS)**，并根据需要配置 `anon` 角色的访问权限。

---

> [!TIP] > **等候输入**：请在收到 `SUPABASE_URL` 和 `SUPABASE_ANON_KEY` 后，将其更新至本项目的全局配置中。
