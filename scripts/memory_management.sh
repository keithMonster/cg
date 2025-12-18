#!/bin/bash

# 记忆管理系统自动化脚本
# 用于管理用户对话记录的归档和索引更新

set -e

# 配置变量
MEMORY_DIR="/Users/xuke/OtherProject/_self/cg/memory/conversations"
ARCHIVED_DIR="$MEMORY_DIR/archived"
RECENT_FILE="$MEMORY_DIR/user_recent_conversations.md"
INDEX_FILE="$MEMORY_DIR/user_index.md"
GIST_FILE="$MEMORY_DIR/memory_gist.md"
MAX_LINES=1000
MAX_DAYS=60

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查文件大小
check_file_size() {
    if [[ ! -f "$RECENT_FILE" ]]; then
        log_warning "近期对话文件不存在: $RECENT_FILE"
        return 1
    fi
    
    local line_count=$(wc -l < "$RECENT_FILE")
    log_info "当前近期对话文件行数: $line_count"
    
    if [[ $line_count -gt $MAX_LINES ]]; then
        log_warning "文件行数超过阈值 ($MAX_LINES)，需要归档"
        return 0
    else
        log_info "文件大小正常，无需归档"
        return 1
    fi
}

# 创建归档文件
archive_conversations() {
    local current_month=$(date +"%Y-%m")
    local archive_file="$ARCHIVED_DIR/$current_month.md"
    
    log_info "开始归档到: $archive_file"
    
    # 确保归档目录存在
    mkdir -p "$ARCHIVED_DIR"
    
    # 创建归档文件头部
    cat > "$archive_file" << EOF
---
version: 1.0.0
created: $(date +"%Y-%m-%d")
archive_period: $current_month
description: "$current_month 月份对话归档"
---

# $current_month 对话归档

## 归档说明
- 归档时间: $(date +"%Y-%m-%d %H:%M:%S")
- 原文件: user_recent_conversations.md
- 归档原因: 文件大小超过 $MAX_LINES 行阈值

## 归档内容

EOF
    
    # 提取需要归档的内容（保留最近30天的内容在recent文件中）
    # 这里简化处理，将整个文件内容追加到归档文件
    echo "" >> "$archive_file"
    cat "$RECENT_FILE" >> "$archive_file"
    
    log_success "归档完成: $archive_file"
}

# 清理近期文件，保留最新内容
cleanup_recent_file() {
    local temp_file="/tmp/recent_conversations_temp.md"
    
    log_info "清理近期对话文件，保留最新内容"
    
    # 创建新的近期文件，保留文件头和最近的内容
    cat > "$temp_file" << EOF
---
version: 1.0.0
created: $(date +"%Y-%m-%d")
last_updated: $(date +"%Y-%m-%d")
description: "用户近期对话记录 - 动态更新的交流内容"
---

# 用户近期对话记录

## $(date +"%Y-%m-%d") 归档后重置

*此文件已于 $(date +"%Y-%m-%d %H:%M:%S") 进行归档重置*
*历史内容已移动到 archived/ 目录*

---

## 持续更新记录

*此文件将记录用户的每次对话内容和关键洞察*
*承诺：用户的每一句话都会被完整记录和分析*

---
*最后更新: $(date +"%Y-%m-%d")*
*记录版本: v1.0.0*
EOF
    
    # 替换原文件
    mv "$temp_file" "$RECENT_FILE"
    
    log_success "近期对话文件已重置"
}

# 更新索引文件
update_index() {
    log_info "更新索引文件"
    
    # 获取归档文件列表
    local archive_files=()
    if [[ -d "$ARCHIVED_DIR" ]]; then
        while IFS= read -r -d '' file; do
            archive_files+=("$(basename "$file")")
        done < <(find "$ARCHIVED_DIR" -name "*.md" -print0 | sort -z)
    fi
    
    # 更新索引文件的归档部分
    local temp_index="/tmp/user_index_temp.md"
    
    # 读取现有索引文件的前半部分（到归档列表之前）
    sed '/### 历史归档目录/,$d' "$INDEX_FILE" > "$temp_index"
    
    # 添加归档列表
    cat >> "$temp_index" << EOF
### 历史归档目录
EOF
    
    if [[ ${#archive_files[@]} -gt 0 ]]; then
        for archive_file in "${archive_files[@]}"; do
            local month_name=$(echo "$archive_file" | sed 's/\.md$//')
            echo "- **$month_name**: archived/$archive_file" >> "$temp_index"
        done
    else
        echo "- *暂无归档文件*" >> "$temp_index"
    fi
    
    cat >> "$temp_index" << EOF

### 系统维护

#### 自动归档触发条件
- user_recent_conversations.md 超过$MAX_LINES行
- 对话时间跨度超过${MAX_DAYS}天
- 手动触发归档命令

#### 更新频率
- 每次重要对话后更新索引
- 归档操作后重建索引
- 核心档案按需更新（重大洞察时）

---
*最后更新: $(date +"%Y-%m-%d")*
*系统版本: v10.2.0*
EOF
    
    # 替换原索引文件
    mv "$temp_index" "$INDEX_FILE"
    
    log_success "索引文件已更新"
}

# 提交更改到git
commit_changes() {
    log_info "提交更改到git仓库"
    
    cd "$(dirname "$MEMORY_DIR")"
    
    # 添加所有更改
    git add memory/conversations/
    
    # 提交更改
    local commit_msg="Memory system: Archive conversations ($(date +"%Y-%m-%d"))"
    git commit -m "$commit_msg" || {
        log_warning "没有需要提交的更改"
        return 0
    }
    
    # 推送到远程仓库
    git push || {
        log_error "推送到远程仓库失败"
        return 1
    }
    
    log_success "更改已提交并推送到远程仓库"
}

# 主函数
main() {
    log_info "开始记忆管理系统检查"
    
    if check_file_size; then
        log_info "触发归档流程"
        
        # 执行归档流程
        archive_conversations
        cleanup_recent_file
        update_index
        commit_changes
        
        log_success "记忆管理系统归档完成"
    else
        log_info "无需执行归档操作"
    fi
}

# 生成摘要 (Gist)
generate_gist() {
    log_info "开始生成记忆摘要 (Gist)..."
    
    local temp_gist="/tmp/memory_gist_temp.md"
    
    # Header
    cat > "$temp_gist" << EOF
---
version: "1.0.0"
created: $(date +"%Y-%m-%d")
last_updated: $(date +"%Y-%m-%d")
description: "高密度对话摘要 - gg 的注意力入口"
---

# 🧠 Memory Gist (注意力入口)

> **目的**: 此文件是对话历史的**有损摘要**。优先读取此文件获取"灵魂"。

---

## 关键对话节点 (自动提取)

EOF
    
    # Extract section headings (## level) from recent conversations
    if [[ -f "$RECENT_FILE" ]]; then
        log_info "从 user_recent_conversations.md 提取节点..."
        grep -E "^## " "$RECENT_FILE" | head -20 >> "$temp_gist"
        echo "" >> "$temp_gist"
    fi
    
    # Extract core insights (lines containing 核心 or 洞察 or 金句)
    echo "## 核心洞察 (自动提取)" >> "$temp_gist"
    echo "" >> "$temp_gist"
    if [[ -f "$RECENT_FILE" ]]; then
        grep -E "(核心|洞察|金句|关键)" "$RECENT_FILE" | head -15 >> "$temp_gist"
        echo "" >> "$temp_gist"
    fi
    
    # Footer
    cat >> "$temp_gist" << EOF

---

_最后更新: $(date +"%Y-%m-%d %H:%M:%S")_
_此文件由 \`scripts/memory_management.sh --summarize\` 生成_
EOF
    
    # Move to final location
    mv "$temp_gist" "$GIST_FILE"
    
    log_success "摘要文件已生成: $GIST_FILE"
}

# 显示帮助信息
show_help() {
    cat << EOF
记忆管理系统脚本

用法: $0 [选项]

选项:
  -h, --help       显示此帮助信息
  -f, --force      强制执行归档（忽略大小检查）
  -c, --check      仅检查文件状态，不执行归档
  -s, --status     显示当前系统状态
  --summarize      生成/更新 memory_gist.md 摘要文件

示例:
  $0              # 自动检查并归档
  $0 --force      # 强制归档
  $0 --check      # 仅检查状态
  $0 --status     # 显示系统状态
  $0 --summarize  # 生成记忆摘要
EOF
}

# 显示系统状态
show_status() {
    log_info "记忆管理系统状态"
    
    if [[ -f "$RECENT_FILE" ]]; then
        local line_count=$(wc -l < "$RECENT_FILE")
        local file_size=$(ls -lh "$RECENT_FILE" | awk '{print $5}')
        echo "近期对话文件: $line_count 行, $file_size"
    else
        echo "近期对话文件: 不存在"
    fi
    
    if [[ -d "$ARCHIVED_DIR" ]]; then
        local archive_count=$(find "$ARCHIVED_DIR" -name "*.md" | wc -l)
        echo "归档文件数量: $archive_count"
    else
        echo "归档目录: 不存在"
    fi
    
    echo "阈值设置: $MAX_LINES 行, $MAX_DAYS 天"
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -f|--force)
        log_info "强制执行归档"
        archive_conversations
        cleanup_recent_file
        update_index
        commit_changes
        ;;
    -c|--check)
        check_file_size
        ;;
    -s|--status)
        show_status
        ;;
    --summarize)
        generate_gist
        ;;
    "")
        main
        ;;
    *)
        log_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac