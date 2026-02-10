#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$ROOT_DIR/runs/weekly"
OUT_DIR="$ROOT_DIR/memory/decisions"
TEMPLATE="$ROOT_DIR/protocols/output.template.md"

mkdir -p "$RUNS_DIR" "$OUT_DIR"

TODAY="$(date +%F)"
INPUT_FILE="$RUNS_DIR/$TODAY-input.md"
OUTPUT_FILE="$OUT_DIR/$TODAY-decision-brief.md"

if [[ ! -f "$INPUT_FILE" ]]; then
  cat > "$INPUT_FILE" <<'EOT'
# Weekly Decision Input

- problem:
- context:
- options:
  - option_1:
  - option_2:
- deadline:
- success_metric:
- non_negotiables:
EOT
  echo "[init] 已创建本周输入文件: $INPUT_FILE"
  echo "请先填写输入后再次执行本脚本。"
  exit 0
fi

if [[ ! -f "$OUTPUT_FILE" ]]; then
  cp "$TEMPLATE" "$OUTPUT_FILE"
  echo "[ok] 已创建决策简报模板: $OUTPUT_FILE"
else
  echo "[skip] 决策简报已存在: $OUTPUT_FILE"
fi

echo "[next] 建议流程："
echo "1) 打开并完善输入: $INPUT_FILE"
echo "2) 按 deliberation.protocol.md 完成 5 轮讨论"
echo "3) 将结论填写到: $OUTPUT_FILE"
