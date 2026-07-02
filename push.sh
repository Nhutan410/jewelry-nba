#!/bin/bash
set -e

# ── Đọc commit message từ tham số, mặc định nếu không truyền ──────────────────
MSG="${1:-chore: update codebase}"

echo "==> Staging all changes..."
git add \
  .gitignore \
  app_instore.py \
  generate_instore_scripts.py \
  requirements.txt \
  data/customer_data_poc_enhanced.xlsx \
  outputs/instore_scripts.json \
  outputs/nba_results_llm.xlsx \
  outputs/nba_messages.json \
  src/instore_script_engine.py \
  src/llm_message_generator.py \
  src/nba_engine_llm.py \
  src/pipeline_llm.py

echo "==> Committing: \"$MSG\""
git commit -m "$MSG"

echo "==> Pushing to origin/main..."
git push origin main

echo "✓ Done! https://github.com/Nhutan410/jewelry-nba"
