#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>"
  exit 1
fi

for ((i=1; i<=$1; i++)); do
  echo "=== Ralph iteration $i / $1 ==="

  result=$(claude --dangerously-skip-permissions -p "@progress.txt @SPEC.md @CONTEXT.md\
  1. Read the PRD and progress file. \
  2. Find the highest-priority incomplete task and implement it. I want to build this new hybridmodels package. \
  3. Run the test suite and type checks (uv run pytest, uv run ruff check). \
  4. Update the PRD with what was done. \
  5. Append your progress to progress.txt. \
  6. Commit your changes. \
  ONLY WORK ON A SINGLE TASK. \
  If the PRD is complete, output <promise>COMPLETE</promise>.")

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo "PRD complete after $i iterations."
    exit 0
  fi
done
