#!/usr/bin/env bash
# Serve an open model with an OpenAI-compatible API for run_eval.py.
# Usage: bash scripts/serve_vllm.sh <hf-model> [port] [tensor-parallel-size]
# Longest ToolHaystack prompts are ~24K tokens, so a 32K context is needed.
set -euo pipefail

MODEL=${1:?usage: serve_vllm.sh <hf-model> [port] [tp]}
PORT=${2:-8000}
TP=${3:-1}

vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --enable-prefix-caching
