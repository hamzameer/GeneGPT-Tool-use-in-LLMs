#!/bin/bash

# Dataset: geneturing_small
# Provider: ollama
# Model: qwen3:4b
# Tool Use: True

python -m src.main \
    --dataset_path data/geneturing_small.json \
    --provider ollama \
    --model qwen3:4b \
    --output_path results/geneturing_small_ollama_qwen3_4b_tools.json \
    --tool_use True 