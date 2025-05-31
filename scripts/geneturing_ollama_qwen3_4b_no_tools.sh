#!/bin/bash

# Dataset: geneturing
# Provider: ollama
# Model: qwen3:4b
# Tool Use: False

python -m src.main \
    --dataset_path data/geneturing.json \
    --provider ollama \
    --model qwen3:4b \
    --output_path results/geneturing_ollama_qwen3_4b_no_tools.json \
    --tool_use False 