#!/bin/bash

# Dataset: genehop_small
# Provider: ollama
# Model: qwen3:4b
# Tool Use: False

python -m src.main \
    --dataset_path data/genehop_small.json \
    --provider ollama \
    --model qwen3:4b \
    --output_path results/genehop_small_ollama_qwen3_4b_no_tools.json \
    --no-tool-use 
