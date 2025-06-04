#!/bin/bash

# Dataset: genehop
# Provider: ollama
# Model: qwen3:4b
# Tool Use: False

python -m src.main \
    --dataset_path data/genehop.json \
    --provider ollama \
    --model qwen3:4b \
    --output_path results/genehop_ollama_qwen3_4b_no_tools.json \
    --no-tool-use 
