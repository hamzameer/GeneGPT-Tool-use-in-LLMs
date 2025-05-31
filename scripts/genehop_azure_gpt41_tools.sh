#!/bin/bash

# Dataset: genehop
# Provider: azure
# Model: gpt-4.1
# Tool Use: True

python -m src.main \
    --dataset_path data/genehop.json \
    --provider azure \
    --model gpt-4.1 \
    --output_path results/genehop_azure_gpt41_tools.json \
    --tool_use True 