#!/bin/bash

# Dataset: genehop_small
# Provider: azure
# Model: gpt-4.1
# Tool Use: True

python -m src.main \
    --dataset_path data/genehop_small.json \
    --provider azure \
    --model gpt-4.1 \
    --output_path results/genehop_small_azure_gpt41_tools.json \
    --tool-use