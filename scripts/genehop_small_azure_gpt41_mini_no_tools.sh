#!/bin/bash

# Dataset: genehop_small
# Provider: azure
# Model: gpt-4.1-mini
# Tool Use: False

python -m src.main \
    --dataset_path data/genehop_small.json \
    --provider azure \
    --model gpt-4.1-mini \
    --output_path results/genehop_small_azure_gpt41_mini_no_tools.json \
    --no-tool-use # Explicitly disable tool use 