#!/bin/bash

# Dataset: geneturing
# Provider: azure
# Model: gpt-4.1-mini
# Tool Use: False

python -m src.main \
    --dataset_path data/geneturing.json \
    --provider azure \
    --model gpt-4.1-mini \
    --output_path results/geneturing_azure_gpt41_mini_no_tools.json \
    --tool_use False 