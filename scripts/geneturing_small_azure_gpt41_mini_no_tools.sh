#!/bin/bash

# Dataset: geneturing_small
# Provider: azure
# Model: gpt-4.1-mini
# Tool Use: False

python -m src.main \
    --dataset_path data/geneturing_small.json \
    --provider azure \
    --model gpt-4.1-mini \
    --output_path results/geneturing_small_azure_gpt41_mini_no_tools.json \
    --no-tool-use