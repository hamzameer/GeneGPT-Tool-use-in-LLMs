#!/bin/bash

# Dataset: genehop
# Provider: azure
# Model: gpt-4.1-mini
# Tool Use: True

python -m src.main \
    --dataset_path data/genehop.json \
    --provider azure \
    --model gpt-4.1-mini \
    --output_path results/genehop_azure_gpt41_mini_tools.json \
    --tool-use 
