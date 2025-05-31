#!/bin/bash

# Run the script with the following arguments:
# --dataset_path data/geneturing_small.json
# --provider ollama
# --model qwen3:4b
# --output_path results/geneturing_small_qwen3.json

python src/main.py --dataset_path data/geneturing_small.json --provider ollama --model qwen3:4b --output_path results/geneturing_small_qwen3.json