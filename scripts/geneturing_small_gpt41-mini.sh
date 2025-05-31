#!/bin/bash

# Run the script with the following arguments:
# --dataset_path data/geneturing_small.json
# --provider azure
# --model gpt-4.1-mini
# --output_path results/geneturing_small_gpt41-mini.json

python src/main.py --dataset_path data/geneturing_small.json --provider azure --model gpt-4.1-mini --output_path results/geneturing_small_gpt41-mini.json