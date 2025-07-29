#!/bin/bash

# Set your API key, file paths, and parameters

API_KEY="your_api_key"
MODEL_NAME="gpt-4o" #Set the model name
NUM_SAMPLES=-1  # Set -1 to process all samples
INPUT_FILE="data/test.xlsx" # Set the input file name
OUTPUT_FILE="results/test.xlsx" # Set the output result file name

echo "Starting the content number detection task for *${NUM_SAMPLES}* samples by model *${MODEL_NAME}*..."

python3 number_detection.py \
  --API_KEY "${API_KEY}" \
  --model_name "${MODEL_NAME}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --num_samples "${NUM_SAMPLES}"

echo "Task completed. Results saved to ${OUTPUT_FILE}."
