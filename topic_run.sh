#!/bin/bash

# Set API key, file paths, and parameters
API_KEY="your_api_key"
MODEL_NAME="gpt-4o" ##gpt-4; gpt-4o; gpt-4o-mini
NUM_SAMPLES=-1  # Set -1 to process all samples
INPUT_FILE="data/test.xlsx" # Set the input file name
OUTPUT_FILE="results_0403/test_topics_${MODEL_NAME}_${NUM_SAMPLES}.xlsx" # Set the output file name
PROMPT_FILE="prompts/topic_prompt.txt"

echo "Starting the content topic assignment task for *${NUM_SAMPLES}* samples by model *${MODEL_NAME}*..."

python3 topic_assignment.py \
  --API_KEY "${API_KEY}" \
  --model_name "${MODEL_NAME}" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --num_samples "${NUM_SAMPLES}" \
  --prompt_file "${PROMPT_FILE}"

echo "Task completed. Results saved to ${OUTPUT_FILE}."
