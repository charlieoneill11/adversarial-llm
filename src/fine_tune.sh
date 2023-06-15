#!/bin/bash

# Accept the current model as a command line argument
current_model="$1"

echo "Current model: $current_model"

# Define the training file path
training_file_path="/Users/charlesoneill/adversarial-llm/data/judge_dataset_prepared_train.jsonl"

# Create a fine-tune job in the background and get the job id
echo "Creating fine-tune job..."
create_output=$(openai api fine_tunes.create -t $training_file_path -m $current_model)
job_id=$(echo $create_output | python -c "import sys, json; print(json.load(sys.stdin)['id'])")

echo "Job ID: $job_id"

# Follow the fine-tune job progress
echo "Following the progress of the fine-tuning job..."
openai api fine_tunes.follow -i "$job_id"

# Get the fine-tuned model's id and return it
# model_id=$(openai api fine_tunes.retrieve -i $job_id | python -c "import sys, json; print(json.load(sys.stdin)['fine_tuned_model'])")
# echo "Fine-tuned model: $model_id"
# echo $model_id