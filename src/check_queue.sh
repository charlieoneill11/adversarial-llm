#!/bin/bash

# Fetch the fine-tuning jobs
output=$(openai api fine_tunes.list)

# Parse the JSON and find the most recent job that hasn't succeeded
pending_job=$(echo $output | jq '.data[] | select(.status!="succeeded") | .id' | head -1)

# If no job is found, print a message
if [ -z "$pending_job" ]; then
  echo "None in the queue."
else
  # Remove the quotes around the job ID
  pending_job=${pending_job%\"}
  pending_job=${pending_job#\"}
  echo "The most recent fine-tuning job that hasn't succeeded has ID: $pending_job"
  # Follow the fine-tune job progress
  echo "Following the progress of the fine-tuning job..."
  openai api fine_tunes.follow -i "$pending_job"
fi