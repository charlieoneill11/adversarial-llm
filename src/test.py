import subprocess
import json
import time

def upload_file(file_path):
    command = f"openai api files.create -p fine-tune -f {file_path}"
    process = subprocess.run(command, capture_output=True, shell=True, text=True)

    if process.stdout:
        try:
            response = json.loads(process.stdout)
            return response["id"]
        except json.JSONDecodeError:
            print("Failed to parse JSON output.")
            print(process.stdout)
            raise
    else:
        print("No output received from file upload command.")
        print(f"stderr: {process.stderr}")
        raise Exception("File upload failed.")

def create_fine_tune(file_id, base_model="curie"):
    command = f"openai api fine_tunes.create -t {file_id} -m {base_model}"
    process = subprocess.run(command, capture_output=True, shell=True, text=True)

    if process.stdout:
        try:
            response = json.loads(process.stdout)
            return response["id"]
        except json.JSONDecodeError:
            print("Failed to parse JSON output.")
            print(process.stdout)
            raise
    else:
        print("No output received from fine-tune creation command.")
        print(f"stderr: {process.stderr}")
        raise Exception("Fine-tuning job creation failed.")

def follow_fine_tune(job_id):
    command = f"openai api fine_tunes.follow -i {job_id}"
    process = subprocess.Popen(command, shell=True)

    while True:
        if process.poll() is not None:
            break
        else:
            time.sleep(15)

    return process.returncode

def main():
    training_file_path = "/Users/charlesoneill/adversarial-llm/data/judge_dataset_prepared_train.jsonl"
    base_model = "curie"

    # Upload the file and get its ID
    file_id = upload_file(training_file_path)
    print(f"File uploaded with ID: {file_id}")

    # Create a fine-tuning job and get its ID
    job_id = create_fine_tune(file_id, base_model)
    print(f"Fine-tuning job created with ID: {job_id}")

    # Follow the progress of the fine-tuning job
    print("Following the progress of the fine-tuning job...")
    return_code = follow_fine_tune(job_id)

    if return_code == 0:
        print("Fine-tuning job completed successfully.")
    else:
        print(f"Fine-tuning job exited with code: {return_code}")

if __name__ == "__main__":
    main()
