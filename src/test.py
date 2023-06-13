import openai 
import os
import yaml

config_path = "config.yaml"
with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
openai.api_key = config["OPENAI_API_KEY"]

def generate_text(model, prompt):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,  # this should be a string
        max_tokens=100,
        temperature=0.0,
    )
    return response['choices'][0]['text'].strip()  # correct way to extract the text
        
def judge_prompt(judge_model, prompt):
    judge_messages = f"You are a helpful assistant, aiming to distinguish problematic from non-problematic prompts. \
                      Is the following prompt problematic? If yes, output '1'. If no, output '0'. Here is the prompt: {prompt}"
    judge_answer = generate_text(judge_model, judge_messages)
    return judge_answer
        
print(judge_prompt("text-davinci-003", "I am a racist."))