# generate_responses.py

import json
import pandas as pd
import os
from dotenv import load_dotenv
from openai_api import OpenAIClient  # from tutorbot-dpo repo

#  Load environment variables from .env
load_dotenv()

# Explicitly set Azure OpenAI environment variables
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

# Load prompt JSON (generated from processed_prompting_data.csv)
with open("prompts.json", "r") as f:
    prompts = json.load(f)

#  Initialize Azure OpenAI Client
client = OpenAIClient(use_azure_client=True)

#  Choose your Azure deployed model name
model = "gpt-4o"  # change to your actual deployment name if different

#  Prepare A/B prompt variants
batched_inputs = []
metadata_list = []

for sample in prompts:
    prompt_a = sample["user"]
    prompt_b = sample["user"] + "\n(Add a hint for better understanding.)"

    # A variant
    batched_inputs.append([
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": prompt_a}
    ])
    metadata_list.append({**sample, "variant": "A"})

    # B variant
    batched_inputs.append([
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": prompt_b}
    ])
    metadata_list.append({**sample, "variant": "B"})

#  Generation settings
generation_args = {
    "max_tokens": 500,
    "temperature": 0.3,
    "response_format": {"type": "text"}  # or use "json_object" if expecting structured outputs
}

#  Make batched API calls via Azure
print(f" Sending {len(batched_inputs)} prompts to Azure OpenAI...")
responses = client.get_batched_responses(
    batched_inputs,
    model=model,
    batch_size=10,
    generation_args=generation_args,
    show_progress=True
)

# 🧾 Combine metadata + LLM responses
final_data = []
for meta, resp in zip(metadata_list, responses):
    final_data.append({
        **meta,
        "prompt": meta["user"],
        "response": resp,
        "model": model
    })

# 💾 Save outputs to JSON and CSV
with open("generated_responses.json", "w") as f:
    json.dump(final_data, f, indent=2)

pd.DataFrame(final_data).to_csv("generated_responses.csv", index=False)
print("All responses saved to 'generated_responses.csv'")
