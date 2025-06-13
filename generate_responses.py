import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai_api import OpenAIClient  # from tutorbot-dpo repo

# Load API credentials
load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")

# Load prompts from CSV
df = pd.read_csv("prompts.csv")

# 🧪 Diagnostic prints
print("📝 Total rows loaded from CSV:", len(df))
print("📋 Sample row (dict view):")
print(df.head(1).to_dict())

# TEMP: Skip filtering to force testing
df = df.head(5)  # use only first 5 for testing

# Initialize OpenAI client
client = OpenAIClient(use_azure_client=True)
model = "gpt-4o"

# Build prompt batches
batched_inputs = []
metadata_list = []

for _, row in df.iterrows():
    prompt_a = row["user"]
    prompt_b = prompt_a + "\n(Add a hint for better understanding.)"
    system_prompt = row["system"]

    # Variant A
    batched_inputs.append([
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_a}]}
    ])
    metadata_list.append({**row.to_dict(), "variant": "A"})

    # Variant B
    batched_inputs.append([
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt_b}]}
    ])
    metadata_list.append({**row.to_dict(), "variant": "B"})

# Generation args
generation_args = {
    "max_tokens": 500,
    "temperature": 0.3,
}

# Run generation
print(f"Calling Azure OpenAI API for {len(batched_inputs)} prompts...")
responses = client.get_batched_responses(
    batched_inputs,
    model=model,
    batch_size=10,
    generation_args=generation_args,
    show_progress=True
)

# Combine responses with metadata
final_data = []
for meta, resp in zip(metadata_list, responses):
    final_data.append({
        **meta,
        "prompt": meta["user"],
        "response": resp,
        "model": model
    })

# Save
with open("generated_responses.json", "w") as f:
    json.dump(final_data, f, indent=2)

pd.DataFrame(final_data).to_csv("generated_responses.csv", index=False)
print("All responses saved to 'generated_responses.csv'")
