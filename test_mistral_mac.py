print("=== SCRIPT STARTED ===")

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

messages = [
    {
        "role": "system",
        "content": (
            "You are a calm, familiar conversational voice. "
            "Do not give advice. Keep replies short."
        )
    },
    {
        "role": "user",
        "content": "Aaj din thoda khali sa lag raha hai."
    }
]

response = client.chat_completion(
    messages=messages,
    max_tokens=120,
    temperature=0.3
)

print("MODEL RESPONSE:")
print(response.choices[0].message.content)

print("=== SCRIPT ENDED ===")
