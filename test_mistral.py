import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def call_mistral(prompt):
    url = "https://router.huggingface.co/hf-inference/text-generation/mistralai/Mistral-7B-Instruct-v0.2"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.3,
            "return_full_text": False
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    print("STATUS:", response.status_code)
    print("BODY:")
    print(response.text)

if __name__ == "__main__":
    prompt = (
        "You are a calm, familiar conversational voice.\n"
        "Do not give advice. Keep replies short.\n\n"
        "User said:\n"
        "Aaj din thoda khali sa lag raha hai."
    )

    call_mistral(prompt)
