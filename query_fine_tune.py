
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read query from file
with open("masked_query.txt", "r", encoding="utf-8") as file:
    user_prompt = file.read()

# Query the fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0125:bangla-ai::BpTFezHI",  # your fine-tuned model ID
    messages=[
        {
            "role": "system",
            "content": "You are a helpful medical assistant."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)

# Output response
print(response.choices[0].message.content)