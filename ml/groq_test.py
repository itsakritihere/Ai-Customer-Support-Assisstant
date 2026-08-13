import os

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


# Create Groq client
client = Groq(
    api_key=api_key
)


# Send a test request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Write a short friendly customer support greeting."
        }
    ],
    temperature=0.3
)


# Display response
print("\nGroq Response:")
print(response.choices[0].message.content)