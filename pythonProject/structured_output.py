import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal
from pydantic import BaseModel, ValidationError

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")


class TechnologyInfo(BaseModel):
    name: str
    definition: str
    difficulty: Literal["beginner", "intermediate", "advanced"]
    keywords: list[str]


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

user_input = input("Enter a technology: ")

messages = [
    {
        "role": "system",
        "content": """
        You are a technical tutor.

        Return your answer as JSON.

        The JSON must contain:
        - name: the technology name
        - definition: a simple definition
        - difficulty: beginner, intermediate, or advanced
        - keywords: a list of important keywords

        
        
        """
    },
    {
        "role": "user",
        "content": f"Explain {user_input}"
    }
]

# response = client.chat.completions.create(
#     model="deepseek-flash",
#     messages=messages,
#     stream=False
# )
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    response_format={
        "type": "json_object"
    },
    stream=False
)

ai_reply = response.choices[0].message.content
data = json.loads(ai_reply)
try:
    data = json.loads(ai_reply)
    technology = TechnologyInfo.model_validate(data)

    print(f"Name: {technology.name}")
    print(f"Definition: {technology.definition}")
    print(f"Difficulty: {technology.difficulty}")
    print(f"Keywords: {technology.keywords}")

except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")

except ValidationError as e:
    print(f"Validation failed: {e}")

