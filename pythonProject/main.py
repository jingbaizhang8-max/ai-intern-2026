import os
from dotenv import load_dotenv
from openai import OpenAI
from pyexpat.errors import messages

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# response = client.chat.completions.create(
#     model = "deepseek-flash",
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful AI assistant."
#         },
#         {
#             "role": "user",
#             "content": "Explain what an API is in simple terms."
#         }
#     ],
#     stream = False
# )
#
# print(response.choices[0].message.content)

def get_ai_reply(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-flash",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"API ERROR: {e}")
        return None



messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break
    messages_user = {
        "role": "user",
        "content": user_input
    }

    messages.append(messages_user)
    print(messages)
    ai_reply = get_ai_reply(messages)
    if ai_reply is None:
        continue
    print(f"AI: {ai_reply}")
    messages.append(
        {
            "role": "assistant",
            "content": ai_reply
        }
    )
