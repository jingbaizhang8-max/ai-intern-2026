import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

client=OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

def multiply(a,b):
    return a * b

def add(a,b):
    return a + b

tool_registry = {
    "multiply": multiply,
    "add": add
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["a","b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

user_input = input("Describe your question: ")

messages = [
    {
        "role": "system",
        "content": """
        You are an AI assistant used in a Python Tool Calling demo.
        
        Do not claim to be Claude, ChatGPT, or any other specific assistant.
        If asked who you are, simply say that you are an AI assistant.
        
        Use tools when they are needed.
        If no tool is needed, answer directly.
        """
    },
    {
        "role": "user",
        "content": f"{user_input}"
    }
]

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools
)

message= response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(function_name)
    print(arguments)
    print(tool_call.id)

    #########################################
    # if function_name == "multiply":
    #     result = multiply(
    #         arguments["a"],
    #         arguments["b"]
    #     )
    # elif function_name == "add":
    #     result = add(
    #         arguments["a"],
    #         arguments["b"]
    #     )
    #########################################
    #replace
    if function_name in tool_registry:
        tool_function = tool_registry[function_name]
        result = tool_function(**arguments)
    else:
        raise ValueError(f"Unknown tool: {function_name}")



    print(result)

    messages.append(
        {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        }
    )

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        }
    )

    final_response = client.chat.completions.create(
        model = "deepseek-flash",
        messages=messages,
        tools = tools
    )

    final_answer = final_response.choices[0].message.content
    print(f"AI: {final_answer}")
else:
    print(f"AI: {message.content}")