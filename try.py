import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
print(endpoint)
print(deployment)
print(api_version)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)

chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information.",
            }
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "Sup?"}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hey! Not much, just here to help. What can I do for you today?",
            }
        ],
    },
]

# Include speech result if speech is enabled
messages = chat_prompt

completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_tokens=800,
    temperature=1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False,
)

print(completion.to_json())
