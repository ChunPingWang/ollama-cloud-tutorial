"""第一次呼叫 Ollama Cloud。

用法:
    python examples/01_hello_cloud.py
"""

import os
import sys

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
)

prompt = sys.argv[1] if len(sys.argv) > 1 else '用三句話解釋什麼是 AI Agent。'
messages = [{'role': 'user', 'content': prompt}]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
    print(part['message']['content'], end='', flush=True)
print()
