"""第一次呼叫 Ollama Cloud。

【這支教你什麼】確認雲端連得上，並看到串流輸出的樣子
【前置知識】00_check_setup.py 跑過且全綠
【下一支】02_tool_calling.py：讓模型能對外部做事

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
