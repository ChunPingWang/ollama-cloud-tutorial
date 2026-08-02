"""用 OpenAI SDK 的相容層接 Ollama Cloud。

適合已經有一套用 OpenAI SDK / LangChain 寫好的程式，不想重寫的情況。

用法:
    python examples/07_openai_compat.py
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url='https://ollama.com/v1',
    api_key=os.environ['OLLAMA_API_KEY'],
)

MODEL = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')

TOOL_SCHEMA = [{
    'type': 'function',
    'function': {
        'name': 'get_temperature',
        'description': '查詢指定城市的目前氣溫',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string', 'description': '城市名稱，例如 Taipei'},
            },
            'required': ['city'],
        },
    },
}]


def get_temperature(city: str) -> str:
    return {'Taipei': '31°C', 'Tokyo': '18°C'}.get(city, '查無資料')


messages = [{'role': 'user', 'content': '台北現在幾度？'}]

completion = client.chat.completions.create(
    model=MODEL, messages=messages, tools=TOOL_SCHEMA,
)
msg = completion.choices[0].message
messages.append(msg)

if msg.tool_calls:
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        print(f'→ 呼叫 {call.function.name}({args})')
        messages.append({
            'role': 'tool',
            'tool_call_id': call.id,          # 相容層用 tool_call_id，不是 tool_name
            'content': get_temperature(**args),
        })

    final = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOL_SCHEMA,
    )
    print(final.choices[0].message.content)
else:
    print(msg.content)
