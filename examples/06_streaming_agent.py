"""串流版的 Agent Loop：邊跑邊顯示思考與回覆。

重點是 tool_calls 分片送來時要用 extend 累積，不能用賦值覆蓋。

用法:
    python examples/06_streaming_agent.py
"""

from _client import MODEL, get_client

client = get_client()

GREY, CYAN, RESET = '\033[90m', '\033[36m', '\033[0m'


def get_temperature(city: str) -> str:
    """查詢指定城市的目前氣溫。

    Args:
        city: 城市名稱，例如 Taipei、Tokyo
    """
    return {'Taipei': '31°C', 'Tokyo': '18°C', 'London': '15°C'}.get(city, '查無資料')


def get_conditions(city: str) -> str:
    """查詢指定城市的天氣狀況。

    Args:
        city: 城市名稱，例如 Taipei、Tokyo
    """
    return {'Taipei': '晴時多雲', 'Tokyo': '陰', 'London': '下雨'}.get(city, '查無資料')


TOOLS = [get_temperature, get_conditions]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

messages = [{'role': 'user', 'content': '台北和東京現在的氣溫與天氣如何？哪個比較適合出門？'}]

while True:
    stream = client.chat(
        model=MODEL, messages=messages, tools=TOOLS, stream=True, think=True,
    )

    thinking, content, tool_calls = '', '', []
    done_thinking = False

    for chunk in stream:
        if chunk.message.thinking:
            thinking += chunk.message.thinking
            print(f'{GREY}{chunk.message.thinking}{RESET}', end='', flush=True)
        if chunk.message.content:
            if not done_thinking:
                done_thinking = True
                print('\n')
            content += chunk.message.content
            print(chunk.message.content, end='', flush=True)
        if chunk.message.tool_calls:
            tool_calls.extend(chunk.message.tool_calls)   # 累積，不要覆蓋

    messages.append({
        'role': 'assistant',
        'thinking': thinking,
        'content': content,
        'tool_calls': tool_calls,
    })

    if not tool_calls:
        print()
        break

    for call in tool_calls:
        name, args = call.function.name, call.function.arguments
        print(f'\n{CYAN}[工具] {name}({args}){RESET}')
        fn = AVAILABLE.get(name)
        result = fn(**args) if fn else f'錯誤：不存在的工具 {name}'
        print(f'{GREY}  → {result}{RESET}')
        messages.append({
            'role': 'tool', 'tool_name': name, 'content': str(result),
        })
