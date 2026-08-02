"""Agent Loop 的最小骨架：模型自己決定要呼叫幾次工具、什麼時候收工。

用法:
    python examples/03_agent_loop.py
"""

from ollama import ChatResponse

from _client import MODEL, get_client

client = get_client()


def add(a: int, b: int) -> int:
    """把兩個整數相加"""
    return a + b


def multiply(a: int, b: int) -> int:
    """把兩個整數相乘"""
    return a * b


TOOLS = [add, multiply]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

messages = [{'role': 'user', 'content': '請計算 (11434 + 12341) * 412'}]

while True:
    response: ChatResponse = client.chat(
        model=MODEL, messages=messages, tools=TOOLS, think=True,
    )
    messages.append(response.message)

    if response.message.thinking:
        print(f'[思考] {response.message.thinking}\n')
    if response.message.content:
        print(f'[回覆] {response.message.content}\n')

    if not response.message.tool_calls:
        break   # 模型不再要求工具 → 任務結束

    for tc in response.message.tool_calls:
        fn = AVAILABLE.get(tc.function.name)
        if fn is None:
            result = f'錯誤：不存在的工具 {tc.function.name}'
        else:
            print(f'[工具] {tc.function.name}({tc.function.arguments})')
            result = fn(**tc.function.arguments)
            print(f'[結果] {result}\n')
        messages.append({
            'role': 'tool',
            'tool_name': tc.function.name,
            'content': str(result),
        })
