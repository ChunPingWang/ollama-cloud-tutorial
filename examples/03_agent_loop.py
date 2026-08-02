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

# 沒有這句系統提示的話，gpt-oss:120b 這種等級的模型會直接心算完給你答案，
# 一次工具都不呼叫——迴圈第一輪就結束，看不到 Agent 的行為。
# 實務上這也是真的：能力強的模型會跳過你希望它用的工具，要明確要求。
messages = [
    {'role': 'system', 'content': '你只能透過 add 與 multiply 工具做算術，'
                                  '嚴禁自行心算或直接寫出答案。'},
    {'role': 'user', 'content': '請計算 (11434 + 12341) * 412'},
]

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
