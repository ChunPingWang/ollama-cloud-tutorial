"""最小的 tool calling 範例：模型要求呼叫工具，我們執行後把結果餵回去。

用法:
    python examples/02_tool_calling.py
"""

from _client import MODEL, get_client

client = get_client()


def get_temperature(city: str) -> str:
    """查詢指定城市的目前氣溫。

    Args:
        city: 城市名稱，例如 Taipei、Tokyo

    Returns:
        該城市的氣溫字串
    """
    table = {'Taipei': '31°C', 'Tokyo': '18°C', 'London': '15°C'}
    return table.get(city, '查無資料')


messages = [{'role': 'user', 'content': '台北現在幾度？'}]

# 第一輪：模型決定要呼叫工具
response = client.chat(model=MODEL, messages=messages, tools=[get_temperature])
messages.append(response.message)

if response.message.tool_calls:
    for call in response.message.tool_calls:
        print(f'→ 模型要求呼叫 {call.function.name}({call.function.arguments})')
        result = get_temperature(**call.function.arguments)
        messages.append({
            'role': 'tool',
            'tool_name': call.function.name,
            'content': str(result),
        })

    # 第二輪：把工具結果餵回去，模型產生最終回答
    final = client.chat(model=MODEL, messages=messages, tools=[get_temperature])
    print(final.message.content)
else:
    print(response.message.content)
