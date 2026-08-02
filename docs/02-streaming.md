# 串流與 Thinking：把思考過程秀出來

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 邊跑邊顯示，以及串流時累積 tool_calls 的正確寫法

[← 結構化輸出](01-structured-output.md)　·　[全部進階主題](README.md)　·　[用 OpenAI SDK 相容層接既有生態系 →](03-openai-compat.md)

---

Agent 任務動輒跑十幾秒，介面上一片空白使用者會以為當機。串流可以邊做邊顯示。

有工具的串流比較麻煩，因為 `tool_calls` 是分片送來的，要自己累積：

```python
while True:
    stream = client.chat(
        model=MODEL, messages=messages, tools=TOOLS, stream=True, think=True,
    )

    thinking, content, tool_calls = '', '', []
    done_thinking = False

    for chunk in stream:
        if chunk.message.thinking:
            thinking += chunk.message.thinking
            print(f'\033[90m{chunk.message.thinking}\033[0m', end='', flush=True)
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
        break

    for call in tool_calls:
        result = AVAILABLE[call.function.name](**call.function.arguments)
        messages.append({
            'role': 'tool', 'tool_name': call.function.name, 'content': str(result),
        })
```

重點是 `tool_calls.extend(...)` 而不是 `=`。一次回應可能包含多個平行工具呼叫，用賦值會只剩最後一個。

完整版在 `examples/06_streaming_agent.py`。

---

---

[← 結構化輸出](01-structured-output.md)　·　[全部進階主題](README.md)　·　[用 OpenAI SDK 相容層接既有生態系 →](03-openai-compat.md)
