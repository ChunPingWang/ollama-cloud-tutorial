# 用 OpenAI SDK 相容層接既有生態系

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 把既有以 OpenAI 為介面的程式接到 Ollama Cloud

[← 串流與 Thinking](02-streaming.md)　·　[全部進階主題](README.md)　·　[接上 MCP →](04-mcp.md)

---

如果你手上已經有一套用 OpenAI SDK、LangChain、LlamaIndex 寫好的東西，不用重寫——Ollama 有 OpenAI 相容端點。

```python
from openai import OpenAI

client = OpenAI(
    base_url='https://ollama.com/v1',
    api_key=os.environ['OLLAMA_API_KEY'],
)

completion = client.chat.completions.create(
    model='gpt-oss:120b',
    messages=[{'role': 'user', 'content': '你好'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'get_temperature',
            'description': '查詢城市氣溫',
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        },
    }],
)
```

完整可執行版本在 `examples/07_openai_compat.py`。相容層支援 tool calling、streaming、vision、JSON mode，以及推理模型的 `reasoning_effort` 參數。

有個容易踩的差異：**把工具結果塞回去時，原生 SDK 用 `tool_name`，相容層用 `tool_call_id`**。兩邊的訊息格式不能混用。

**該用哪一個？** 如果是新專案，我會建議用原生 `ollama` SDK：函式自動轉 schema 這件事省下的樣板程式碼很可觀，而且 `think` 參數在原生 SDK 裡更直觀。相容層留給「要接既有框架」或「想保留隨時換供應商的彈性」的情況。

---

---

[← 串流與 Thinking](02-streaming.md)　·　[全部進階主題](README.md)　·　[接上 MCP →](04-mcp.md)
