# 結構化輸出：讓 Agent 吐出可以直接用的資料

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 雲端不強制 format 的坑，以及兩種實測可行的替代方案

[← 回到核心路徑](../README.md)　·　[全部進階主題](README.md)　·　[串流與 Thinking →](02-streaming.md)

---

Agent 常常是更大流程裡的一環，下游需要的是 JSON 不是散文。

## 1. 先講一個雲端的坑：`format` 不會生效

網路上（含 Ollama 官方文件）教的做法是這個：

```python
response = client.chat(
    model=MODEL,
    messages=[...],
    format=ReviewReport.model_json_schema(),   # ← 在雲端上會被忽略
)
report = ReviewReport.model_validate_json(response.message.content)
```

**這在本地 `ollama serve` 上有效，在 Ollama Cloud 的 hosted API 上無效。** 我實測過三條路（2026-08，`gpt-oss:120b` 與 `gemma4:31b`）：

| 做法 | 結果 |
| --- | --- |
| 原生 `chat(format=<json schema>)` | 被忽略，回傳 Markdown 散文 |
| 原生 `chat(format='json')` | 被忽略，回傳包在 ` ```json ` 圍籬裡的內容 |
| OpenAI 相容層 `response_format={'type':'json_schema',...}` | 被忽略，回傳散文 |

直接 `curl` 打 `https://ollama.com/api/chat` 帶 `format` 也一樣，所以不是 SDK 的問題，是雲端端點沒有實作這個約束。照文件寫的話，你會拿到這個：

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ReviewReport
  Invalid JSON: expected value at line 1 column 1 [input_value='以下是 **payment.py**...']
```

好消息是有兩條路可以走，而且都驗證過能用。

## 2. 方案 A：借 tool calling 來做 schema 約束（推薦）

關鍵觀察：**`format` 在雲端失效，但 `tools` 是確實生效的。** 那就把想要的 schema 包成一個工具，讓模型「呼叫」它——工具參數天生就是 JSON Schema 約束的產物。

```python
def structured_via_tool(prompt: str, schema_model: type[BaseModel],
                        attempts: int = 3) -> BaseModel:
    """把 Pydantic model 包成一個工具，逼模型以工具參數的形式交出結構化資料。"""
    tool = {
        'type': 'function',
        'function': {
            'name': 'submit',
            'description': '提交最終結果。這是唯一的作答方式，必須呼叫。',
            'parameters': schema_model.model_json_schema(),
        },
    }
    messages = [{'role': 'user', 'content': prompt}]

    for attempt in range(1, attempts + 1):
        response = client.chat(
            model=MODEL, messages=messages, tools=[tool],
            options={'temperature': 0},          # 結構化輸出不需要創意
        )

        if response.message.tool_calls:
            args = response.message.tool_calls[0].function.arguments
            try:
                return schema_model.model_validate(args)
            except ValidationError as exc:
                # 把驗證錯誤回饋給模型，讓它自己修
                messages.append(response.message)
                messages.append({
                    'role': 'tool', 'tool_name': 'submit',
                    'content': f'資料格式不符，請修正後重新呼叫 submit：{exc}',
                })
        else:
            messages.append(response.message)
            messages.append({
                'role': 'user',
                'content': '請不要用文字回答，必須呼叫 submit 工具提交結果。',
            })

    raise RuntimeError(f'嘗試 {attempts} 次仍拿不到合法的結構化輸出')
```

用起來就一行：

```python
report = structured_via_tool(f'請審查以下程式碼：\n{code}', ReviewReport)
for f in report.findings:
    print(f'[{f.severity}] {f.file}:{f.line} — {f.issue}')
```

**那個重試迴圈不是裝飾。** 我實測時第一次就沒過，是靠把 `ValidationError` 原文回饋給模型才修正的。這是整個模式的精髓：**Pydantic 的錯誤訊息寫得夠清楚，模型讀得懂，於是驗證器本身變成了一個修正迴路。**

想讓約束更嚴格就用 `Literal`：

```python
severity: Literal['high', 'medium', 'low']   # 模型填 'critical' 會被擋下來要求重填
```

## 3. 方案 B：prompt + 去圍籬 + 驗證（備援）

比較土砲，但少一層工具的間接性，模型比較不會分心：

```python
def _strip_fence(text: str) -> str:
    """模型很愛把 JSON 包在 ```json ... ``` 裡，先拆掉。"""
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    return (match.group(1) if match else text).strip()


messages = [{
    'role': 'user',
    'content': f'{prompt}\n\n只輸出符合以下 JSON Schema 的 JSON，不要任何說明文字：\n{schema}',
}]
response = client.chat(model=MODEL, messages=messages, options={'temperature': 0})
report = ReviewReport.model_validate_json(_strip_fence(response.message.content))
```

`_strip_fence` 是必要的，不是防禦性程式碼——實測中模型**每次**都會加圍籬，即使你叫它不要。

兩個方案的完整版（含重試）都在 `examples/05_structured_output.py`，執行後會兩種都跑一遍給你比較。

## 4. 跟 Agent Loop 怎麼搭

**不要在 Agent Loop 裡同時掛工具跟 `submit`**，模型會搞不清楚該繼續查資料還是該交卷。分兩階段：

1. **蒐集階段**：Agent Loop 帶著真正的工具跑（[〈實戰〉](../README.md#8-實戰一個會讀專案的-codebase-agent)那套），直到模型不再要求工具
2. **收斂階段**：把蒐集到的結論當成 prompt，單獨呼叫一次 `structured_via_tool()` 交出 JSON

多花一次呼叫，換來下游拿得到乾淨的資料，划算。

---

---

[← 回到核心路徑](../README.md)　·　[全部進階主題](README.md)　·　[串流與 Thinking →](02-streaming.md)
