# 接框架與可觀測性：LangChain / Langfuse / LangSmith

> **難度**：進階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 什麼時候該用框架，以及怎麼看見 Agent 在想什麼

[← 正式上線前要處理的事](07-production.md)　·　[全部進階主題](README.md)　·　[多輪對話與 Context 管理 →](09-memory-context.md)

---

前面十二節都是手刻的。這節談什麼時候該把工作交給框架，以及一個比框架更該優先做的事——**可觀測性**。

## 1. 先講判準：什麼時候該用框架

我的看法是：**先手刻過一次，再決定要不要用框架。** 你已經讀到這裡了，[〈手刻 Agent Loop〉](../README.md#7-手刻-agent-loop)那 40 行迴圈你看得懂，那麼任何框架對你來說都只是那個迴圈的包裝，不會變成黑盒子。

該用框架的訊號：

- 需要 **checkpoint / 中斷續跑**（LangGraph 的 `checkpointer`）
- 需要 **人在迴圈中**（`interrupt_before`）
- 團隊已經有一整套 LangChain 的 retriever、memory、chain 要沿用

不該用的訊號：

- 只是要跑個工具迴圈——手刻的 40 行更好懂、更好改、debug 時堆疊更淺
- 想要框架幫你解決模型能力問題——它不會

## 2. LangChain 接 Ollama Cloud

關鍵只有兩個參數：`base_url` 指到 `https://ollama.com`，`client_kwargs` 塞認證 header。

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='gpt-oss:120b',
    base_url='https://ollama.com',
    client_kwargs={'headers': {'Authorization': f'Bearer {api_key}'}},
)
```

工具呼叫正常運作：

```python
reply = llm.bind_tools([get_temperature]).invoke('台北現在幾度？')
# → tool_calls: [{'name': 'get_temperature', 'args': {'city': 'Taipei'}, ...}]
```

`create_agent` 也能直接跑完整個 Agent Loop：

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm, tools=TOOLS,
    system_prompt='你是氣象助理，必須透過工具查詢，不可自行編造數據。',
)
result = agent.invoke({'messages': [{'role': 'user', 'content': '台北和東京哪個適合出門？'}]})
```

> **讀 `result['messages']` 的小陷阱**：中途的 `AIMessage` 只帶 `tool_calls`、`content` 是空字串。直接取 `messages[-1].content` 有機會拿到空的，要往回找第一個 `content` 非空的 `AIMessage`。

#### `with_structured_output()` 在雲端會壞掉

這是[〈結構化輸出〉](01-structured-output.md)那個坑的連鎖效應，**值得單獨標出來**：

```python
llm.with_structured_output(Country).invoke('Tell me about Canada.')
# ❌ OutputParserException: Invalid json output: ## Canada – A Snapshot | Aspect | ...
```

原因是它預設走 JSON mode，而 Ollama Cloud 不強制 `format`。解法是明講走 function calling——等於[〈結構化輸出〉](01-structured-output.md)的方案 A，只是由 LangChain 代勞：

```python
structured = llm.with_structured_output(Country, method='function_calling')
structured.invoke('Tell me about Canada. 請務必呼叫工具回報結果。')
# ✅ name='Canada' capital='Ottawa'
```

那句「請務必呼叫工具回報結果」不是廢話。我第一次沒加，模型不呼叫工具，結果是 `None`——不是例外，是**靜默的 None**，比拋錯還難查。

#### 順手處理偶發 500

實測跑 `create_agent` 時撞到這個：

```
ollama._types.ResponseError: Internal Server Error (ref: 75e95caf-...) (status code: 500)
During task with name 'model' and id '70c8e2cf-...'
```

Agent 輪數越多，撞上的機率越高——整段就這樣炸掉了。`ChatOllama` 沒有 `max_retries`，但整個 agent 是 Runnable，包一層就好：

```python
resilient_agent = agent.with_retry(stop_after_attempt=3)
```

完整範例在 `examples/09_langchain_agent.py`。

## 3. Langfuse：把「可觀測性」從口號變成東西

[〈正式上線前要處理的事〉](07-production.md)說「至少把每輪的 `tool_calls` 和 `thinking` 記下來」。Langfuse 就是拿來做這件事的：開源、可自架、與供應商無關。

用 `@observe` 標記，三種 `as_type` 對應 trace 上三種節點：

```python
from langfuse import get_client, observe

langfuse = get_client()


@observe(as_type='tool')          # trace 上顯示成工具節點
def add(a: int, b: int) -> int:
    """把兩個整數相加"""
    return a + b


@observe(as_type='generation', name='ollama-chat')    # 有 model / usage 欄位
def call_model(messages: list) -> object:
    response = client.chat(model=MODEL, messages=messages, tools=TOOLS, think=True)

    usage = {}
    for src, dst in [('prompt_eval_count', 'input'), ('eval_count', 'output')]:
        value = getattr(response, src, None)
        if value is not None:
            usage[dst] = value

    # 把 thinking 也送上去——這是事後除錯唯一的線索來源
    langfuse.update_current_generation(
        model=MODEL,
        input=messages,
        output={
            'content': response.message.content,
            'thinking': response.message.thinking,
            'tool_calls': [
                {'name': tc.function.name, 'arguments': tc.function.arguments}
                for tc in (response.message.tool_calls or [])
            ],
        },
        usage_details=usage or None,
    )
    return response


@observe(as_type='agent', name='math-agent')          # 整趟任務的根節點
def run_agent(question: str, max_turns: int = 8) -> str:
    ...
```

**`langfuse.flush()` 一定要呼叫。** 短命的腳本跑完就結束行程，trace 還在緩衝區裡沒送出去，你會對著空的 dashboard 懷疑人生。

`prompt_eval_count` / `eval_count` 是 Ollama 回應裡的 token 數。雖然 Ollama Cloud 是按 GPU 時間計費、token 不直接對應帳單，但它是**你能拿到最好的 context 膨脹指標**——Agent 跑到第幾輪開始失控，看這條線最準。

實跑一趟三輪的算術 Agent，Langfuse 收到 6 個 span：

```
● math-agent    type=agent       turns_used: 3
● ollama-chat   type=generation  model=gpt-oss:120b  usage={"input":200,"output":92}
● add           type=tool        input={"a":11434,"b":12341}  output=23775
● ollama-chat   type=generation  usage={"input":297,"output":41}
● multiply      type=tool        input={"a":23775,"b":412}   output=9795300
● ollama-chat   type=generation  usage={"input":351,"output":23}
```

注意 input token 從 200 → 297 → 351 一路長——**這就是 context 累積的樣子**，[〈實戰〉](../README.md#8-實戰一個會讀專案的-codebase-agent)講的「工具輸出一定要截斷」在這張圖上看得最清楚。

接 LangChain 的話更省事，掛個 callback 就好：

```python
from langfuse.langchain import CallbackHandler

agent.invoke({...}, config={'callbacks': [CallbackHandler()]})
```

實測會自動抓到 17 個 span（`ChatOllama` generation ×6、`tools` chain ×3、各工具節點、`LangGraph` 根節點），完全不用改業務程式碼。

完整範例在 `examples/10_langfuse_tracing.py`。

## 4. 兩種 Langfuse 部署方式

**雲端版**（最快上手）：到 [cloud.langfuse.com](https://cloud.langfuse.com) 開專案拿金鑰，三個環境變數就完事。

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

**自架版**（資料不出境）：官方 compose 檔會起 Langfuse web/worker、PostgreSQL、ClickHouse、Redis、MinIO 一整套。

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d          # 起來後開 http://localhost:3000 建帳號與專案
```

然後把 `LANGFUSE_HOST` 指到 `http://localhost:3000` 即可，程式碼一行都不用改。

> **誠實標註**：雲端版與自架版我都沒有在寫這篇時實際部署（自架那套資源吃很重，這台開發機沒有 compose plugin）。上面的設定照官方文件寫。**但我確實驗證了 trace 真的送得出去、內容正確**——用的是下面這招。

**第三條路：不用帳號也能驗證**

我寫了一台假的 Langfuse 接收端 `examples/tools/fake_langfuse_server.py`，把收到的 OTLP 內容解碼印出來。要確認「我的 instrumentation 有沒有寫對」，這個比起一整套 compose 快得多：

```bash
# 終端機 A
python examples/tools/fake_langfuse_server.py

# 終端機 B
export LANGFUSE_HOST=http://localhost:3999
export LANGFUSE_PUBLIC_KEY=pk-lf-fake
export LANGFUSE_SECRET_KEY=sk-lf-fake
python examples/10_langfuse_tracing.py
```

上一節那六個 span 就是這樣抓出來的。

> 踩雷紀錄：Langfuse SDK v4 是 OpenTelemetry 架構，**預設用 protobuf 而不是 JSON** 送到 `/api/public/otel/v1/traces`。第一版的假 server 只會 `json.loads()`，收到的是一團二進位就以為沒收到。要裝 `opentelemetry-proto` 解碼才看得懂。

## 5. LangSmith：什麼時候才選它

LangSmith 是 LangChain 官方的託管觀測平台，功能定位跟 Langfuse 高度重疊。它其實**已經隨 `langchain-core` 一起裝進來了**，要開只是兩個環境變數：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
```

設完之後所有 LangChain 呼叫自動上報，不用改程式。手刻的 Agent Loop 則要用 `@traceable` 包：

```python
from langsmith import traceable

@traceable(run_type='tool')
def add(a: int, b: int) -> int:
    return a + b
```

**選擇建議**：

| | Langfuse | LangSmith |
| --- | --- | --- |
| 授權 | 開源，可自架 | 閉源 SaaS |
| 資料落地 | 自架則完全可控 | 上 LangSmith 的伺服器 |
| 與框架的關係 | 與供應商無關，手刻也好接 | 圍繞 LangChain 設計 |
| 上手成本 | 三個環境變數 | 兩個環境變數 |

如果你整套都在 LangChain 上、又不介意資料上雲，LangSmith 是最省事的。**其餘情況我會選 Langfuse**——特別是本文這種手刻 Agent Loop 的做法，Langfuse 的 `@observe` 比 `@traceable` 更貼合，而且自架這個選項在有合規要求時是決定性的。

兩個都不要裝的話也還有辦法：把每輪的 `tool_calls` 和 `thinking` 寫進結構化 log，一樣能事後追。只是沒有 UI，苦一點。

---

---

[← 正式上線前要處理的事](07-production.md)　·　[全部進階主題](README.md)　·　[多輪對話與 Context 管理 →](09-memory-context.md)
