# 用 Ollama Cloud 從零打造一個 AI Agent

> 一篇可以照著跑完的實作文。從註冊、第一次呼叫，一路做到一個會自己決定用哪個工具、能連續多輪推理的 Agent。
> 全部程式碼都在 `examples/` 底下，複製貼上就能執行。

---

## 目錄

1. [為什麼是 Ollama Cloud](#1-為什麼是-ollama-cloud)
2. [兩種連線模式，先搞懂差別](#2-兩種連線模式先搞懂差別)
3. [環境準備](#3-環境準備)
4. [Hello Cloud：第一次呼叫](#4-hello-cloud第一次呼叫)
5. [Agent 的心臟：Tool Calling](#5-agent-的心臟tool-calling)
6. [手刻 Agent Loop](#6-手刻-agent-loop)
7. [實戰：一個會讀專案的 Codebase Agent](#7-實戰一個會讀專案的-codebase-agent)
8. [結構化輸出：讓 Agent 吐出可以直接用的資料](#8-結構化輸出讓-agent-吐出可以直接用的資料)
9. [串流與 Thinking：把思考過程秀出來](#9-串流與-thinking把思考過程秀出來)
10. [用 OpenAI SDK 相容層接既有生態系](#10-用-openai-sdk-相容層接既有生態系)
11. [接上 MCP：不用自己寫工具](#11-接上-mcp不用自己寫工具)
12. [正式上線前要處理的事](#12-正式上線前要處理的事)
13. [結語](#13-結語)

---

## 1. 為什麼是 Ollama Cloud

寫 Agent 最痛的一件事，是**模型能力**跟**部署成本**互相拉扯。

Agent 需要模型會穩定地做工具呼叫（tool calling）、能規劃多步驟、輸出格式不會亂跑。這種能力通常只有中大型模型才做得好。但一個 120B 的模型要在自己機器上跑，光 VRAM 就先勸退大部分人。

Ollama Cloud 做的事很直白：**維持 Ollama 原本的 API 與開發體驗，把模型推論丟到雲端 GPU 上跑。**

實際帶來的好處：

- **本地開發不用改架構**。原本用 `ollama.chat()` 寫的程式，換個 model 名稱就跑在雲上了。
- **模型選擇一次到位**。`gpt-oss:120b`、`qwen3.5:122b`、`kimi-k2.6`、`deepseek-v4-pro`、`glm-5.2` 這些會做 tool calling 的模型都在上面。
- **依 GPU 時間計費，不是 token**。對 Agent 這種「一次任務打十幾輪、context 越滾越長」的場景，計費模型比較好預估——你付的是實際算力時間，不會因為把整份文件塞進 context 就被 token 數懲罰。
- **可以無痛降回本地**。開發時用雲端大模型驗證流程，之後要把某些步驟換成本地小模型省錢，改 model 字串就好。

至於什麼時候**不該**用它：需要資料完全不出境的場景，那就老老實實跑本地模型。Ollama Cloud 的價值在於「本地體驗 + 雲端算力」，不在於資料隔離。

---

## 2. 兩種連線模式，先搞懂差別

這是最多人一開始卡住的地方。Ollama Cloud 有兩種用法，模型名稱寫法不一樣：

### 模式 A：本地 Ollama 代理（local offload）

你本機仍然跑著 Ollama，但模型名稱帶 `-cloud` 或 `:cloud` 後綴。Ollama 偵測到後綴，會自動把這次推論轉發到雲端，再把結果送回來。

```bash
ollama signin                      # 瀏覽器登入 ollama.com
ollama run gpt-oss:120b-cloud      # 直接在 CLI 用雲端模型
```

程式端完全不用改：

```python
from ollama import chat
resp = chat(model='gpt-oss:120b-cloud', messages=[...])   # 走本機 :11434，轉發到雲端
```

**適合**：本機開發、混用本地與雲端模型、CLI 互動。
**限制**：機器上得裝 Ollama 並登入。部署到沒有 Ollama 的容器就不能用。

### 模式 B：直連 ollama.com API（direct API）

把 `https://ollama.com` 當成一台遠端的 Ollama Host，用 API key 做 Bearer 認證。模型名稱**不帶** `-cloud` 後綴。

```python
from ollama import Client
client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
)
resp = client.chat(model='gpt-oss:120b', messages=[...])   # 注意：沒有 -cloud
```

**適合**：正式部署、CI、Serverless、Docker——任何不想裝 Ollama 的環境。
**這篇文章的所有範例都用模式 B**，因為它才是 Agent 真正上線時會用的形態。

> **踩雷提醒**：模式 B 用 `gpt-oss:120b-cloud` 會失敗，模式 A 用 `gpt-oss:120b` 會去找本地模型然後跟你要 `ollama pull`。名稱後綴不是裝飾品。

---

## 3. 環境準備

### 3.1 拿 API Key

1. 到 [ollama.com](https://ollama.com) 註冊 / 登入
2. 進 [ollama.com/settings/keys](https://ollama.com/settings/keys) 建立一把 API key
3. 設進環境變數：

```bash
export OLLAMA_API_KEY="your_api_key_here"
```

建議寫進 `.env`，並確認 `.env` 有進 `.gitignore`。這篇的範例會用 `python-dotenv` 讀取。

### 3.2 安裝套件

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`：

```
ollama>=0.6
pydantic>=2.0
python-dotenv>=1.0
openai>=1.0      # 只有第 10 節的相容層範例會用到
mcp>=2.0         # 只有第 11 節的 MCP 範例會用到
```

### 3.3 驗證連線

```bash
curl https://ollama.com/api/tags -H "Authorization: Bearer $OLLAMA_API_KEY"
```

看到一坨模型 JSON 就代表通了。

### 3.4 選模型

Agent 對模型只有兩個硬需求：**支援 tools**、**推理夠穩**。

**不要照官網的模型頁挑，直接問 API。** 網頁上列的名稱跟 hosted API 實際接受的不一定一致（例如網頁有 `qwen3.5:122b`，API 上其實是 `qwen3.5:397b`，打前者會 404）：

```bash
curl -s https://ollama.com/api/tags -H "Authorization: Bearer $OLLAMA_API_KEY" \
  | python3 -c "import json,sys; [print(m['name']) for m in json.load(sys.stdin)['models']]"
```

我實測時（2026-08）拿到 18 個：

```
deepseek-v4-flash      deepseek-v4-flash:0731   deepseek-v4-pro
gemma4:31b             glm-5.1                  glm-5.2
gpt-oss:120b           gpt-oss:20b              kimi-k2.6
kimi-k2.7-code         kimi-k3                  minimax-m2.7
minimax-m3             mistral-large-3:675b     nemotron-3-nano:30b
nemotron-3-super       nemotron-3-ultra         qwen3.5:397b
```

**但「列出來」不等於「你能用」。** 免費方案下，大部分模型會直接擋：

```
ResponseError: this model requires a subscription, upgrade for access:
https://ollama.com/upgrade
```

實測免費帳號可用的是 `gpt-oss:120b`、`gpt-oss:20b`、`gemma4:31b`。`qwen3.5:397b`、`deepseek-v4-flash`、`glm-5.2` 都要訂閱。所以挑模型的順序是：**先確認方案配額，再談能力**。

| 模型 | 能力 | 適合場景 |
| --- | --- | --- |
| `gpt-oss:120b` | Tools, Thinking | 通用首選，工具呼叫穩定；免費可用 |
| `gpt-oss:20b` | Tools, Thinking | 簡單分類、路由，降本用；免費可用 |
| `gemma4:31b` | Vision, Tools, Thinking | 需要看圖時；免費可用 |
| `kimi-k2.6` | Vision, Tools, Thinking | 長 context、複雜多步驟規劃；需訂閱 |
| `qwen3.5:397b` | Vision, Tools, Thinking | 中文長文本；需訂閱 |
| `glm-5.2` | Tools, Thinking | 中文任務；需訂閱 |

本文統一用 `gpt-oss:120b`（全部範例都是用它實跑驗證的）。想換就改 `MODEL` 常數（或設 `OLLAMA_MODEL` 環境變數），其他程式一行都不用動。

### 3.5 範例檔案一覽

| 檔案 | 對應章節 | 內容 |
| --- | --- | --- |
| `examples/_client.py` | — | 共用的 client 設定，其他範例都從這裡取 |
| `examples/01_hello_cloud.py` | 第 4 節 | 第一次呼叫（串流） |
| `examples/02_tool_calling.py` | 第 5 節 | 最小 tool calling |
| `examples/03_agent_loop.py` | 第 6 節 | Agent Loop 骨架 |
| `examples/04_codebase_agent.py` | 第 7 節 | 完整的 Codebase Agent |
| `examples/05_structured_output.py` | 第 8 節 | 結構化輸出（兩階段模式） |
| `examples/06_streaming_agent.py` | 第 9 節 | 串流版 Agent Loop |
| `examples/07_openai_compat.py` | 第 10 節 | OpenAI SDK 相容層 |
| `examples/mcp_server_demo.py` | 第 11 節 | 示範用的 MCP Server（工單系統） |
| `examples/08_mcp_agent.py` | 第 11 節 | MCP ↔ Ollama 橋接 + Agent |

文章裡的程式碼片段為了自我完整會重複寫 client 設定，實際檔案則統一 `from _client import MODEL, get_client`。直接 `python examples/xx.py` 執行即可，Python 會自動把 `examples/` 加進 import 路徑。

---

## 4. Hello Cloud：第一次呼叫

`examples/01_hello_cloud.py`：

```python
import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
)

messages = [{'role': 'user', 'content': '用三句話解釋什麼是 AI Agent。'}]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
    print(part['message']['content'], end='', flush=True)
print()
```

```bash
python examples/01_hello_cloud.py
```

`Client` 是整篇文章唯一需要記住的設定點。之後所有範例都是拿同一個 client 去做更複雜的事。

---

## 5. Agent 的心臟：Tool Calling

Agent 跟一般聊天機器人的差別只有一句話：**Agent 能對外部世界做事**。而「做事」的介面就是 tool calling。

流程是四步：

1. 你把可用工具的描述交給模型
2. 模型判斷這輪該不該呼叫工具，要的話回一個 `tool_calls`
3. **你的程式**實際執行那個函式（模型不會也不能自己執行）
4. 把執行結果以 `role: 'tool'` 塞回對話，再問模型一次

Ollama 的 Python SDK 有個很省事的設計：**直接把 Python 函式丟進 `tools=[]`，它會自動從型別標註和 docstring 生成 schema。**

`examples/02_tool_calling.py`：

```python
import os
from dotenv import load_dotenv
from ollama import Client

load_dotenv()
client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
)
MODEL = 'gpt-oss:120b'


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
```

docstring 不是寫給人看的，**它就是模型看到的工具說明書**。工具描述寫得爛，模型就會挑錯工具或帶錯參數——這是 Agent 除錯時第一個該檢查的地方，比調 prompt 有效得多。

---

## 6. 手刻 Agent Loop

上一節是寫死的兩輪。真正的 Agent 要能自己決定「還要不要再做一步」，所以核心是一個迴圈：

```
while True:
    呼叫模型
    有 tool_calls？ → 執行、把結果塞回去、繼續迴圈
    沒有？          → 模型認為任務完成了，跳出
```

`examples/03_agent_loop.py`：

```python
import os
from dotenv import load_dotenv
from ollama import Client, ChatResponse

load_dotenv()
client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
)
MODEL = 'gpt-oss:120b'


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
```

**這 40 行就是 Agent 的全部本質。** 市面上的 Agent 框架，剝掉抽象層之後核心也是這個迴圈。

`think=True` 會讓支援 thinking 的模型把推理過程放進 `message.thinking`，跟 `message.content` 分開。除錯時非常有用——你可以直接看到模型為什麼選了那個工具。

> **實測踩到的坑**：一開始我沒寫那句系統提示，結果 `gpt-oss:120b` 直接心算出 9,795,300，一次工具都沒呼叫，迴圈第一輪就結束。這不是 bug——**能力強的模型會跳過它覺得不必要的工具**。如果某個步驟你「一定」要它走工具（為了正確性、為了留稽核紀錄），就得在系統提示裡明講。

三個立刻要補的防護（下一節的實戰版本會加上）：

- **迴圈上限**：模型有可能鬼打牆，一定要設 `max_turns`。
- **工具例外處理**：工具丟例外時不要讓程式炸掉，把錯誤訊息當成工具結果回給模型，它通常會自己換個做法重試。
- **未知工具**：模型偶爾會幻覺出不存在的工具名，要擋。

---

## 7. 實戰：一個會讀專案的 Codebase Agent

現在做一個真的有用的東西：**丟給它一個問題，它自己去翻專案目錄、讀檔、搜尋，然後回答你**。

工具設計三個：`list_files`、`read_file`、`search_code`。全部限制在指定的根目錄底下，避免 Agent 亂跑到 `/etc`。

完整程式在 `examples/04_codebase_agent.py`，這裡看關鍵片段。

### 7.1 工具定義（含安全邊界）

```python
ROOT = Path(os.environ.get('AGENT_ROOT', '.')).resolve()
MAX_FILE_BYTES = 40_000


def _safe_path(relative_path: str) -> Path:
    """把相對路徑解析成絕對路徑，並確保沒有跳出 ROOT。"""
    target = (ROOT / relative_path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f'路徑超出允許範圍：{relative_path}')
    return target


def list_files(relative_path: str = '.') -> str:
    """列出專案中某個目錄底下的檔案與子目錄。

    Args:
        relative_path: 相對於專案根目錄的路徑，預設為根目錄本身

    Returns:
        每行一個項目，目錄結尾帶 /
    """
    target = _safe_path(relative_path)
    if not target.is_dir():
        return f'{relative_path} 不是目錄'
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith('.'):
            continue
        entries.append(f'{p.name}/' if p.is_dir() else p.name)
    return '\n'.join(entries) or '(空目錄)'


def read_file(relative_path: str) -> str:
    """讀取專案中某個檔案的完整內容。

    Args:
        relative_path: 相對於專案根目錄的檔案路徑

    Returns:
        檔案內容，過長時會被截斷
    """
    target = _safe_path(relative_path)
    if not target.is_file():
        return f'{relative_path} 不存在或不是檔案'
    data = target.read_text(encoding='utf-8', errors='replace')
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES] + '\n... (內容過長已截斷)'
    return data


def search_code(pattern: str, relative_path: str = '.') -> str:
    """在專案中以純文字關鍵字搜尋，回傳符合的檔案與行號。

    Args:
        pattern: 要搜尋的關鍵字（不分大小寫）
        relative_path: 搜尋範圍，預設整個專案

    Returns:
        每行格式為 檔案路徑:行號: 該行內容，最多 50 筆
    """
    root = _safe_path(relative_path)
    needle = pattern.lower()
    hits = []
    for p in root.rglob('*'):
        if not p.is_file() or any(part.startswith('.') for part in p.parts):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                if needle in line.lower():
                    hits.append(f'{p.relative_to(ROOT)}:{i}: {line.strip()[:160]}')
                    if len(hits) >= 50:
                        return '\n'.join(hits) + '\n... (結果過多已截斷)'
        except (UnicodeDecodeError, PermissionError):
            continue
    return '\n'.join(hits) or f'找不到符合 "{pattern}" 的內容'
```

幾個刻意的設計：

- **每個工具都回傳字串**，而且失敗時回「人話錯誤訊息」而不是丟例外。模型讀得懂 `找不到符合 "xxx" 的內容`，然後會自己換關鍵字再試一次。
- **輸出一定要截斷**。沒有上限的話，Agent 讀到一個 5MB 的 log 就會把 context 撐爆。
- **`_safe_path` 是必要的**，不是加分項。模型帶 `../../etc/passwd` 進來不是理論風險，是實際會發生的事。

### 7.2 Agent 主體

```python
SYSTEM_PROMPT = """你是一個程式碼庫分析助手。

工作方式：
- 先用 list_files 建立對專案結構的認識，再決定要讀哪些檔案
- 需要找特定符號或設定時，用 search_code 比逐檔讀取有效率
- 只根據實際讀到的檔案內容回答，不要臆測沒看過的程式碼
- 蒐集到足夠資訊後，直接給出結論，不要再繼續呼叫工具

回答時請用繁體中文，並在提及程式碼時附上 檔案路徑:行號。"""


def run_agent(question: str, max_turns: int = 12) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = client.chat(
            model=MODEL, messages=messages, tools=TOOLS, think=True,
        )
        messages.append(response.message)

        if response.message.thinking:
            print(f'\n\033[90m[第 {turn} 輪思考] '
                  f'{response.message.thinking.strip()[:300]}\033[0m')

        if not response.message.tool_calls:
            return response.message.content

        for tc in response.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments
            print(f'\033[36m[工具] {name}({args})\033[0m')
            fn = AVAILABLE.get(name)
            if fn is None:
                result = f'錯誤：沒有名為 {name} 的工具，可用的有 {list(AVAILABLE)}'
            else:
                try:
                    result = fn(**args)
                except Exception as exc:                  # noqa: BLE001
                    result = f'工具執行失敗：{type(exc).__name__}: {exc}'
            preview = result.replace("\n", " ")[:120]
            print(f'\033[90m  → {preview}\033[0m')
            messages.append({
                'role': 'tool', 'tool_name': name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'
```

跑跑看：

```bash
export AGENT_ROOT=/path/to/your/project
python examples/04_codebase_agent.py "這個專案的進入點在哪裡？主要模組怎麼切的？"
```

你會看到它自己 `list_files` → 挑幾個可疑檔案 `read_file` → 發現線索後 `search_code` → 最後給出帶行號的回答。**整個規劃過程沒有一行是你寫死的**，是模型自己排的。

### 7.3 這裡面藏著的幾個工程重點

**系統提示要寫「停止條件」。** `蒐集到足夠資訊後，直接給出結論` 這句很關鍵。沒有它，模型常常會過度探索，一路讀到 max_turns 才停。

**工具粒度要抓對。** 太細（`open_file` / `read_line` / `close_file`）會讓輪數暴增，每一輪都是一次雲端呼叫；太粗（`analyze_whole_project`）等於你自己把邏輯寫死了，Agent 沒有發揮空間。一個好的判準是：**每個工具對應人類會做的一個動作**。

**Context 是會累積的。** 每輪的 tool 結果都留在 `messages` 裡。長任務跑到後面 context 會很可觀，這也是為什麼工具輸出的截斷上限那麼重要。

---

## 8. 結構化輸出：讓 Agent 吐出可以直接用的資料

Agent 常常是更大流程裡的一環，下游需要的是 JSON 不是散文。

### 8.1 先講一個雲端的坑：`format` 不會生效

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

### 8.2 方案 A：借 tool calling 來做 schema 約束（推薦）

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

### 8.3 方案 B：prompt + 去圍籬 + 驗證（備援）

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

### 8.4 跟 Agent Loop 怎麼搭

**不要在 Agent Loop 裡同時掛工具跟 `submit`**，模型會搞不清楚該繼續查資料還是該交卷。分兩階段：

1. **蒐集階段**：Agent Loop 帶著真正的工具跑（第 7 節那套），直到模型不再要求工具
2. **收斂階段**：把蒐集到的結論當成 prompt，單獨呼叫一次 `structured_via_tool()` 交出 JSON

多花一次呼叫，換來下游拿得到乾淨的資料，划算。

---

## 9. 串流與 Thinking：把思考過程秀出來

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

## 10. 用 OpenAI SDK 相容層接既有生態系

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

## 11. 接上 MCP：不用自己寫工具

第 7 節那三個工具是我們自己刻的。但如果 Agent 要能查 GitHub、讀 Slack、打資料庫、操作瀏覽器呢？一個一個手寫，寫到天荒地老。

**MCP（Model Context Protocol）解決的就是這件事**：它把「工具」標準化成一個協定，任何人寫的 MCP Server 都能被任何 MCP Client 使用。現在 GitHub、Notion、Playwright、PostgreSQL、檔案系統……都有現成的 Server。接上去，你的 Agent 立刻多幾十個工具。

### 11.1 先講清楚：Ollama 沒有內建 MCP

這點要先說，因為很多文章講得很含糊。**截至目前，Ollama 本身不支援 MCP**（官方 repo 的 [issue #7865](https://github.com/ollama/ollama/issues/7865) 還開著）。你不能像設定 Claude Desktop 那樣丟一個 `mcp.json` 給 Ollama 就完事。

但這其實沒什麼大不了，因為橋接的工作量小到有點好笑：

| | MCP | Ollama |
| --- | --- | --- |
| 工具清單 | `client.list_tools()` → `tool.name` / `tool.description` / `tool.input_schema` | `tools=[{'type':'function','function':{'name','description','parameters'}}]` |
| 執行工具 | `await client.call_tool(name, args)` | 你自己執行，結果以 `role: 'tool'` 塞回去 |

兩邊的參數描述**都是 JSON Schema**。所以橋接層的本質只是搬欄位——大概 30 行。

架構長這樣：

```
                 ┌──────────────────────────────┐
  你的 Agent ───→│  MCPToolBridge（30 行）       │
   (Ollama)      │  list_tools() → tools schema │
       ↑         │  tool_calls   → call_tool()  │
       └─────────│                              │
   tool 結果      └───────┬──────────────────────┘
                         │ MCP 協定 (stdio / HTTP)
                 ┌───────┴───────┬──────────────┐
              工單系統        GitHub        檔案系統
             (自己寫的)      (官方 Server)   (官方 Server)
```

### 11.2 準備一台 MCP Server

為了讓範例能獨立跑，先自己寫一台。用 MCP Python SDK 寫 Server 跟寫普通函式差不多——`@mcp.tool()` 會從型別標註和 docstring 自動生出 schema，跟第 5 節 Ollama SDK 的做法是同一個思路。

`examples/mcp_server_demo.py`（節錄）：

```python
from mcp.server import MCPServer

mcp = MCPServer('TicketSystem')

TICKETS = {
    'T-101': {'title': '結帳頁在 Safari 會卡住', 'status': 'open',
              'priority': 'high', 'assignee': 'alice',
              'comments': ['使用者回報只有 iOS 18 會發生']},
    # ...
}


@mcp.tool()
def list_tickets(status: str = 'all') -> str:
    """列出工單，可依狀態篩選。

    Args:
        status: 篩選條件，可填 open、closed 或 all（預設）
    """
    rows = []
    for tid, t in TICKETS.items():
        if status != 'all' and t['status'] != status:
            continue
        rows.append(f"{tid} [{t['status']}/{t['priority']}] {t['title']}")
    return '\n'.join(rows) or f'沒有狀態為 {status} 的工單'


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """取得單一工單的完整內容，包含所有留言。

    Args:
        ticket_id: 工單編號，例如 T-101
    """
    t = TICKETS.get(ticket_id)
    if t is None:
        raise ValueError(f'找不到工單 {ticket_id}，現有的有 {list(TICKETS)}')
    return f"編號：{ticket_id}\n標題：{t['title']}\n狀態：{t['status']}\n..."


if __name__ == '__main__':
    mcp.run()   # 預設走 stdio，由呼叫端以子行程啟動
```

注意 `get_ticket` 裡那個 `raise ValueError`。**MCP 的錯誤處理很聰明：Server 端丟例外不會讓 Client 也炸掉**，而是變成一個 `is_error=True` 的正常回應。這正是我們要的——模型讀得到錯誤訊息，就能自己修正參數重試。

### 11.3 橋接層

核心就這個類別（完整版在 `examples/08_mcp_agent.py`）：

```python
from contextlib import AsyncExitStack
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolBridge:
    """連上一或多台 MCP Server，把它們的工具攤平成 Ollama 的 tools 陣列。

    工具名稱會加上 server 前綴（例如 tickets__get_ticket），避免多台
    server 提供同名工具時撞在一起。
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._routes: dict[str, tuple[Client, str]] = {}
        self.tools: list[dict] = []

    async def connect(self, server_name: str, transport) -> list[str]:
        client = await self._stack.enter_async_context(Client(transport))
        listing = await client.list_tools()

        for tool in listing.tools:
            qualified = f'{server_name}__{tool.name}'
            self._routes[qualified] = (client, tool.name)
            self.tools.append({
                'type': 'function',
                'function': {
                    'name': qualified,
                    'description': tool.description or '',
                    'parameters': tool.input_schema,   # MCP 給的就是 JSON Schema
                },
            })
        return [t.name for t in listing.tools]

    async def call(self, qualified_name: str, arguments: dict) -> str:
        route = self._routes.get(qualified_name)
        if route is None:
            return f'錯誤：沒有名為 {qualified_name} 的工具，可用的有 {list(self._routes)}'

        client, mcp_name = route
        result = await client.call_tool(mcp_name, arguments)

        # 工具內部丟例外不會在這裡 raise，而是回傳 is_error=True。
        # 把錯誤訊息原樣交給模型，它通常會自己換個做法重試。
        text = self._blocks_to_text(result)
        return f'工具執行失敗：{text}' if result.is_error else text

    @staticmethod
    def _blocks_to_text(result) -> str:
        """CallToolResult.content 是一串 block，這裡只取得出文字的部分。"""
        parts = []
        for block in result.content:
            text = getattr(block, 'text', None)
            parts.append(text if text is not None
                         else f'[{getattr(block, "type", "unknown")} 類型的內容，此 Agent 未處理]')
        if not parts and getattr(result, 'structured_content', None) is not None:
            return str(result.structured_content)
        return '\n'.join(parts) or '(工具沒有回傳內容)'

    async def aclose(self) -> None:
        await self._stack.aclose()
```

三個設計重點：

- **`AsyncExitStack`**：MCP 連線是 async context manager，每台 Server 的生命週期都得管。用 `AsyncExitStack` 就能連任意多台，最後一次 `aclose()` 全部收乾淨。
- **工具名稱加前綴**（`tickets__get_ticket`）：接兩台以上 Server 時，同名工具幾乎一定會撞——`search`、`list`、`get` 這種名字太常見了。前綴是必須的，不是潔癖。
- **`content` 是 block 陣列**：MCP 的回傳可能是文字、圖片、embedded resource。這裡只處理文字，其他型別回一個佔位訊息，至少模型知道「有東西但我看不到」，而不是靜默吃掉。

### 11.4 Agent 主體

因為 MCP SDK 是 async 的，Agent Loop 也要換成 `AsyncClient`。除此之外，結構跟第 7 節一模一樣：

```python
from ollama import AsyncClient

async def run_agent(bridge: MCPToolBridge, question: str, max_turns: int = 12) -> str:
    client = AsyncClient(
        host='https://ollama.com',
        headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = await client.chat(
            model=MODEL, messages=messages, tools=bridge.tools, think=True,
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        # 同一輪的多個工具呼叫可以平行送出，省下往返時間
        calls = list(response.message.tool_calls)
        results = await asyncio.gather(*(
            bridge.call(tc.function.name, tc.function.arguments) for tc in calls
        ))

        for tc, result in zip(calls, results):
            messages.append({
                'role': 'tool', 'tool_name': tc.function.name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'
```

**`asyncio.gather` 是換成 async 之後白賺的好處。** 模型一輪要求查三張工單時，同步版本會一張一張跑，async 版本一次打完。工具是 I/O bound（HTTP、資料庫、子行程）時差距很明顯。

接上 Server 並啟動：

```python
bridge = MCPToolBridge()
try:
    names = await bridge.connect(
        'tickets',
        stdio_client(StdioServerParameters(
            command=sys.executable, args=['examples/mcp_server_demo.py'],
        )),
    )
    print(f'已連上 MCP server「tickets」，工具：{names}')
    answer = await run_agent(bridge, question)
    print(answer)
finally:
    await bridge.aclose()
```

跑跑看：

```bash
pip install mcp
python examples/08_mcp_agent.py "有哪些還沒關掉的工單？最急的那張在講什麼？"
```

輸出大致是：

```
已連上 MCP server「tickets」，工具：['list_tickets', 'get_ticket', 'add_comment']
[MCP] tickets__list_tickets({'status': 'open'})
  → T-101 [open/high] 結帳頁在 Safari 會卡住 — 負責人 alice ...
[MCP] tickets__get_ticket({'ticket_id': 'T-101'})
  → 編號：T-101 標題：結帳頁在 Safari 會卡住 ...
============================================================
目前有兩張未關閉的工單...
```

**注意這裡沒有任何一行是我們寫的工具邏輯。** Agent 的工具全部來自 MCP Server，而那台 Server 可以是別人寫的、跑在別台機器上的。

### 11.5 接現成的 MCP Server

自己寫的能接，別人的當然也能。`connect()` 的第二個參數換掉就好。

**本地 stdio server**（官方的檔案系統 Server，用 npx 跑）：

```python
await bridge.connect('fs', stdio_client(StdioServerParameters(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-filesystem', '/path/to/project'],
)))
```

**需要金鑰的 server**：透過 `env` 傳，不要寫進 args：

```python
await bridge.connect('github', stdio_client(StdioServerParameters(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-github'],
    env={'GITHUB_PERSONAL_ACCESS_TOKEN': os.environ['GITHUB_TOKEN']},
)))
```

**遠端 HTTP server**：`Client` 會從字串型別自動判斷用 streamable HTTP transport，直接給 URL 就行：

```python
await bridge.connect('notion', 'https://your-mcp-host/mcp')
```

要帶認證 header 的話，就自己組 transport：

```python
import httpx2
from mcp.client.streamable_http import streamable_http_client

async with httpx2.AsyncClient(
    headers={'Authorization': f'Bearer {token}'},
    timeout=httpx2.Timeout(30.0, read=300.0),
) as http_client:
    transport = streamable_http_client('https://your-mcp-host/mcp', http_client=http_client)
    await bridge.connect('notion', transport)
```

### 11.6 接 MCP 之後才會遇到的坑

**工具數量會爆炸，而且要花錢。** 一台 GitHub Server 可能就給你 30 個工具，接三台就快 100 個。這些工具的完整 schema **每一輪**都要送給模型，是實打實的 context 成本，而且太多選項會讓模型挑錯工具。實務做法是**篩選**——只註冊這個 Agent 真正需要的：

```python
ALLOWED = {'tickets': {'list_tickets', 'get_ticket'}}   # 唯讀，不給 add_comment

for tool in listing.tools:
    if tool.name not in ALLOWED.get(server_name, {tool.name}):
        continue
    # ... 註冊
```

**寫入型工具要有關卡。** MCP Server 提供什麼工具是它決定的，裡面很可能有 `delete_file`、`close_issue`、`send_message`。在 `bridge.call()` 加一道確認：

```python
DESTRUCTIVE = {'fs__write_file', 'github__create_issue', 'tickets__add_comment'}

async def call(self, qualified_name, arguments):
    if qualified_name in DESTRUCTIVE:
        answer = input(f'\n⚠ Agent 要執行 {qualified_name}({arguments})，允許嗎？[y/N] ')
        if answer.strip().lower() != 'y':
            return '使用者拒絕了這次操作，請改用其他方式或直接回報無法完成。'
    ...
```

被拒絕時**要回一句話給模型**，不要回空字串。模型收到「使用者拒絕了」會轉而說明狀況；收到空字串則常常會傻傻地再試一次。

**第三方 Server 的描述文字是不可信輸入。** 工具的 `description` 會原封不動進入你的 prompt。如果那台 Server 不是你的，等於讓外部來源往你的 context 裡塞字——這是 prompt injection 的標準入口。接不熟的 Server 前先自己 `list_tools()` 看一遍它寫了什麼。

**stdio server 是子行程。** 它會繼承你的環境變數、以你的權限跑。`npx -y` 那行是在下載並執行別人的程式碼，該有的警覺不能少。

---

## 12. 正式上線前要處理的事

前面的程式碼是為了讀起來清楚。真的要上線，這幾件事跑不掉：

### 重試與逾時

網路會抖，雲端服務會偶發 5xx。至少包一層指數退避：

```python
import time
from ollama import ResponseError

def chat_with_retry(client, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat(**kwargs)
        except ResponseError as exc:
            if exc.status_code and exc.status_code < 500:
                raise                       # 4xx 是你的問題，重試沒用
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

注意 4xx 不要重試——認證錯誤、模型名稱打錯、參數格式不對，重試一百次也一樣。

### 成本控制

Ollama Cloud 依 GPU 時間計費，所以省錢的方向跟 token 計費不太一樣：

- **減少輪數比減少 context 有效**。每一輪都是一次完整的請求往返，工具設計得好、一輪能拿到足夠資訊，比壓縮 prompt 有用得多。
- **`max_turns` 一定要設**。失控的 Agent Loop 是帳單殺手。
- **分層用模型**。路由、分類這種簡單判斷丟 `qwen3.5:9b` 或 `deepseek-v4-flash`，只有主要規劃用大模型。
- **免費額度會遇到 rate limit**，正式跑要看一下 [ollama.com/pricing](https://ollama.com/pricing) 的方案配額。

### 可觀測性

至少把每輪的 `tool_calls` 和 `thinking` 記下來。Agent 出錯時你要回答的問題是「它為什麼決定做這件事」，而那個答案只在 thinking 裡。

### 安全邊界

- API key 用環境變數或 secret manager，不要進版控
- **能寫入的工具（刪檔、發 API、送信）一律加人工確認關卡**，別讓模型直接觸發不可逆操作
- 路徑、SQL、shell 指令這類參數，一律當成不可信輸入來驗證——模型的輸出本質上就是使用者可以間接控制的內容

### 資料落地

雲端模式下對話內容會送到 Ollama 的伺服器。有合規要求的資料，該過濾就過濾，或改跑本地模型。

---

## 13. 結語

回頭看，整篇文章的核心其實只有第 6 節那個迴圈：

> **呼叫模型 → 有工具就執行 → 結果餵回去 → 重複，直到模型說完成。**

Agent 不是什麼魔法，是「一個會用工具的 while 迴圈」。難的從來不是這個迴圈，而是：

- 工具的**粒度**跟**描述**寫得夠不夠好（這決定 Agent 聰不聰明，比換模型有效）
- 停止條件跟輪數上限（這決定它會不會失控）
- 工具輸出的截斷跟錯誤處理（這決定它撐不撐得住真實資料）

第 11 節的 MCP 也沒有改變這個結構——它只是把「工具從哪來」外包出去。迴圈還是那個迴圈，只是 `tools` 陣列不再是你手寫的函式，而是從別台 Server 撈回來的。這也是為什麼建議先手刻過第 6 節那 40 行再接 MCP：知道底下在做什麼，之後任何框架你都能一眼看穿。

Ollama Cloud 在這條路上的貢獻，是把「模型能力」這個變數從等式裡拿掉了——你不用再為了跑得動而遷就一個工具呼叫時好時壞的小模型，也不用為了跑大模型去租 GPU。同一套 API，本地跟雲端隨你切。

### 下一步可以玩的

- **把 MCP 接到底**：第 11 節只接了一台自製 Server，實務上會同時掛 GitHub、檔案系統、資料庫，再加上工具篩選與人工確認關卡
- **多 Agent 分工**：一個 planner 負責拆任務，多個 worker 平行執行，最後 synthesizer 收斂
- **記憶層**：把跨 session 的結論存進向量庫，讓 Agent 記得上次的發現
- **降本實驗**：把流程中的簡單步驟逐一換成小模型，量測品質掉多少

---

## 參考資料

- [Ollama Cloud 官方文件](https://docs.ollama.com/cloud)
- [Tool Calling 文件](https://docs.ollama.com/capabilities/tool-calling)
- [Structured Outputs 文件](https://docs.ollama.com/capabilities/structured-outputs)
- [OpenAI 相容性文件](https://docs.ollama.com/api/openai-compatibility)
- [Cloud models 公告](https://ollama.com/blog/cloud-models)
- [雲端模型清單](https://ollama.com/search?c=cloud)
- [定價](https://ollama.com/pricing)
- [MCP 官方網站](https://modelcontextprotocol.io/)
- [MCP Python SDK 文件](https://py.sdk.modelcontextprotocol.io/)
- [MCP Client 開發教學](https://modelcontextprotocol.io/docs/develop/build-client)
- [Ollama MCP 支援討論串（issue #7865）](https://github.com/ollama/ollama/issues/7865)
