# 用 Ollama Cloud 從零打造一個 AI Agent

> 一篇可以照著跑完的實作文。從註冊、第一次呼叫，一路做到一個會自己決定用哪個工具、能連續多輪推理的 Agent。
> 全部程式碼都在 `examples/` 底下，複製貼上就能執行。

---

## 這篇是給誰看的

假設你會寫 Python，但沒做過 AI Agent。你不需要懂機器學習，也不需要有 GPU。

讀完你會有一個能自己決定用哪個工具、能連續多輪推理、能讀你自己文件的 Agent——而且每一行都是你看得懂的程式碼，不是框架的黑盒子。

**文章裡的每個數字都是實測的**，包含那些「照官方文件寫會失敗」的地方。踩過的坑我都留在原地並標明，因為那些通常比成功案例更有用。

---

## 三條學習路徑

**不要從頭讀到尾**，照你的目的挑一條：

### 🚀 路徑 A：三十分鐘搞懂 Agent 是什麼

> 本頁第 1 → 4 → 5 → 6 → 7 節

跑完你會有一個會用工具的 Agent，並且理解它的本質只是一個 while 迴圈。**大部分人到這裡就夠用了。**

```bash
python examples/00_check_setup.py     # 先確認環境
python examples/01_hello_cloud.py
python examples/02_tool_calling.py
python examples/03_agent_loop.py      # ← 全文最核心的 40 行
```

### 🛠 路徑 B：做一個真的能用的東西

> 路徑 A ＋ 本頁第 8 節 ＋ [結構化輸出](docs/01-structured-output.md) ＋ [成本控制](docs/05-cost.md)

加上真實 Agent 需要的防護（路徑邊界、輸出截斷、例外處理）、下游能用的結構化輸出，以及不燒錢的模型選擇。

```bash
python examples/04_codebase_agent.py "這個專案的進入點在哪裡？"
python examples/05_structured_output.py
python examples/11_model_router.py
```

### 📚 路徑 C：完整讀完

> 本頁 8 節 ＋ [docs/](docs/README.md) 全部 8 篇

MCP 工具生態、RAG 與自動化評估、框架整合、可觀測性、上線準備。
docs/ 各篇彼此獨立，建議當作**參考手冊**——遇到問題再翻對應那篇。

---

## 這個 repo 的結構

```
README.md              ← 你在這裡。核心路徑，從零到一個能用的 Agent
docs/                  ← 進階主題，各篇獨立
examples/              ← 全部可執行，數字就是閱讀順序
  00_check_setup.py    ← 第一支該跑的
  01–03                核心：呼叫 → 工具 → Agent Loop
  04–07                進階：實戰、結構化輸出、串流、相容層
  08、11–14            專題：MCP、成本、RAG、評估
  09–10                框架與可觀測性
  15–18                記憶管理、測試、長期記憶、部署
Dockerfile             ← 209MB，裡面沒有 Ollama（見部署那篇）
check_docs.py          ← 文件一致性檢查（改文件後跑）
```

---

## 目錄

**基礎——先讀這些**

| 節 | 標題 | 難度 |
| --- | --- | --- |
| 1 | [為什麼是 Ollama Cloud](#1-為什麼是-ollama-cloud) | 入門 |
| 2 | [兩種連線模式，先搞懂差別](#2-兩種連線模式先搞懂差別) | 入門 |
| 3 | [技術棧與工具鏈全景](#3-技術棧與工具鏈全景) | 入門 |
| 4 | [環境準備](#4-環境準備) | 入門 |

**核心——Agent 的本質**

| 節 | 標題 | 難度 |
| --- | --- | --- |
| 5 | [Hello Cloud：第一次呼叫](#5-hello-cloud第一次呼叫) | 入門 |
| 6 | [Agent 的心臟：Tool Calling](#6-agent-的心臟tool-calling) | 入門 |
| 7 | [手刻 Agent Loop](#7-手刻-agent-loop) | ⭐ **最重要的一節** |
| 8 | [實戰：一個會讀專案的 Codebase Agent](#8-實戰一個會讀專案的-codebase-agent) | 中階 |

**進階——需要哪塊看哪塊**（獨立文件，見 [docs/](docs/README.md)）

| 主題 | 難度 |
| --- | --- |
| [結構化輸出](docs/01-structured-output.md) | 中階 |
| [串流與 Thinking](docs/02-streaming.md) | 中階 |
| [OpenAI SDK 相容層](docs/03-openai-compat.md) | 中階 |
| [接上 MCP](docs/04-mcp.md) | 進階 |
| [GPU 時間計費下的成本控制](docs/05-cost.md) | 中階 |
| [RAG 與自動化評估](docs/06-rag.md) | 進階 |
| [正式上線前要處理的事](docs/07-production.md) | 中階 |
| [框架與可觀測性](docs/08-frameworks-observability.md) | 進階 |
| [多輪對話與 Context 管理](docs/09-memory-context.md) | 中階 |
| [測試 Agent](docs/10-testing.md) | 中階 |
| [跨 Session 長期記憶](docs/11-persistent-memory.md) | 進階 |
| [部署上線](docs/12-deployment.md) | 中階 |
| [硬體選型](docs/13-hardware.md) | 中階 |


---

## 1. 為什麼是 Ollama Cloud

寫 Agent 最痛的一件事，是**模型能力**跟**部署成本**互相拉扯。

Agent 需要模型會穩定地做工具呼叫（tool calling）、能規劃多步驟、輸出格式不會亂跑。這種能力通常只有中大型模型才做得好。但一個 120B 的模型要在自己機器上跑，光 VRAM 就先勸退大部分人。

Ollama Cloud 做的事很直白：**維持 Ollama 原本的 API 與開發體驗，把模型推論丟到雲端 GPU 上跑。**

實際帶來的好處：

- **本地開發不用改架構**。原本用 `ollama.chat()` 寫的程式，換個 model 名稱就跑在雲上了。
- **模型選擇一次到位**。`gpt-oss:120b`、`qwen3.5:397b`、`kimi-k2.6`、`deepseek-v4-pro`、`glm-5.2` 這些會做 tool calling 的模型都在上面。
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

## 3. 技術棧與工具鏈全景

在裝任何東西之前，先看清楚整張圖。這節不寫程式碼，是給你一個「哪個零件負責什麼、什麼時候才需要它」的地圖——後面每一節都會回到這張圖上的某一格。

### 3.1 一張圖看完

```
┌─────────────────────────────────────────────────────────────┐
│  你的 Agent 程式（Python）                                    │
│                                                              │
│   while True:  呼叫模型 → 有工具就執行 → 結果餵回 → 重複        │  ← [〈手刻 Agent Loop〉](#7-手刻-agent-loop)
└───┬──────────────┬──────────────┬──────────────┬────────────┘
    │              │              │              │
    │ 推論          │ 工具          │ 知識          │ 觀測
    ↓              ↓              ↓              ↓
┌─────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ 模型層   │  │ 工具層     │  │ 檢索層     │  │ 可觀測層   │
├─────────┤  ├───────────┤  ├───────────┤  ├───────────┤
│ Ollama  │  │ 自己寫的   │  │ 切塊       │  │ Langfuse  │
│  Cloud  │  │  函式      │  │ BM25      │  │ LangSmith │
│         │  │           │  │ embedding │  │           │
│ 本地     │  │ MCP       │  │ 向量比對   │  │ 結構化 log │
│  Ollama │  │  Server   │  │ 重排       │  │           │
└─────────┘  └───────────┘  └───────────┘  └───────────┘
  [〈兩種連線模式，先搞懂差別〉](#2-兩種連線模式先搞懂差別)       第 6/12 節      [〈RAG〉](docs/06-rag.md)        [〈接框架與可觀測性〉](docs/08-frameworks-observability.md)
```

四層裡面，**只有模型層和工具層是必要的**。檢索層要等你需要讓 Agent 讀你自己的文件才用得到，可觀測層要等你上線才會痛。初學者照這個順序點技能樹就好。

### 3.2 逐層說明：每個零件解決什麼問題

#### 模型層

| 零件 | 解決什麼 | 什麼時候需要 |
| --- | --- | --- |
| **Ollama Cloud** | 跑得動 120B 級模型，不用買 GPU | 一開始就需要 |
| **本地 Ollama** | 離線、資料不出境、embedding、微調 | 做 RAG 或有合規要求時 |
| `ollama` Python SDK | 官方 client，函式自動轉 schema | 一開始就需要 |
| `openai` SDK | 接既有以 OpenAI 為介面的程式 | 只有要沿用舊程式時 |

#### 工具層——Agent 的手腳

| 零件 | 解決什麼 | 什麼時候需要 |
| --- | --- | --- |
| **自己寫的 Python 函式** | 最直接。SDK 會自動從型別標註與 docstring 生 schema | 一開始就需要 |
| **MCP Server** | 不用自己寫。GitHub、檔案系統、資料庫都有現成的 | 需要接外部系統時 |
| `mcp` SDK | MCP 的 client/server 實作 | 同上 |

#### 檢索層——讓 Agent 讀你的文件（RAG）

這層零件最多，也最容易被賣過頭。**照下表由上而下加，能停就停**：

| 零件 | 解決什麼 | 我的建議 |
| --- | --- | --- |
| **切塊（chunking）** | 文件太長塞不進 context | **必要**。依 Markdown 標題切，別用固定字元數 |
| **BM25 關鍵字檢索** | 找出該讀哪一段 | **先試這個**。手寫六十行，零依賴，實測 Recall@1 85% |
| **Embedding 模型** | 讓「意思相近」也找得到，不只是「字一樣」 | 使用者會用自己的話問時才需要。**雲端沒有，要跑本地** |
| **向量資料庫** | 語料大到用 Python list 掃不動 | 幾千段以內不需要。先用 list |
| **重排模型（reranker）** | 檢索出十段，挑出最相關的三段 | 檢索夠準就不用。這是最後才加的 |
| **評估資料集** | 回答「我的 RAG 到底準不準」 | **比向量資料庫更該先做**。見 [RAG 篇的自動化評估](docs/06-rag.md#9-怎麼自動驗證-rag-的正確率) |

> 這張表的順序是有意的。很多 RAG 教學一開始就叫你裝向量資料庫，但**評估**才是你最早該做的事——沒有評估，你連換了 embedding 模型有沒有變好都不知道。

#### 可觀測層——Agent 出錯時你要能回答「它為什麼這樣做」

| 零件 | 解決什麼 | 什麼時候需要 |
| --- | --- | --- |
| **結構化 log** | 把 `thinking` 與 `tool_calls` 記下來 | **最低限度，一開始就該做** |
| **Langfuse** | 樹狀 trace、token 用量、成本歸因。開源可自架 | 開始有多輪 Agent 就值得 |
| **LangSmith** | 同上，但閉源 SaaS、繞著 LangChain 設計 | 整套都在 LangChain 上時 |

#### 框架層（選用）

| 零件 | 解決什麼 | 我的建議 |
| --- | --- | --- |
| **LangChain / LangGraph** | checkpoint、中斷續跑、人在迴圈中 | **先手刻過一次再用**。不然框架對你是黑盒子 |

### 3.3 三種常見組合

**組合一：最小可用 Agent**（第 5–8 節）

```
ollama SDK  +  自己寫的函式  +  print 除錯
```

兩個套件，一個檔案。**大部分「幫我自動化某件事」的需求到這裡就夠了。**

**組合二：能讀公司文件的問答 Agent**（[〈RAG〉](docs/06-rag.md)）

```
ollama SDK  +  切塊 + BM25  +  檢索當作工具
             （語料大或使用者用口語提問時，再加本地 embedding）
```

注意這裡**沒有向量資料庫**。六個段落用 Python list 掃就好，加了只是徒增部署複雜度。

**組合三：上線形態**（第 15–16 節）

```
組合一或二  +  重試機制  +  Langfuse trace  +  max_turns 上限
            +  評估資料集（改動前後跑一次，確認沒退步）
```

### 3.4 這個 repo 用到的完整清單

| 套件 | 版本 | 用在哪 | 必要性 |
| --- | --- | --- | --- |
| `ollama` | ≥0.6 | 全部 | 必要 |
| `python-dotenv` | ≥1.0 | 全部 | 必要 |
| `pydantic` | ≥2.0 | [〈結構化輸出〉](docs/01-structured-output.md) 結構化輸出 | 必要 |
| `openai` | ≥1.0 | [〈用 OpenAI SDK 相容層接既有生態系〉](docs/03-openai-compat.md) 相容層 | 選用 |
| `mcp` | ≥2.0 | [〈接上 MCP〉](docs/04-mcp.md) MCP | 選用 |
| `langchain`, `langchain-ollama` | ≥1.0 | [〈接框架與可觀測性〉](docs/08-frameworks-observability.md) | 選用 |
| `langfuse` | ≥4.0 | [〈接框架與可觀測性〉](docs/08-frameworks-observability.md) | 選用 |
| `opentelemetry-proto` | ≥1.0 | 假 Langfuse server 解碼 | 選用 |

**檢索層一個套件都不用裝。** BM25、切塊、餘弦相似度全部是手寫的純 Python，加起來不到一百二十行（`examples/rag_common.py`）。這是刻意的——我想讓你看清楚裡面在做什麼，而不是 `pip install` 一個黑盒子。

---

## 4. 環境準備

### 4.1 拿 API Key

1. 到 [ollama.com](https://ollama.com) 註冊 / 登入
2. 進 [ollama.com/settings/keys](https://ollama.com/settings/keys) 建立一把 API key
3. 設進環境變數：

```bash
export OLLAMA_API_KEY="your_api_key_here"
```

建議寫進 `.env`，並確認 `.env` 有進 `.gitignore`。這篇的範例會用 `python-dotenv` 讀取。

### 4.2 安裝套件

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`：

```
ollama>=0.6
pydantic>=2.0
python-dotenv>=1.0

openai>=1.0             # [〈用 OpenAI SDK 相容層接既有生態系〉](docs/03-openai-compat.md)：OpenAI 相容層
mcp>=2.0                # [〈接上 MCP〉](docs/04-mcp.md)：MCP 整合
langchain>=1.0          # [〈接框架與可觀測性〉](docs/08-frameworks-observability.md)
langchain-ollama>=1.0   # [〈接框架與可觀測性〉](docs/08-frameworks-observability.md)
langfuse>=4.0           # [〈接框架與可觀測性〉](docs/08-frameworks-observability.md)
opentelemetry-proto>=1.0  # 選用：假 Langfuse server 解碼 OTLP 用
```

### 4.3 驗證連線

```bash
python examples/00_check_setup.py
```

這支會一次檢查五件事，並且在失敗時直接給你可以照做的修復指令：

```
1. API Key        ✅ 讀到了（長度 57，開頭 a58f…）
2. 連線與可用模型   ✅ 連得上，帳號可見 18 個模型
     ✅ gpt-oss:120b    可用
     ✅ gpt-oss:20b     可用
     ✅ gemma4:31b      可用
     —  qwen3.5:397b   需要訂閱
3. 套件           ✅ ollama / pydantic / dotenv …
4. 本地 Ollama     ✅ embedding 可用：embeddinggemma（768 維）
結果  環境就緒，可以從 examples/01_hello_cloud.py 開始。
```

**注意第 2 項會實際去打每個模型**（只要一個 token，GPU 時間可忽略），因為「列在清單上」不等於「你的方案能用」——這是最多人卡住的地方。

只想確認連線的話，一行 curl 也可以：

```bash
curl https://ollama.com/api/tags -H "Authorization: Bearer $OLLAMA_API_KEY"
```

### 4.4 選模型

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

### 4.5 範例檔案一覽

| 檔案 | 對應章節 | 內容 |
| --- | --- | --- |
| `examples/00_check_setup.py` | [〈環境準備〉](#4-環境準備) | **先跑這支**：環境自檢，失敗時給你修復指令 |
| `examples/_client.py` | — | 共用 client（內建重試），其他範例都從這裡取 |
| `examples/01_hello_cloud.py` | [〈Hello Cloud〉](#5-hello-cloud第一次呼叫) | 第一次呼叫（串流） |
| `examples/02_tool_calling.py` | [〈Agent 的心臟〉](#6-agent-的心臟tool-calling) | 最小 tool calling |
| `examples/03_agent_loop.py` | [〈手刻 Agent Loop〉](#7-手刻-agent-loop) | ⭐ Agent Loop 骨架，全文最核心 |
| `examples/04_codebase_agent.py` | [〈實戰〉](#8-實戰一個會讀專案的-codebase-agent) | 完整的 Codebase Agent |
| `examples/05_structured_output.py` | [〈結構化輸出〉](docs/01-structured-output.md) | 結構化輸出（兩種可行方案） |
| `examples/06_streaming_agent.py` | [〈串流與 Thinking〉](docs/02-streaming.md) | 串流版 Agent Loop |
| `examples/07_openai_compat.py` | [〈用 OpenAI SDK 相容層接既有生態系〉](docs/03-openai-compat.md) | OpenAI SDK 相容層 |
| `examples/mcp_server_demo.py` | [〈接上 MCP〉](docs/04-mcp.md) | 示範用的 MCP Server（工單系統） |
| `examples/08_mcp_agent.py` | [〈接上 MCP〉](docs/04-mcp.md) | MCP ↔ Ollama 橋接 + Agent |
| `examples/11_model_router.py` | [〈實際使用情境〉](docs/05-cost.md) | 分層路由降本，含成本量測 |
| `examples/rag_common.py` | [〈RAG〉](docs/06-rag.md) | 切塊與 BM25（純手寫，零依賴） |
| `examples/corpus/handbook.md` | [〈RAG〉](docs/06-rag.md) | RAG 範例用的示範語料 |
| `examples/12_rag_cloud_only.py` | [〈RAG〉](docs/06-rag.md) | 純雲端 agentic RAG（零 embedding） |
| `examples/13_rag_hybrid.py` | [〈RAG〉](docs/06-rag.md) | 混合式：本地 embedding + 雲端生成 |
| `examples/corpus/eval_set.json` | [〈RAG〉](docs/06-rag.md) | RAG 評估標註集（15 題） |
| `examples/14_rag_eval.py` | [〈RAG〉](docs/06-rag.md) | RAG 自動化評估（兩層指標） |
| `examples/09_langchain_agent.py` | [〈接框架與可觀測性〉](docs/08-frameworks-observability.md) | LangChain 接雲端 + Langfuse callback |
| `examples/10_langfuse_tracing.py` | [〈接框架與可觀測性〉](docs/08-frameworks-observability.md) | 用 `@observe` 追蹤手刻 Agent Loop |
| `examples/tools/fake_langfuse_server.py` | [〈接框架與可觀測性〉](docs/08-frameworks-observability.md) | 假的 Langfuse 接收端，免帳號驗證 trace |

> 檔名的數字是**閱讀順序**，不是章節編號——`11_model_router.py` 對應[〈實際使用情境〉](docs/05-cost.md)。
> 照數字順序跑就對了。

文章裡的程式碼片段為了自我完整會重複寫 client 設定，實際檔案則統一 `from _client import MODEL, get_client`。直接 `python examples/03_agent_loop.py` 這樣執行即可，Python 會自動把 `examples/` 加進 import 路徑。

---

## 5. Hello Cloud：第一次呼叫

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

## 6. Agent 的心臟：Tool Calling

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

## 7. 手刻 Agent Loop

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

## 8. 實戰：一個會讀專案的 Codebase Agent

現在做一個真的有用的東西：**丟給它一個問題，它自己去翻專案目錄、讀檔、搜尋，然後回答你**。

工具設計三個：`list_files`、`read_file`、`search_code`。全部限制在指定的根目錄底下，避免 Agent 亂跑到 `/etc`。

完整程式在 `examples/04_codebase_agent.py`，這裡看關鍵片段。

### 8.1 工具定義（含安全邊界）

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

### 8.2 Agent 主體

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

### 8.3 這裡面藏著的幾個工程重點

**系統提示要寫「停止條件」。** `蒐集到足夠資訊後，直接給出結論` 這句很關鍵。沒有它，模型常常會過度探索，一路讀到 max_turns 才停。

**工具粒度要抓對。** 太細（`open_file` / `read_line` / `close_file`）會讓輪數暴增，每一輪都是一次雲端呼叫；太粗（`analyze_whole_project`）等於你自己把邏輯寫死了，Agent 沒有發揮空間。一個好的判準是：**每個工具對應人類會做的一個動作**。

**Context 是會累積的。** 每輪的 tool 結果都留在 `messages` 裡。長任務跑到後面 context 會很可觀，這也是為什麼工具輸出的截斷上限那麼重要。

---

## 接下來：進階主題

核心路徑到這裡結束。你已經有一個會用工具、有防護、能處理真實資料的 Agent。

底下每一篇都是獨立的，**需要哪塊看哪塊**：

| 主題 | 難度 | 什麼時候需要 |
| --- | --- | --- |
| [結構化輸出：讓 Agent 吐出可以直接用的資料](docs/01-structured-output.md) | 中階 | 要把 Agent 的輸出交給下游程式時 |
| [串流與 Thinking：把思考過程秀出來](docs/02-streaming.md) | 中階 | 任務跑很久、使用者盯著空白畫面時 |
| [用 OpenAI SDK 相容層接既有生態系](docs/03-openai-compat.md) | 中階 | 手上已經有一套用 OpenAI SDK 寫的程式 |
| [接上 MCP：不用自己寫工具](docs/04-mcp.md) | 進階 | 要接 GitHub、檔案系統、資料庫等現成工具 |
| [實際使用情境：GPU 時間計費下的成本控制](docs/05-cost.md) | 中階 | 開始在意帳單時 |
| [RAG：為什麼雲端做不了，以及兩條可走的路](docs/06-rag.md) | 進階 | 要讓 Agent 讀你自己的文件 |
| [正式上線前要處理的事](docs/07-production.md) | 中階 | 要真的上線時 |
| [接框架與可觀測性：LangChain / Langfuse / LangSmith](docs/08-frameworks-observability.md) | 進階 | 想用框架，或 Agent 出錯時查不出原因 |
| [多輪對話與 Context 管理](docs/09-memory-context.md) | 中階 | 使用者問第二句「那東京呢？」的時候 |
| [測試 Agent](docs/10-testing.md) | 中階 |
| [跨 Session 長期記憶](docs/11-persistent-memory.md) | 進階 |
| [部署上線](docs/12-deployment.md) | 中階 |
| [硬體選型](docs/13-hardware.md) | 中階 | 想安心重構，不必每次人工點一遍 |

完整索引：[docs/README.md](docs/README.md)

---

## 結語

回頭看，整篇文章的核心其實只有[〈手刻 Agent Loop〉](#7-手刻-agent-loop)那個迴圈：

> **呼叫模型 → 有工具就執行 → 結果餵回去 → 重複，直到模型說完成。**

Agent 不是什麼魔法，是「一個會用工具的 while 迴圈」。難的從來不是這個迴圈，而是：

- 工具的**粒度**跟**描述**寫得夠不夠好（這決定 Agent 聰不聰明，比換模型有效）
- 停止條件跟輪數上限（這決定它會不會失控）
- 工具輸出的截斷跟錯誤處理（這決定它撐不撐得住真實資料）

[〈接上 MCP〉](docs/04-mcp.md)的 MCP 也沒有改變這個結構——它只是把「工具從哪來」外包出去。迴圈還是那個迴圈，只是 `tools` 陣列不再是你手寫的函式，而是從別台 Server 撈回來的。這也是為什麼建議先手刻過[〈手刻 Agent Loop〉](#7-手刻-agent-loop)那 40 行再接 MCP：知道底下在做什麼，之後任何框架你都能一眼看穿。

Ollama Cloud 在這條路上的貢獻，是把「模型能力」這個變數從等式裡拿掉了——你不用再為了跑得動而遷就一個工具呼叫時好時壞的小模型，也不用為了跑大模型去租 GPU。同一套 API，本地跟雲端隨你切。

### 下一步可以玩的

- **把 MCP 接到底**：[〈接上 MCP〉](docs/04-mcp.md)只接了一台自製 Server，實務上會同時掛 GitHub、檔案系統、資料庫，再加上工具篩選與人工確認關卡
- **多 Agent 分工**：一個 planner 負責拆任務，多個 worker 平行執行，最後 synthesizer 收斂
- **RAG 進階**：[〈RAG〉](docs/06-rag.md)只做到單一檢索工具，實務上還有混合檢索（向量 + BM25 加權合併）、重排模型、以及檢索品質的離線評估
- **記憶層**：把跨 session 的結論存進向量庫，讓 Agent 記得上次的發現
- **降本實驗**：拿[〈實際使用情境〉](docs/05-cost.md)的量測方法，對你自己的流量算一次損益平衡點，再決定路由策略

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
- [LangChain 文件](https://python.langchain.com/)
- [langchain-ollama 套件](https://pypi.org/project/langchain-ollama/)
- [Langfuse 文件](https://langfuse.com/docs)
- [Langfuse 自架指南](https://langfuse.com/self-hosting)
- [LangSmith 文件](https://docs.smith.langchain.com/)
