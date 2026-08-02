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

這篇有點長。**不要從頭讀到尾**，照你的目的挑一條：

### 🚀 路徑 A：三十分鐘搞懂 Agent 是什麼

> 第 1 → 4 → 5 → 6 → 7 節

跑完你會有一個會用工具的 Agent，並且理解它的本質只是一個 while 迴圈。**大部分人到這裡就夠用了。**

```bash
python examples/00_check_setup.py     # 先確認環境
python examples/01_hello_cloud.py
python examples/02_tool_calling.py
python examples/03_agent_loop.py      # ← 全文最核心的 40 行
```

### 🛠 路徑 B：做一個真的能用的東西

> 路徑 A ＋ 第 8 → 9 → 13 節

加上真實 Agent 需要的防護（路徑邊界、輸出截斷、例外處理）、可用的結構化輸出，以及成本控制。

### 📚 路徑 C：完整讀完

> 全部十七節

包含 MCP 工具生態、RAG、框架整合、可觀測性、上線準備。建議當作**參考手冊**，需要哪塊看哪塊，不用一次讀完。

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

**進階——讓它能真的上線**

| 節 | 標題 | 難度 |
| --- | --- | --- |
| 9 | [結構化輸出：讓 Agent 吐出可以直接用的資料](#9-結構化輸出讓-agent-吐出可以直接用的資料) | 中階 |
| 10 | [串流與 Thinking：把思考過程秀出來](#10-串流與-thinking把思考過程秀出來) | 中階 |
| 11 | [用 OpenAI SDK 相容層接既有生態系](#11-用-openai-sdk-相容層接既有生態系) | 中階 |
| 12 | [接上 MCP：不用自己寫工具](#12-接上-mcp不用自己寫工具) | 進階 |
| 13 | [實際使用情境：GPU 時間計費下的成本控制](#13-實際使用情境gpu-時間計費下的成本控制) | 中階 |
| 14 | [RAG：為什麼雲端做不了，以及兩條可走的路](#14-rag為什麼雲端做不了以及兩條可走的路) | 進階 |
| 15 | [正式上線前要處理的事](#15-正式上線前要處理的事) | 中階 |
| 16 | [接框架與可觀測性：LangChain / Langfuse / LangSmith](#16-接框架與可觀測性langchain--langfuse--langsmith) | 進階 |
| 17 | [結語](#17-結語) | — |

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
│   while True:  呼叫模型 → 有工具就執行 → 結果餵回 → 重複        │  ← 第 7 節
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
  第 2 節       第 6/12 節      第 14 節        第 16 節
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
| **評估資料集** | 回答「我的 RAG 到底準不準」 | **比向量資料庫更該先做**。見第 14.9 節 |

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

**組合二：能讀公司文件的問答 Agent**（第 14 節）

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
| `pydantic` | ≥2.0 | 第 9 節 結構化輸出 | 必要 |
| `openai` | ≥1.0 | 第 11 節 相容層 | 選用 |
| `mcp` | ≥2.0 | 第 12 節 MCP | 選用 |
| `langchain`, `langchain-ollama` | ≥1.0 | 第 16 節 | 選用 |
| `langfuse` | ≥4.0 | 第 16 節 | 選用 |
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

openai>=1.0             # 第 11 節：OpenAI 相容層
mcp>=2.0                # 第 12 節：MCP 整合
langchain>=1.0          # 第 16 節
langchain-ollama>=1.0   # 第 16 節
langfuse>=4.0           # 第 16 節
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
| `examples/00_check_setup.py` | 第 4 節 | **先跑這支**：環境自檢，失敗時給你修復指令 |
| `examples/_client.py` | — | 共用 client（內建重試），其他範例都從這裡取 |
| `examples/01_hello_cloud.py` | 第 5 節 | 第一次呼叫（串流） |
| `examples/02_tool_calling.py` | 第 6 節 | 最小 tool calling |
| `examples/03_agent_loop.py` | 第 7 節 | ⭐ Agent Loop 骨架，全文最核心 |
| `examples/04_codebase_agent.py` | 第 8 節 | 完整的 Codebase Agent |
| `examples/05_structured_output.py` | 第 9 節 | 結構化輸出（兩種可行方案） |
| `examples/06_streaming_agent.py` | 第 10 節 | 串流版 Agent Loop |
| `examples/07_openai_compat.py` | 第 11 節 | OpenAI SDK 相容層 |
| `examples/mcp_server_demo.py` | 第 12 節 | 示範用的 MCP Server（工單系統） |
| `examples/08_mcp_agent.py` | 第 12 節 | MCP ↔ Ollama 橋接 + Agent |
| `examples/11_model_router.py` | 第 13 節 | 分層路由降本，含成本量測 |
| `examples/rag_common.py` | 第 14 節 | 切塊與 BM25（純手寫，零依賴） |
| `examples/corpus/handbook.md` | 第 14 節 | RAG 範例用的示範語料 |
| `examples/12_rag_cloud_only.py` | 第 14 節 | 純雲端 agentic RAG（零 embedding） |
| `examples/13_rag_hybrid.py` | 第 14 節 | 混合式：本地 embedding + 雲端生成 |
| `examples/corpus/eval_set.json` | 第 14 節 | RAG 評估標註集（15 題） |
| `examples/14_rag_eval.py` | 第 14 節 | RAG 自動化評估（兩層指標） |
| `examples/09_langchain_agent.py` | 第 16 節 | LangChain 接雲端 + Langfuse callback |
| `examples/10_langfuse_tracing.py` | 第 16 節 | 用 `@observe` 追蹤手刻 Agent Loop |
| `examples/tools/fake_langfuse_server.py` | 第 16 節 | 假的 Langfuse 接收端，免帳號驗證 trace |

> 檔名的數字是**閱讀順序**，不是章節編號——`11_model_router.py` 對應第 13 節。
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

## 9. 結構化輸出：讓 Agent 吐出可以直接用的資料

Agent 常常是更大流程裡的一環，下游需要的是 JSON 不是散文。

### 9.1 先講一個雲端的坑：`format` 不會生效

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

### 9.2 方案 A：借 tool calling 來做 schema 約束（推薦）

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

### 9.3 方案 B：prompt + 去圍籬 + 驗證（備援）

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

### 9.4 跟 Agent Loop 怎麼搭

**不要在 Agent Loop 裡同時掛工具跟 `submit`**，模型會搞不清楚該繼續查資料還是該交卷。分兩階段：

1. **蒐集階段**：Agent Loop 帶著真正的工具跑（第 8 節那套），直到模型不再要求工具
2. **收斂階段**：把蒐集到的結論當成 prompt，單獨呼叫一次 `structured_via_tool()` 交出 JSON

多花一次呼叫，換來下游拿得到乾淨的資料，划算。

---

## 10. 串流與 Thinking：把思考過程秀出來

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

## 11. 用 OpenAI SDK 相容層接既有生態系

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

## 12. 接上 MCP：不用自己寫工具

第 8 節那三個工具是我們自己刻的。但如果 Agent 要能查 GitHub、讀 Slack、打資料庫、操作瀏覽器呢？一個一個手寫，寫到天荒地老。

**MCP（Model Context Protocol）解決的就是這件事**：它把「工具」標準化成一個協定，任何人寫的 MCP Server 都能被任何 MCP Client 使用。現在 GitHub、Notion、Playwright、PostgreSQL、檔案系統……都有現成的 Server。接上去，你的 Agent 立刻多幾十個工具。

### 12.1 先講清楚：Ollama 沒有內建 MCP

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

### 12.2 準備一台 MCP Server

為了讓範例能獨立跑，先自己寫一台。用 MCP Python SDK 寫 Server 跟寫普通函式差不多——`@mcp.tool()` 會從型別標註和 docstring 自動生出 schema，跟第 6 節 Ollama SDK 的做法是同一個思路。

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

### 12.3 橋接層

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

### 12.4 Agent 主體

因為 MCP SDK 是 async 的，Agent Loop 也要換成 `AsyncClient`。除此之外，結構跟第 8 節一模一樣：

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

### 12.5 接現成的 MCP Server

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

### 12.6 接 MCP 之後才會遇到的坑

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

## 13. 實際使用情境：GPU 時間計費下的成本控制

前面十一節都在講「怎麼做出來」。這節講「做出來之後怎麼不燒錢」——而且會顛覆幾個直覺。

Ollama Cloud **按 GPU 時間計費，不是按 token**。這一句話改變了所有優化的方向，但大部分人（包括我一開始）還是帶著 token 計費的習慣在思考。

### 13.1 三個實測數字，先打破直覺

同一個分類任務「這句話是 bug / feature / question？」，三個免費可用的模型都答對了 `bug`。成本呢（三次的中位數）：

| 模型 | GPU 時間 | 輸出 tokens | 答案 |
| --- | --- | --- | --- |
| `gemma4:31b` | **0.41s** | 2 | ✅ bug |
| `gpt-oss:120b` | 1.41s | 101 | ✅ bug |
| `gpt-oss:20b` | 1.77s | 100 | ✅ bug |

三個反直覺的地方：

**一、`gpt-oss:20b` 比 `gpt-oss:120b` 還貴。** 參數少了六分之五，GPU 時間反而多 25%。「小模型比較便宜」在按 token 計費時成立，在按 GPU 時間計費時**不成立**。

**二、`gpt-oss` 回一個詞要燒 100 個 token。** 因為它是 thinking 模型，回答之前一定先推理。分類這種不需要思考的任務，那 100 個 token 全是浪費。

**三、`think=False` 關不掉。** 我試過了：

| 設定 | GPU 時間 | thinking 長度 |
| --- | --- | --- |
| `think=True` | 1.64s | 303 字元 |
| `think=False` | 1.12s | **292 字元** |

thinking 內容還在，只是不一定回傳給你。**選了 thinking 模型就是選了它的成本，這個開關省不掉。** 要省就得換非 thinking 模型。

### 13.2 最有效的一招：叫它閉嘴

這是我整輪實測中投報率最高的發現。同樣是「把『今天天氣很好』翻譯成英文」：

| 設定 | GPU 時間 | 輸出 tokens | 回答 |
| --- | --- | --- | --- |
| `gemma4:31b` 無約束 | **3.29s** | 222 | 「這句話最常見的翻譯有以下幾種，視你想表達的口氣而定：**1. 最通用…」 |
| `gemma4:31b` + 一句系統提示 | **0.32s** | 7 | 「The weather is great today.」 |

**十倍。** 那句系統提示就是：

```python
TERSE = '直接給答案，不要解釋、不要列出多個選項、不要加註解。'
```

在 GPU 時間計費下，**囉嗦才是成本主因，模型大小是次要的**。一個會長篇大論的小模型，比一個簡潔的大模型還貴。

同一招對 thinking 模型效果有限（`gpt-oss:120b` 只從 1.77s 降到 1.05s），因為省不掉的推理 token 佔了大部分——這又回到 12.1 的結論。

### 13.3 分層路由：先判難度再決定用誰

有了上面兩點，降本策略就清楚了：**用便宜的非 thinking 模型擋掉簡單請求，只有真的需要推理的才升級。**

```python
CHEAP = 'gemma4:31b'        # 非 thinking，回應短
SMART = 'gpt-oss:120b'      # thinking 模型，貴但強

ROUTER_PROMPT = """判斷以下請求需要哪個等級的模型處理，只回一個字：

S = 簡單：分類、抽取、改寫、翻譯、格式轉換、事實查詢
C = 複雜：多步驟推理、數學計算、程式除錯、需要權衡的決策

請求：{task}

只回 S 或 C，不要任何其他文字。"""


def route_and_answer(task: str) -> dict:
    verdict, route_cost = ask(CHEAP, ROUTER_PROMPT.format(task=task))
    is_complex = verdict.upper().startswith('C')
    chosen = SMART if is_complex else CHEAP

    # 簡單任務才壓長度；複雜任務需要模型把推理寫出來，壓了反而會答錯
    answer, answer_cost = ask(chosen, task, terse=not is_complex)
    ...
```

實測結果（`examples/11_model_router.py`，每項取三次中位數）：

```
簡單任務（2 題）平均 GPU：路由 0.79s vs 全用大模型 1.97s → 省 60%
複雜任務（1 題）平均 GPU：路由 6.92s vs 全用大模型 6.14s → 多花 13%

損益平衡點：簡單請求需佔總流量 22% 以上才划算
```

**這招不是免費的。** 每個請求都多一次路由呼叫，複雜任務因此貴 13%。只有簡單請求佔比夠高才划得來——實測這個組合的損益平衡點是 22%。大部分真實產品的流量遠高於這個比例，所以通常划算，但**你該用自己的流量算一次，而不是相信任何人的百分比**。

> 我第一版寫這個範例時沒加 `TERSE`，結果總計「多花 13%」——分層路由反而更貴。原因就是 12.2 那件事：便宜模型囉嗦起來一點都不便宜。這個負面結果我留在範例的註解裡，因為它比成功案例更有教育意義。

### 13.4 量成本一定要取中位數

雲端 GPU 時間的波動比想像中大。同一題 `gpt-oss:120b`、同樣的 prompt、`temperature=0`，連跑六次：

```
1.10s  1.21s  1.14s  3.69s  0.93s  1.07s
```

中位數 1.12s，但有一次是 3.69s——**三倍**。`gemma4:31b` 就穩得多（0.24s～0.34s）。

這代表：

- **單次測量會給你完全相反的結論。** 我有一輪測量剛好撞上尖峰，得出「路由多花 20%」，重測就變成「省 60%」
- 範例程式因此把量測改成取三次中位數（`REPS = 3`）
- 上線後如果有 SLA 要求，要看的是 **p95 而不是平均**

### 13.5 其他省 GPU 時間的招式

**減少輪數比壓縮 prompt 有效。** 每一輪都是一次完整的請求往返。工具設計得好、一輪能拿到足夠資訊，比省 context 有用得多——這也是第 8 節「工具粒度要抓對」的成本面理由。

**`max_turns` 是帳單的保險絲。** 失控的 Agent Loop 可以在你沒注意時跑掉大量 GPU 時間。

**用 token 數當 context 膨脹的儀表板。** 雖然不直接對應帳單，但第 16 節那張 trace 上 input token 從 200 → 297 → 351 一路長的曲線，是你判斷「該截斷工具輸出了」最直觀的指標。

**簡單任務不要開 `think=True`。** 對能關的模型有效；對 `gpt-oss` 系列沒用（見 13.1）。

---

## 14. RAG：為什麼雲端做不了，以及兩條可走的路

### 14.1 先講結論：缺的是 embedding，不是 RAG

網路上「Ollama Cloud + RAG」的教學不少，但幾乎都是拿本地 Ollama 的做法直接改個 host 就貼上來，沒人實際打過那個端點。我打了，結果是：

**Ollama Cloud 的 hosted API 沒有可用的 embedding。**

RAG 有三個環節，它只缺一個：

| 環節 | 誰做 | Ollama Cloud |
| --- | --- | --- |
| ① 向量化（embedding） | 模型 | ❌ **沒有** |
| ② 檢索（相似度／關鍵字比對） | 你的程式 | ✅ 不受影響 |
| ③ 生成（把檢索結果餵給模型） | 模型 | ✅ 這是它的本業 |

### 14.2 先搞懂 embedding 在 RAG 裡做什麼

如果你還不確定 embedding 是什麼，這節缺的那一塊會很抽象。先花三分鐘。

**核心問題：電腦不懂「意思」。**

RAG 要解決的是：使用者問一句話，怎麼從一萬份文件裡找出該讀哪一段。最直覺的做法是關鍵字比對，但它比對的是**字**，不是**意思**：

```
使用者問：「同事寫的東西太大包看不完」
文件寫的：「PR 超過四百行就該拆」
```

兩句講同一件事，**字面重疊是零**。BM25 只會茫然。

**Embedding 做的事：把「意思」變成座標。**

它把一段文字壓成固定長度的數字陣列（`embeddinggemma` 是 768 個浮點數）：

```
"PR 超過四百行就該拆"       → [ 0.021, -0.334,  0.118, … ]
"同事寫的東西太大包看不完"   → [ 0.019, -0.341,  0.122, … ]   ← 座標很接近
"資料庫遷移要可回滾"         → [-0.412,  0.088, -0.290, … ]   ← 離很遠
```

關鍵在於這個空間是**依語意排列**的：意思相近的句子座標就相近。於是「找相關段落」這個模糊問題，變成「算哪個向量離查詢向量最近」這個純數學問題：

```python
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
```

**它在管線中出現兩次，很多人只想到一次：**

```
【建索引時，跑一次】
  文件 → 切塊 ─────┐
                   ├──→ ┌───────────┐ ──→ 向量存起來
【查詢時，每次都跑】 │     │ EMBEDDING │
  使用者問題 ───────┘     └───────────┘ ──→ 查詢向量
                                              ↓
                                     算相似度、排序（純數學，不用模型）
                                              ↓
                                   取前 k 段 → 塞進 prompt → 生成模型回答
```

**這就是「雲端缺 embedding」為什麼致命**：不是只有建索引要用，**每次查詢都要跑一次**。而且查詢向量必須用同一個模型產生，否則座標系不同，算出來的距離毫無意義。

所以在 Ollama Cloud 上：

- ❌ 無法把文件轉成向量 → **建不了索引**
- ❌ 無法把問題轉成向量 → **即使索引是別處建的也查不了**
- ✅ 生成那一半完全正常

一句話：**Ollama Cloud 能讀你找到的東西，但沒辦法幫你找。**

### 14.3 實測證據

```
POST https://ollama.com/api/embed        → 401 {"error": "unauthorized"}
POST https://ollama.com/api/embeddings   → 404 {"error": "path \"/api/embeddings\" not found"}
POST https://ollama.com/v1/embeddings    → 404 {"error": "path \"/v1/embeddings\" not found"}
```

第一行是關鍵：`/api/embed` **回 401 而不是 404**，代表端點存在但拒絕服務。而且用的是同一把 key——同一支程式打 `gpt-oss:120b` 的 chat 完全正常。所以不是憑證問題，是這個能力沒開。

另外兩個端點連路由都沒有。至於模型清單，`/api/tags` 的 18 個我逐一比對過，`nomic-embed-text`、`mxbai-embed-large`、`embeddinggemma`、`bge-m3`、`all-minilm`、`qwen3-embedding` **全部不在**，一個 embedding 模型都沒有。

### 14.4 為什麼會這樣設計

這其實完全符合它的商業模型。Ollama Cloud **按 GPU 時間計費**，賣點是「跑你本機跑不動的大模型」。而 embedding 模型的特性正好相反：

- **小** — `embeddinggemma` 才幾百 MB，CPU 就跑得動
- **快** — 單次毫秒級，用 GPU 秒數計價根本收不到錢
- **呼叫次數極多** — 建索引時每個 chunk 打一次，一萬份文件就是一萬次

embedding 是**最不需要雲端 GPU 的那類工作**。放進按 GPU 時間計費的服務，對雙方都不划算。理解這點之後，「缺 embedding」就不是缺陷，而是定位。

### 14.5 共同的框法：RAG-as-a-tool

在講兩條路之前，先講一件比選型更重要的事：**不要寫固定管線。**

傳統 RAG 是：切塊 → 向量化 → 檢索 top-k → 塞進 prompt → 生成。這條管線只查一次，關鍵字沒抓好就完了。

既然我們前面十二節都在做 Agent，就該用 Agent 的方式做 RAG：**把檢索做成一個工具，讓模型自己決定要不要查、查幾次、換什麼說法再查。**

```python
def search_handbook(query: str, top_k: int = 3) -> str:
    """在內部工程手冊中搜尋與問題相關的段落。

    Args:
        query: 搜尋關鍵字或問題。用具體的詞效果較好，例如「資料庫遷移 回滾」
        top_k: 要回傳幾個段落，預設 3，最多 5

    Returns:
        相關段落的內容，每段標明出處與標題
    """
```

這其實就是第 8 節 Codebase Agent 的模式——那裡的 `search_code` 本質上就是一個 retrieval tool，只是當時沒叫它 RAG。

系統提示裡有一句是整個 RAG 品質的關鍵：

```
- **只根據檢索到的內容回答**。手冊沒寫的就說「手冊未涵蓋」，不要用常識補
```

實測問「公司的年假規定是幾天？」（語料完全沒有），Agent 查完 `list_topics` 就回答「手冊未涵蓋此項內容。」——沒有幻覺。這句約束比任何檢索調校都有效。

### 14.6 方案 B：純雲端，零 embedding

既然不能用向量，就用關鍵字。BM25 手寫不到六十行，不需要任何套件。

中文的麻煩是沒有空格。這裡用 **bigram（相鄰兩字）當詞**，土砲但意外地夠用：

```python
def _tokenize(text: str) -> list[str]:
    """中英混合的粗略斷詞。

    中文沒有空格，這裡用 bigram（相鄰兩字）當作詞——土砲但夠用，
    而且不需要額外的斷詞套件。英文與數字照原樣切。
    """
    lowered = text.lower()
    latin = re.findall(r'[a-z0-9_]+', lowered)
    han = re.findall(r'[一-鿿]', lowered)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return latin + han + bigrams
```

切塊策略也值得一提：**依 Markdown 標題切，不要用固定字元數。** 技術文件的每個 `##` 段落本來就是一個語意單位，這樣切幾乎不會出現「一句話被腰斬」。

跑起來：

```bash
python examples/12_rag_cloud_only.py "資料庫遷移要注意什麼？"
```

```
語料：6 個段落（純關鍵字檢索，零 embedding）
[第 1 輪檢索] list_topics({})
[第 2 輪檢索] search_handbook({'query': '資料庫遷移 注意', 'top_k': 5})
  → 【handbook.md — 資料庫遷移】(相關度 19.0)

依據《資料庫遷移》段落，遷移時需要注意：
1. 每個 Pull Request 只能包含一個遷移檔…
```

注意它先 `list_topics` 摸清楚有哪些主題，再決定搜什麼——這是 agentic 的價值，固定管線做不到。

**優點**：完全不碰本地服務，Docker、Serverless、CI 都能跑，跟第 2 節的模式 B 一致。

### 14.7 方案 A：混合式——本地 embedding + 雲端生成

雲端沒有 embedding，但**你本機的 Ollama 有**。兩邊一起用：

```
向量化  → 本機 http://localhost:11434（embeddinggemma，768 維）
生成    → 雲端 https://ollama.com（gpt-oss:120b）
```

```bash
ollama pull embeddinggemma
```

```python
from ollama import Client

client = get_client()                                 # 雲端：負責生成
local = Client(host='http://localhost:11434')         # 本機：負責向量化


def _embed(texts: list[str]) -> list[list[float]]:
    return local.embed(model='embeddinggemma', input=texts)['embeddings']
```

這剛好是第 2 節兩種連線模式併用的實例。而且有個常被忽略的好處：**語料不出境**。只有問題和檢索到的片段會送到雲端，整份文件庫留在本地——對有合規要求的場景，這個架構比純雲端更好，不是妥協。

> **踩雷紀錄**：我第一次打本機 `/api/embed` 拿到 `501 This server does not support embeddings. Start it with --embeddings`。看起來像要重啟服務，其實不是——那是因為當時本機只有 `gemma4:12b` 這個**聊天**模型，runner 自然沒開 embedding 支援。`ollama pull` 一個真正的 embedding 模型就好，Ollama 本身不用動。

### 14.8 向量 vs 關鍵字：什麼時候值得付這個複雜度

這是本節最實用的部分。同一個語料，同一批問題，兩種檢索各自撈到什麼（`--compare`）：

| 問題 | 向量檢索 | BM25 |
| --- | --- | --- |
| 資料庫遷移要注意什麼？ | 〈資料庫遷移〉✅ | 〈資料庫遷移〉✅ |
| 告警一直響但都不是真的問題？ | 〈監控與告警〉✅ | 〈監控與告警〉✅ |
| 出事的時候先做什麼？ | 〈事故處理〉✅ | 〈事故處理〉✅ |
| 不小心把密碼寫進程式碼了 | 〈密鑰管理〉✅ | 〈密鑰管理〉✅ |
| **怎麼避免改壞正式環境** | 〈部署流程〉✅ | 〈資料庫遷移〉❌ |
| **同事寫的東西太大包看不完** | 〈程式碼審查〉✅ | 〈事故處理〉❌ |

結論很清楚，而且跟直覺不太一樣：

**用詞跟文件接近時，BM25 完全夠用，甚至更準。** 前四題兩者一致，而 BM25 不用跑模型、不用建索引、毫秒級回應。我原本以為向量會全面勝出，實測打臉。

**一旦使用者用自己的話問，向量才拉開差距。** 「太大包看不完」對上文件裡的「PR 超過四百行就該拆」——沒有任何字面重疊，BM25 只能瞎猜。

所以判準是：**你的使用者會用文件的詞彙，還是自己的話？** 內部工具、開發者查手冊 → BM25 就好。面向一般使用者的客服、產品問答 → 值得付出 embedding 的複雜度。

還有一個細節值得注意。問「新人第一天該看什麼」（語料完全沒涵蓋）時，向量的最高相似度只有 **0.200**，遠低於命中時的 0.3～0.6。**這個分數可以當作「沒有相關內容」的訊號**——設個門檻，低於它就直接回「找不到」，比讓模型讀一堆不相關的段落再自己判斷可靠。BM25 的分數沒有這種可比性，這是向量檢索另一個不太被提起的優勢。

---

### 14.9 怎麼自動驗證 RAG 的正確率

上一節那張對照表是我手動看六題湊出來的。**手動看永遠不夠**——你改了切塊策略、換了 embedding 模型、調了 prompt，怎麼知道是變好還是變壞？

答案是建一個評估集，然後每次改動都重跑。`examples/14_rag_eval.py` 是可執行的版本。

**關鍵設計：分成兩層，因為 RAG 有兩個獨立的失敗點。**

```
檢索錯了      → 模型再強也答不對    ← 第一層抓這個
檢索對了但答錯 → prompt 或生成的問題  ← 第二層抓這個
```

混在一起量，你永遠不知道該去修哪邊。

**第一層：檢索指標（不花 GPU 時間，秒級跑完）**

```bash
python examples/14_rag_eval.py
```

```
BM25 關鍵字檢索（13 題）
  Recall@1 85%   Recall@3 100%   MRR 0.923
    paraphrase     Recall@1 71% (5/7)
    vocab-match    Recall@1 100% (6/6)

向量語意檢索（13 題）
  Recall@1 92%   Recall@3 100%   MRR 0.962
    paraphrase     Recall@1 86% (6/7)
    vocab-match    Recall@1 100% (6/6)
```

這組數字**用量化的方式證實了 14.8 節的定性結論**：用詞一致時兩者都是 100%，差距全部出在 `paraphrase`（71% vs 86%）。我在標註時就替每題標了 `phrasing`，所以能直接看到差在哪裡——**分組看比看總分有用得多**。

這層不呼叫任何模型，所以可以在每次改動後無痛重跑。

**第二層：端到端指標（要跑模型）**

```bash
python examples/14_rag_eval.py --end-to-end
```

三個指標：

| 指標 | 量什麼 | 實測 |
| --- | --- | --- |
| 答案正確率 | 答案含不含標註的關鍵事實 | 85–100%（跑次間有波動） |
| **拒答正確率** | 語料沒涵蓋時有沒有正確說「未涵蓋」 | 100% |
| **忠實度** | LLM-as-judge：有沒有講檢索內容裡沒有的東西 | **77%** |

**忠實度只有 77% 是這整節最有價值的發現。** 答案全對，但約四分之一的回答夾帶了手冊裡沒寫的內容——模型在自行發揮。例如問「不小心把密碼寫進程式碼了」，它回答時加了「立即將密碼從程式碼中移除，並提交新的 commit」，聽起來很合理，但手冊沒這樣寫。

**這種失敗用眼睛看幾乎抓不到**，因為答案讀起來完全正確。只有把檢索脈絡和答案並排丟給判官模型才會現形。

三個實作上的提醒：

**一、判官也要用 tool calling 拿結構化結果。** 雲端不強制 `format`（第 9 節），所以我讓判官只回 Y/N 單字並用字首判斷。

**二、評估工具自己也會有 bug。** 我第一版用字串比對，模型答「15 分鐘」卻被判成缺少「十五分鐘」——**RAG 是對的，是評估工具製造了假陰性**。中文數字與阿拉伯數字是這類評估最常見的坑。修法是讓每項關鍵事實可以是「可接受寫法」的陣列：

```json
{ "answer_must_contain": [["十五分鐘", "15 分鐘", "15分鐘"]] }
```

**三、端到端指標有隨機性。** 同一份評估集連跑兩次，正確率從 100% 掉到 85%，純粹是模型換了個說法。所以**單次結果不能當驗收標準**，要看趨勢，或把 `temperature` 固定並跑多次取多數。

還有個容易誤解的地方：BM25 的 Recall@1 只有 85%，但端到端正確率卻是 100%。原因是 Recall@3 有 100%，而 Agent 一次取 3–5 段、還能換關鍵字重查——**agentic 檢索有自我修正能力，檢索的第一名不完美不代表答案會錯**。這也是為什麼兩層都要量。

---

## 15. 正式上線前要處理的事

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

第 13 節整節都在講這件事，這裡只補上線相關的兩點：

- **配額與 rate limit**。免費方案除了模型受限（見 4.4），也有速率限制。正式跑之前看一下 [ollama.com/pricing](https://ollama.com/pricing) 的方案配額，並且在程式裡把 429 當成需要退避的錯誤處理。
- **設預算告警**。GPU 時間計費的好處是可預估，前提是你真的有在看。把第 16 節的 trace 接起來，至少能回答「這個月哪個 Agent 吃掉最多時間」。

### 可觀測性

至少把每輪的 `tool_calls` 和 `thinking` 記下來。Agent 出錯時你要回答的問題是「它為什麼決定做這件事」，而那個答案只在 thinking 裡。

### 安全邊界

- API key 用環境變數或 secret manager，不要進版控
- **能寫入的工具（刪檔、發 API、送信）一律加人工確認關卡**，別讓模型直接觸發不可逆操作
- 路徑、SQL、shell 指令這類參數，一律當成不可信輸入來驗證——模型的輸出本質上就是使用者可以間接控制的內容

### 資料落地

雲端模式下對話內容會送到 Ollama 的伺服器。有合規要求的資料，該過濾就過濾，或改跑本地模型。

---

## 16. 接框架與可觀測性：LangChain / Langfuse / LangSmith

前面十二節都是手刻的。這節談什麼時候該把工作交給框架，以及一個比框架更該優先做的事——**可觀測性**。

### 16.1 先講判準：什麼時候該用框架

我的看法是：**先手刻過一次，再決定要不要用框架。** 你已經讀到這裡了，第 7 節那 40 行迴圈你看得懂，那麼任何框架對你來說都只是那個迴圈的包裝，不會變成黑盒子。

該用框架的訊號：

- 需要 **checkpoint / 中斷續跑**（LangGraph 的 `checkpointer`）
- 需要 **人在迴圈中**（`interrupt_before`）
- 團隊已經有一整套 LangChain 的 retriever、memory、chain 要沿用

不該用的訊號：

- 只是要跑個工具迴圈——手刻的 40 行更好懂、更好改、debug 時堆疊更淺
- 想要框架幫你解決模型能力問題——它不會

### 16.2 LangChain 接 Ollama Cloud

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

這是第 9 節那個坑的連鎖效應，**值得單獨標出來**：

```python
llm.with_structured_output(Country).invoke('Tell me about Canada.')
# ❌ OutputParserException: Invalid json output: ## Canada – A Snapshot | Aspect | ...
```

原因是它預設走 JSON mode，而 Ollama Cloud 不強制 `format`。解法是明講走 function calling——等於第 9 節的方案 A，只是由 LangChain 代勞：

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

### 16.3 Langfuse：把「可觀測性」從口號變成東西

第 15 節說「至少把每輪的 `tool_calls` 和 `thinking` 記下來」。Langfuse 就是拿來做這件事的：開源、可自架、與供應商無關。

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

注意 input token 從 200 → 297 → 351 一路長——**這就是 context 累積的樣子**，第 8 節講的「工具輸出一定要截斷」在這張圖上看得最清楚。

接 LangChain 的話更省事，掛個 callback 就好：

```python
from langfuse.langchain import CallbackHandler

agent.invoke({...}, config={'callbacks': [CallbackHandler()]})
```

實測會自動抓到 17 個 span（`ChatOllama` generation ×6、`tools` chain ×3、各工具節點、`LangGraph` 根節點），完全不用改業務程式碼。

完整範例在 `examples/10_langfuse_tracing.py`。

### 16.4 兩種 Langfuse 部署方式

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

### 16.5 LangSmith：什麼時候才選它

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

## 17. 結語

回頭看，整篇文章的核心其實只有第 7 節那個迴圈：

> **呼叫模型 → 有工具就執行 → 結果餵回去 → 重複，直到模型說完成。**

Agent 不是什麼魔法，是「一個會用工具的 while 迴圈」。難的從來不是這個迴圈，而是：

- 工具的**粒度**跟**描述**寫得夠不夠好（這決定 Agent 聰不聰明，比換模型有效）
- 停止條件跟輪數上限（這決定它會不會失控）
- 工具輸出的截斷跟錯誤處理（這決定它撐不撐得住真實資料）

第 12 節的 MCP 也沒有改變這個結構——它只是把「工具從哪來」外包出去。迴圈還是那個迴圈，只是 `tools` 陣列不再是你手寫的函式，而是從別台 Server 撈回來的。這也是為什麼建議先手刻過第 7 節那 40 行再接 MCP：知道底下在做什麼，之後任何框架你都能一眼看穿。

Ollama Cloud 在這條路上的貢獻，是把「模型能力」這個變數從等式裡拿掉了——你不用再為了跑得動而遷就一個工具呼叫時好時壞的小模型，也不用為了跑大模型去租 GPU。同一套 API，本地跟雲端隨你切。

### 下一步可以玩的

- **把 MCP 接到底**：第 12 節只接了一台自製 Server，實務上會同時掛 GitHub、檔案系統、資料庫，再加上工具篩選與人工確認關卡
- **多 Agent 分工**：一個 planner 負責拆任務，多個 worker 平行執行，最後 synthesizer 收斂
- **RAG 進階**：第 14 節只做到單一檢索工具，實務上還有混合檢索（向量 + BM25 加權合併）、重排模型、以及檢索品質的離線評估
- **記憶層**：把跨 session 的結論存進向量庫，讓 Agent 記得上次的發現
- **降本實驗**：拿第 13 節的量測方法，對你自己的流量算一次損益平衡點，再決定路由策略

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
