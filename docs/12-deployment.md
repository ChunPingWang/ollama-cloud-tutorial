# 部署上線：把 Agent 包成服務

> **難度**：中階　|　**前置**：[正式上線前要處理的事](07-production.md)
> 服務化要處理的五件事，以及為什麼容器裡不用裝 Ollama

[← 跨 Session 長期記憶](11-persistent-memory.md)　·　[全部進階主題](README.md)　·　[回到核心路徑 →](../README.md)

---

## 1. 為什麼容器裡沒有 Ollama

這是整篇最重要的一件事，也是[核心路徑第 2 節](../README.md#2-兩種連線模式先搞懂差別)那個「兩種連線模式」的實際回報。

因為我們用 **direct API 模式**，`ollama.com` 就是遠端的推論主機，容器只需要一個 HTTP client：

```dockerfile
FROM python:3.13-slim
RUN pip install --no-cache-dir "ollama>=0.6" "python-dotenv>=1.0"
```

實測建出來的映像檔：

```
ollama-agent-demo:test  209MB
```

如果走的是**本地代理模式**（模型名稱帶 `-cloud`），你得把整個 Ollama runtime 塞進映像檔，而且容器裡還要跑 `ollama serve` 並登入——好幾 GB，還多一個要顧的行程。

**當初選 direct API 模式，回報就在這裡。**

## 2. 服務化要處理的五件事

`examples/18_deploy_server.py` 刻意只用 Python 標準函式庫，不引入 FastAPI/Flask。理由跟 [`rag_common.py`](06-rag.md) 一樣：讓你看清楚服務化真正要處理的是什麼。換成 FastAPI 只是換個殼，底下這五件事一件都不會少。

### ① 啟動時就驗證設定

```python
def preflight():
    """把「會失敗的設定」在啟動時就炸掉，而不是留給第一個使用者。

    容器編排系統會因為啟動失敗而不把流量導進來；
    但如果你等到第一個請求才發現沒有 API key，那個使用者就吃到 500 了。
    """
    if not os.environ.get('OLLAMA_API_KEY'):
        sys.exit('[FATAL] 缺少 OLLAMA_API_KEY，拒絕啟動。')
    ...
    if not rag.CHUNKS:
        sys.exit('[FATAL] 語料是空的，拒絕啟動。')
```

實測沒給 key 時：

```
$ docker run --rm ollama-agent-demo:test
[FATAL] 缺少 OLLAMA_API_KEY，拒絕啟動。
```

「語料是空的就拒絕啟動」這條特別值得加——路徑打錯、`COPY` 漏了、volume 沒掛上，症狀都是「服務正常但答案都是查無資料」，很難查。

### ② 健康檢查要分兩種

```python
if self.path == '/healthz':
    # 存活檢查：行程還活著就好，不要在這裡打模型——
    # 那會讓雲端的健康檢查變成你的帳單。
    ...
elif self.path == '/readyz':
    # 就緒檢查：設定齊全、語料載入完成
    self._json(200, {'status': 'ready', 'chunks': len(AGENT.CHUNKS)})
```

**健康檢查絕對不要呼叫模型。** 每 30 秒一次、乘上多個副本，那是一筆穩定的固定支出，而且對「服務有沒有活著」這個問題完全沒有增加資訊。

### ③ 請求逾時

Agent 迴圈可能跑很久（多輪工具呼叫），不能讓連線無限掛著：

```python
worker = threading.Thread(target=work, daemon=True)
worker.start()
worker.join(REQUEST_TIMEOUT)

if worker.is_alive():
    # 執行緒無法強制中止，只能放生並回應逾時。
    # 這也是實務上會改用 asyncio 或子行程的原因之一。
    self._json(504, {'error': f'處理超過 {REQUEST_TIMEOUT} 秒'})
```

那個註解是真話，不是藉口——**Python 沒辦法強制殺死執行緒**。要真正取消進行中的工作，得用 `asyncio`（[MCP 那篇](04-mcp.md)的 Agent 就是 async 的）或把 Agent 跑在子行程裡。

### ④ 並行控制與過載保護

```python
_inflight = threading.BoundedSemaphore(int(os.environ.get('MAX_CONCURRENT', '4')))

if not _inflight.acquire(blocking=False):
    # 滿載時明確回 429，讓上游知道要退避，
    # 而不是讓請求無限排隊、最後全部逾時。
    self._json(429, {'error': '同時處理的請求已達上限，請稍後再試'})
```

Agent 是 I/O bound（等雲端回應），所以執行緒模型夠用。但**上限一定要設**：每個進行中的請求都在燒 GPU 時間，無限並行等於無限帳單。

順帶一提，輸入長度也要設上限：

```python
if len(question) > MAX_QUESTION_LEN:
    # 沒有上限的話，一個超長 prompt 就能讓單一請求燒掉大量 GPU 時間
    self._json(413, {'error': f'question 超過 {MAX_QUESTION_LEN} 字'})
```

### ⑤ 優雅關閉

```python
def shutdown(signum, _frame):
    # 收到 SIGTERM（容器停止）時：健康檢查先轉紅讓流量停止進來，
    # 手上的請求做完再真正關閉。直接 exit 會讓使用者看到連線中斷。
    _shutting_down.set()
    threading.Thread(target=server.shutdown, daemon=True).start()

signal.signal(signal.SIGTERM, shutdown)
```

實測：

```
$ docker stop -t 15 $CID
[ok] 4.82s question='PR 太大要怎麼辦？'
[signal] 收到 SIGTERM，開始優雅關閉
[bye] 已關閉
```

這對 Agent 特別重要，因為單一請求可能已經燒了幾十秒的 GPU 時間——中途砍掉那些錢就白花了。

## 3. Dockerfile 的幾個決定

```dockerfile
# 用非 root 執行。容器逃逸的第一道防線，成本是零。
RUN useradd --create-home --uid 10001 agent
USER agent

# 只裝服務真正需要的。langchain/langfuse/mcp 是教學用的選用套件，
# 生產映像檔沒必要背著它們。
RUN pip install --no-cache-dir "ollama>=0.6" "python-dotenv>=1.0"
```

**API key 一定要在執行期注入：**

```dockerfile
# 絕不 COPY 進映像檔也不寫在 ENV。
# 映像檔會被推到 registry、被別人 pull、每一層都留在歷史裡。
```

```bash
docker run -e OLLAMA_API_KEY=... ollama-agent-demo
```

正式環境用 secret manager（k8s Secret、Cloud Run secret、AWS Secrets Manager），不要用 `-e` 傳——那會出現在 `docker inspect` 和行程列表裡。

`.dockerignore` 也要記得擋掉 `.env`：

```
.venv/
.git/
.env
docs/
examples/memory.db
```

## 4. 完整實測

```bash
docker build -t ollama-agent-demo .
docker run -d -p 8080:8080 -e OLLAMA_API_KEY="$OLLAMA_API_KEY" ollama-agent-demo
```

```
$ curl localhost:8080/healthz
{"status": "ok"}

$ curl localhost:8080/readyz
{"status": "ready", "chunks": 6}

$ curl -X POST localhost:8080/ask -H 'Content-Type: application/json' \
       -d '{"question":"PR 太大要怎麼辦？"}'
{"answer": "依據〈程式碼審查〉段落：\n\n- **超過 400 行**的 PR 必須拆分…",
 "elapsed_s": 4.82}
```

錯誤處理也都測過：

| 請求 | 回應 |
| --- | --- |
| `POST /ask` body 不是 JSON | `400 body 不是合法 JSON` |
| `POST /ask` 沒有 question | `400 缺少 question` |
| `GET /nope` | `404 not found` |
| 沒有 `OLLAMA_API_KEY` | 拒絕啟動 |

## 5. 狀態要放哪裡

這個範例是**無狀態**的——語料是唯讀的，烘進映像檔。但如果你加上[長期記憶](11-persistent-memory.md)，那個 SQLite 就是狀態，容器一重啟就沒了。

| 狀態 | 放哪裡 |
| --- | --- |
| 語料（唯讀、不常變） | 烘進映像檔，最簡單 |
| 語料（會變） | 掛 volume 或開機時從物件儲存拉 |
| 長期記憶、對話歷史 | **外部資料庫**（Postgres / 託管 SQLite / Redis） |

容器裡的 SQLite 只適合單副本 + 掛載 volume。要水平擴展就得換成外部資料庫——這不是 Agent 特有的問題，是所有服務都一樣。

## 6. 換成 FastAPI 要改什麼

如果你要用框架（多數團隊會），對應關係是：

| 這裡的做法 | FastAPI |
| --- | --- |
| `preflight()` | `lifespan` 的啟動段 |
| `/healthz` `/readyz` | 一樣自己寫，邏輯不變 |
| `threading` + `join(timeout)` | `asyncio.wait_for`（**可以真的取消**，比執行緒好） |
| `BoundedSemaphore` | `asyncio.Semaphore` 或反向代理層限流 |
| `signal.signal` | `lifespan` 的關閉段 |

**用 async 版本是真的比較好**，因為 Agent 等的是網路 I/O，而且逾時能真正取消。這個範例用執行緒只是為了不引入依賴、把邏輯攤開給你看。

---

[← 跨 Session 長期記憶](11-persistent-memory.md)　·　[全部進階主題](README.md)　·　[回到核心路徑 →](../README.md)
