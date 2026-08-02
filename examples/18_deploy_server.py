"""把 Agent 包成 HTTP 服務，準備上線。

【這支教你什麼】Agent 服務化要處理的五件事，以及為什麼容器裡不用裝 Ollama
【前置知識】04_codebase_agent.py、12_rag_cloud_only.py
【下一支】—（範例結束）

這支刻意只用 Python 標準函式庫，不引入 FastAPI/Flask。
理由跟 rag_common.py 一樣：讓你看清楚服務化真正要處理的是什麼，
而不是被框架的樣板蓋住。實務上換成 FastAPI 只是換個殼，
底下這五件事一件都不會少：

  1. 啟動時就驗證設定（不要等第一個請求進來才發現沒有 API key）
  2. 健康檢查端點（k8s / Cloud Run / ALB 都要）
  3. 請求逾時（Agent 迴圈可能跑很久，不能讓連線無限掛著）
  4. 並行處理（Agent 是 I/O bound，單執行緒會塞住）
  5. 優雅關閉（收到 SIGTERM 要把手上的請求做完）

用法:
    python examples/18_deploy_server.py
    curl localhost:8080/healthz
    curl -X POST localhost:8080/ask -H 'Content-Type: application/json' \\
         -d '{"question":"資料庫遷移要注意什麼？"}'
"""

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.environ.get('PORT', '8080'))          # Cloud Run 等平台會注入 PORT
REQUEST_TIMEOUT = float(os.environ.get('AGENT_TIMEOUT', '120'))
MAX_QUESTION_LEN = 2000

_shutting_down = threading.Event()
_inflight = threading.BoundedSemaphore(
    int(os.environ.get('MAX_CONCURRENT', '4')))


# --------------------------------------------------------------------------
# 1. 啟動時就驗證設定
# --------------------------------------------------------------------------

def preflight() -> None:
    """把「會失敗的設定」在啟動時就炸掉，而不是留給第一個使用者。

    容器編排系統會因為啟動失敗而不把流量導進來；
    但如果你等到第一個請求才發現沒有 API key，那個使用者就吃到 500 了。
    """
    if not os.environ.get('OLLAMA_API_KEY'):
        sys.exit('[FATAL] 缺少 OLLAMA_API_KEY，拒絕啟動。')

    try:
        import importlib
        rag = importlib.import_module('12_rag_cloud_only')
    except Exception as exc:                          # noqa: BLE001
        sys.exit(f'[FATAL] 載入 Agent 失敗：{exc}')

    if not rag.CHUNKS:
        sys.exit('[FATAL] 語料是空的，拒絕啟動。')

    print(f'[boot] 語料 {len(rag.CHUNKS)} 段，模型 {rag.MODEL}', flush=True)
    return rag


AGENT = preflight()


# --------------------------------------------------------------------------
# 服務
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'agent-demo'

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- 2. 健康檢查 ----
    def do_GET(self) -> None:                        # noqa: N802
        if self.path == '/healthz':
            # 存活檢查：行程還活著就好，不要在這裡打模型——
            # 那會讓雲端的健康檢查變成你的帳單。
            if _shutting_down.is_set():
                self._json(503, {'status': 'shutting_down'})
            else:
                self._json(200, {'status': 'ok'})
        elif self.path == '/readyz':
            # 就緒檢查：設定齊全、語料載入完成
            self._json(200, {'status': 'ready', 'chunks': len(AGENT.CHUNKS)})
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self) -> None:                       # noqa: N802
        if self.path != '/ask':
            self._json(404, {'error': 'not found'})
            return
        if _shutting_down.is_set():
            self._json(503, {'error': '服務正在關閉'})
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError):
            self._json(400, {'error': 'body 不是合法 JSON'})
            return

        question = str(payload.get('question', '')).strip()
        if not question:
            self._json(400, {'error': '缺少 question'})
            return
        if len(question) > MAX_QUESTION_LEN:
            # 輸入長度要設上限。沒有上限的話，一個超長 prompt
            # 就能讓單一請求燒掉大量 GPU 時間。
            self._json(413, {'error': f'question 超過 {MAX_QUESTION_LEN} 字'})
            return

        # ---- 4. 並行控制 ----
        if not _inflight.acquire(blocking=False):
            # 滿載時明確回 429，讓上游知道要退避，
            # 而不是讓請求無限排隊、最後全部逾時。
            self._json(429, {'error': '同時處理的請求已達上限，請稍後再試'})
            return

        try:
            self._run_agent(question)
        finally:
            _inflight.release()

    # ---- 3. 請求逾時 ----
    def _run_agent(self, question: str) -> None:
        result: dict = {}
        started = time.monotonic()

        def work():
            try:
                result['answer'] = AGENT.run_agent(question, verbose=False)
            except Exception as exc:                  # noqa: BLE001
                result['error'] = f'{type(exc).__name__}: {exc}'

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        worker.join(REQUEST_TIMEOUT)

        elapsed = round(time.monotonic() - started, 2)

        if worker.is_alive():
            # 執行緒無法強制中止，只能放生並回應逾時。
            # 這也是實務上會改用 asyncio 或子行程的原因之一。
            print(f'[timeout] {elapsed}s question={question[:40]!r}', flush=True)
            self._json(504, {'error': f'處理超過 {REQUEST_TIMEOUT} 秒',
                             'elapsed_s': elapsed})
            return

        if 'error' in result:
            print(f'[error] {result["error"]}', flush=True)
            # 不要把內部例外原文吐給呼叫端，那可能洩漏路徑或設定
            self._json(500, {'error': '內部錯誤', 'elapsed_s': elapsed})
            return

        print(f'[ok] {elapsed}s question={question[:40]!r}', flush=True)
        self._json(200, {'answer': result['answer'], 'elapsed_s': elapsed})

    def log_message(self, *args) -> None:
        pass                                          # 用我們自己的結構化 log


# --------------------------------------------------------------------------
# 5. 優雅關閉
# --------------------------------------------------------------------------

def main() -> None:
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    server.daemon_threads = True

    def shutdown(signum, _frame):
        # 收到 SIGTERM（容器停止）時：健康檢查先轉紅讓流量停止進來，
        # 手上的請求做完再真正關閉。直接 exit 會讓使用者看到連線中斷。
        print(f'[signal] 收到 {signal.Signals(signum).name}，開始優雅關閉',
              flush=True)
        _shutting_down.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f'[boot] 監聽 0.0.0.0:{PORT}　逾時 {REQUEST_TIMEOUT}s', flush=True)
    server.serve_forever()
    server.server_close()
    print('[bye] 已關閉', flush=True)


if __name__ == '__main__':
    main()
