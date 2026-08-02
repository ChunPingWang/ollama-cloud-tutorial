"""一台假的 Langfuse 接收端，用來驗證 trace 真的送得出去。

沒有 Langfuse 帳號、也不想起一整套 docker compose 時，用這台來確認
「我的程式有沒有正確產生 trace」——它會把收到的 OTLP 內容摘要印出來。

用法（另開一個終端機）：
    python examples/tools/fake_langfuse_server.py

然後把 Langfuse 指到它：
    export LANGFUSE_HOST=http://localhost:3999
    export LANGFUSE_PUBLIC_KEY=pk-lf-fake
    export LANGFUSE_SECRET_KEY=sk-lf-fake
    python examples/10_langfuse_tracing.py

這只驗證「送得出去、內容對不對」，不是 Langfuse 的替代品——
沒有 UI、沒有儲存、沒有查詢。
"""

import gzip
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 3999
SPAN_LOG = os.environ.get('FAKE_LANGFUSE_LOG', '/tmp/fake_langfuse_spans.jsonl')
GREY, CYAN, GREEN, RESET = '\033[90m', '\033[36m', '\033[32m', '\033[0m'

_seen_spans = 0


def _record(entry: dict) -> None:
    """同時寫檔，這樣背景執行也能事後檢查收到什麼。"""
    with open(SPAN_LOG, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _protobuf_to_dict(body: bytes) -> dict | None:
    """把 OTLP 的 protobuf payload 轉成跟 JSON 版一樣的結構。

    需要 opentelemetry-proto（pip install opentelemetry-proto）。
    沒裝的話回 None，呼叫端會退回「只記長度」的模式。
    """
    try:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
    except ImportError:
        return None

    try:
        message = ExportTraceServiceRequest()
        message.ParseFromString(body)
        return MessageToDict(message, preserving_proto_field_name=False)
    except Exception:                                 # noqa: BLE001
        return None


def _decode_attr(value: dict) -> object:
    """OTLP 的 AnyValue 是個 tagged union，拆出實際的值。"""
    for key in ('stringValue', 'intValue', 'doubleValue', 'boolValue'):
        if key in value:
            return value[key]
    if 'arrayValue' in value:
        return [_decode_attr(v) for v in value['arrayValue'].get('values', [])]
    return value


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:                       # noqa: N802
        global _seen_spans

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        if self.headers.get('Content-Encoding') == 'gzip':
            body = gzip.decompress(body)

        print(f'\n{CYAN}── POST {self.path} ({len(body)} bytes) ──{RESET}')
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Langfuse 的 OTLP exporter 預設送 protobuf，不是 JSON
            payload = _protobuf_to_dict(body)
            if payload is None:
                print(f'{GREY}  (無法解析的 body，長度 {len(body)}){RESET}')
                _record({'path': self.path, 'undecodable_bytes': len(body)})
                self._ok()
                return

        for resource in payload.get('resourceSpans', []):
            for scope in resource.get('scopeSpans', []):
                for span in scope.get('spans', []):
                    _seen_spans += 1
                    attrs = {
                        a['key']: _decode_attr(a.get('value', {}))
                        for a in span.get('attributes', [])
                    }
                    name = span.get('name', '?')
                    obs_type = attrs.get('langfuse.observation.type', '-')
                    print(f'{GREEN}  ✓ span #{_seen_spans}: {name}{RESET} '
                          f'{GREY}[type={obs_type}]{RESET}')
                    _record({'span': name, 'type': obs_type,
                             'attrs': {k: str(v)[:400] for k, v in attrs.items()}})
                    for key in ('langfuse.observation.input',
                                'langfuse.observation.output',
                                'langfuse.observation.model.name',
                                'langfuse.trace.name'):
                        if key in attrs:
                            short = str(attrs[key]).replace('\n', ' ')[:110]
                            print(f'{GREY}      {key.split(".")[-1]}: {short}{RESET}')

        self._ok()

    def _ok(self) -> None:
        self.send_response(207)      # Langfuse 的批次接收端回 207 Multi-Status
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"partialSuccess":{}}')

    def do_GET(self) -> None:        # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def log_message(self, *args) -> None:
        pass                          # 關掉預設的 access log，輸出才乾淨


if __name__ == '__main__':
    print(f'假 Langfuse 接收端啟動於 http://localhost:{PORT}')
    print('把 LANGFUSE_HOST 指到這裡，然後執行你的 Agent。Ctrl-C 結束。\n')
    try:
        HTTPServer(('localhost', PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print(f'\n總共收到 {_seen_spans} 個 span。')
        sys.exit(0)
