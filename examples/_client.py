"""共用的 Ollama Cloud client 設定。

【這支教你什麼】連線設定唯一的真實來源，以及為什麼重試不是選配
【不用單獨執行】其他範例 `from _client import MODEL, get_client` 取用

所有範例都從這裡取得 client，避免每支程式重複一樣的樣板。

這裡也內建了重試。不是為了嚴謹好看——實測跑多輪 Agent 時真的會撞到
Ollama Cloud 偶發的 500，Agent 輪數越多機率越高。沒有重試的話，
跑到第七輪炸掉，前面六輪的 GPU 時間就白花了。
"""

import os
import sys
import time

from dotenv import load_dotenv
from ollama import Client, ResponseError

load_dotenv()

MODEL = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')
CLOUD_HOST = 'https://ollama.com'


def get_api_key() -> str:
    """讀取 API key，沒有的話給出可以照做的指示而不是 KeyError。"""
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        sys.exit(
            '找不到 OLLAMA_API_KEY。\n'
            '請到 https://ollama.com/settings/keys 建立 key，然後：\n'
            '  export OLLAMA_API_KEY="..."\n'
            '或複製 .env.example 成 .env 並填入。\n'
            '（提示：非互動式 shell 不會自動載入 ~/.zshrc，'
            '所以 export 過不代表這支程式讀得到）'
        )
    return api_key


class RetryingClient(Client):
    """在 Client 外面包一層指數退避重試。

    只重試 5xx 與連線錯誤。4xx 不重試——認證錯誤、模型名稱打錯、
    參數格式不對，重試一百次也一樣，只是白燒時間。
    """

    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_retries = max_retries

    def chat(self, *args, **kwargs):
        last_error = None
        for attempt in range(self._max_retries):
            try:
                return super().chat(*args, **kwargs)
            except ResponseError as exc:
                status = getattr(exc, 'status_code', None)
                if status is not None and status < 500:
                    raise                       # 4xx 是你的問題，重試沒用
                last_error = exc
            except (ConnectionError, TimeoutError) as exc:
                last_error = exc

            if attempt < self._max_retries - 1:
                backoff = 2 ** attempt
                print(f'  (雲端回應異常，{backoff}s 後重試 '
                      f'{attempt + 2}/{self._max_retries})', file=sys.stderr)
                time.sleep(backoff)

        raise last_error


def get_client(max_retries: int = 3) -> RetryingClient:
    """建立直連 ollama.com 的 client（direct API 模式）。

    注意：這個模式下模型名稱不帶 -cloud 後綴。
    """
    return RetryingClient(
        host=CLOUD_HOST,
        headers={'Authorization': 'Bearer ' + get_api_key()},
        max_retries=max_retries,
    )
