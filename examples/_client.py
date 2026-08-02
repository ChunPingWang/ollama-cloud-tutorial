"""共用的 Ollama Cloud client 設定。

所有範例都從這裡取得 client，避免每支程式重複一樣的樣板。
"""

import os
import sys

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

MODEL = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')


def get_client() -> Client:
    """建立直連 ollama.com 的 client（direct API 模式）。

    注意：這個模式下模型名稱不帶 -cloud 後綴。
    """
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        sys.exit(
            '找不到 OLLAMA_API_KEY。\n'
            '請到 https://ollama.com/settings/keys 建立 key，然後：\n'
            '  export OLLAMA_API_KEY="..."\n'
            '或複製 .env.example 成 .env 並填入。'
        )
    return Client(
        host='https://ollama.com',
        headers={'Authorization': 'Bearer ' + api_key},
    )
