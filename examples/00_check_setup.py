"""環境自檢：跑任何範例之前先跑這支。

【這支教你什麼】確認你的環境到底缺什麼，而不是在第五個範例才發現
【花費】幾乎不花 GPU 時間（只送一個字給模型）

它會檢查五件事，每一項失敗時直接給你可以照做的修復指令：

  1. API key 讀得到嗎
  2. 連得上 ollama.com 嗎
  3. 你的方案實際可以用哪些模型（列出來不等於你能用）
  4. 各章節需要的套件裝了沒
  5. 本地 Ollama 與 embedding 模型（第 13 節混合式 RAG 才需要）

用法:
    python examples/00_check_setup.py
"""

import importlib.util
import os
import sys

from dotenv import load_dotenv

load_dotenv()

GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    '\033[32m', '\033[31m', '\033[33m', '\033[90m', '\033[1m', '\033[0m')

OK, FAIL, WARN = f'{GREEN}✅{RESET}', f'{RED}❌{RESET}', f'{YELLOW}⚠️ {RESET}'

# 這篇文章實際驗證過的模型；免費方案通常只有前三個能用
PROBE_MODELS = ['gpt-oss:120b', 'gpt-oss:20b', 'gemma4:31b',
                'qwen3.5:397b', 'deepseek-v4-flash', 'glm-5.2']

PACKAGES = [
    ('ollama', '全部章節', True),
    ('pydantic', '第 8 節 結構化輸出', True),
    ('dotenv', '全部章節', True),
    ('openai', '第 10 節 OpenAI 相容層', False),
    ('mcp', '第 11 節 MCP 整合', False),
    ('langchain_ollama', '第 15 節 LangChain', False),
    ('langfuse', '第 15 節 Langfuse', False),
]

problems: list[str] = []


def section(title: str) -> None:
    print(f'\n{BOLD}{title}{RESET}')


# --------------------------------------------------------------------------

section('1. API Key')

api_key = os.environ.get('OLLAMA_API_KEY')
if api_key:
    print(f'  {OK} 讀到了（長度 {len(api_key)}，開頭 {api_key[:4]}…）')
else:
    print(f'  {FAIL} 找不到 OLLAMA_API_KEY')
    print(f'{GREY}     到 https://ollama.com/settings/keys 建立，然後：{RESET}')
    print(f'{GREY}       cp .env.example .env   # 再把 key 填進去{RESET}')
    print(f'{GREY}     注意：非互動式 shell 不會載入 ~/.zshrc，'
          f'export 過不代表程式讀得到{RESET}')
    problems.append('缺 OLLAMA_API_KEY')


# --------------------------------------------------------------------------

section('2. 連線與可用模型')

if not api_key:
    print(f'  {GREY}（沒有 key，跳過）{RESET}')
else:
    from ollama import Client, ResponseError

    client = Client(host='https://ollama.com',
                    headers={'Authorization': f'Bearer {api_key}'})

    try:
        listed = [m['model'] for m in client.list()['models']]
        print(f'  {OK} 連得上，帳號可見 {len(listed)} 個模型')
    except Exception as exc:                          # noqa: BLE001
        print(f'  {FAIL} 連不上：{str(exc)[:100]}')
        problems.append('連不上 ollama.com')
        listed = []

    if listed:
        print(f'{GREY}     實際測試哪些能用（列出來 ≠ 你的方案能用）：{RESET}')
        usable = []
        for model in PROBE_MODELS:
            if model not in listed:
                print(f'       {GREY}—  {model:20s} 不在你的清單裡{RESET}')
                continue
            try:
                # 只要一個字，把 GPU 時間壓到最低
                client.chat(model=model,
                            messages=[{'role': 'user', 'content': '嗨'}],
                            options={'num_predict': 1})
                print(f'       {OK} {model:20s} 可用')
                usable.append(model)
            except ResponseError as exc:
                reason = ('需要訂閱' if 'subscription' in str(exc).lower()
                          else str(exc)[:40])
                print(f'       {GREY}—  {model:20s} {reason}{RESET}')

        if not usable:
            print(f'  {FAIL} 沒有任何模型可用')
            problems.append('沒有可用模型')
        else:
            default = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')
            if default in usable:
                print(f'  {OK} 預設模型 {default} 可用')
            else:
                print(f'  {WARN} 預設模型 {default} 不可用，'
                      f'請設定 OLLAMA_MODEL={usable[0]}')
                problems.append(f'預設模型不可用，建議改用 {usable[0]}')


# --------------------------------------------------------------------------

section('3. 套件')

for module, where, required in PACKAGES:
    if importlib.util.find_spec(module):
        print(f'  {OK} {module:20s} {GREY}{where}{RESET}')
    elif required:
        print(f'  {FAIL} {module:20s} {where} — 必要')
        problems.append(f'缺套件 {module}')
    else:
        print(f'  {GREY}—  {module:20s} {where} — 未裝，該章節無法執行{RESET}')


# --------------------------------------------------------------------------

section('4. 本地 Ollama（只有第 13 節混合式 RAG 需要）')

try:
    from ollama import Client as LocalClient

    local = LocalClient(host='http://localhost:11434')
    local_models = [m['model'] for m in local.list()['models']]
    print(f'  {OK} 本地 Ollama 有在跑，{len(local_models)} 個模型')

    embed_models = [m for m in local_models
                    if any(k in m for k in ('embed', 'bge', 'minilm'))]
    if embed_models:
        try:
            dim = len(local.embed(model=embed_models[0],
                                  input=['測試'])['embeddings'][0])
            print(f'  {OK} embedding 可用：{embed_models[0]}（{dim} 維）')
        except Exception as exc:                      # noqa: BLE001
            print(f'  {FAIL} {embed_models[0]} 無法 embed：{str(exc)[:60]}')
    else:
        print(f'  {GREY}—  沒有 embedding 模型。第 13 節混合式 RAG 需要：{RESET}')
        print(f'{GREY}       ollama pull embeddinggemma{RESET}')
except Exception:                                     # noqa: BLE001
    print(f'  {GREY}—  本地 Ollama 沒在跑（只影響第 13 節的混合式 RAG，'
          f'其他章節不需要）{RESET}')


# --------------------------------------------------------------------------

section('結果')

if problems:
    print(f'  {RED}有 {len(problems)} 個問題要先解決：{RESET}')
    for p in problems:
        print(f'    • {p}')
    sys.exit(1)

print(f'  {GREEN}環境就緒，可以從 examples/01_hello_cloud.py 開始。{RESET}')
