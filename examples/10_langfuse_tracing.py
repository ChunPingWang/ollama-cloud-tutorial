"""用 Langfuse 追蹤手刻的 Agent Loop。

第 12 節說「至少把每輪的 tool_calls 和 thinking 記下來」——這支就是把那句話
變成可執行的東西。Agent 出錯時你要回答的是「它為什麼決定做這件事」，
而那個答案只在 thinking 裡；Langfuse 讓它變成可以事後翻閱的樹狀 trace。

需要三個環境變數：
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://cloud.langfuse.com     # 自架就換成自己的位址

沒有 Langfuse 帳號也想確認 trace 送得出去的話，先另開終端機跑：
    python examples/tools/fake_langfuse_server.py
再把 LANGFUSE_HOST 指到 http://localhost:3999。

用法:
    python examples/10_langfuse_tracing.py "請計算 (11434 + 12341) * 412"
"""

import os
import sys

from dotenv import load_dotenv
from langfuse import get_client, observe

from _client import MODEL, get_client as get_ollama

load_dotenv()

client = get_ollama()
langfuse = get_client()

GREY, CYAN, RESET = '\033[90m', '\033[36m', '\033[0m'


# --------------------------------------------------------------------------
# 工具：加上 as_type='tool'，在 trace 上會顯示成工具節點
# --------------------------------------------------------------------------

@observe(as_type='tool')
def add(a: int, b: int) -> int:
    """把兩個整數相加"""
    return a + b


@observe(as_type='tool')
def multiply(a: int, b: int) -> int:
    """把兩個整數相乘"""
    return a * b


TOOLS = [add, multiply]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

SYSTEM_PROMPT = ('你只能透過 add 與 multiply 工具做算術，'
                 '嚴禁自行心算或直接寫出答案。')


# --------------------------------------------------------------------------
# 每一次模型呼叫記成一個 generation，才會有 model / input / output 欄位
# --------------------------------------------------------------------------

@observe(as_type='generation', name='ollama-chat')
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
        metadata={'total_duration_ns': getattr(response, 'total_duration', None)},
    )
    return response


@observe(as_type='agent', name='math-agent')
def run_agent(question: str, max_turns: int = 8) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = call_model(messages)
        messages.append(response.message)

        if response.message.thinking:
            print(f'{GREY}[第 {turn} 輪思考] '
                  f'{response.message.thinking.strip()[:160]}{RESET}')

        if not response.message.tool_calls:
            langfuse.update_current_span(metadata={'turns_used': turn})
            return response.message.content

        for tc in response.message.tool_calls:
            fn = AVAILABLE.get(tc.function.name)
            if fn is None:
                result = f'錯誤：不存在的工具 {tc.function.name}'
            else:
                print(f'{CYAN}[工具] {tc.function.name}({tc.function.arguments}){RESET}')
                result = fn(**tc.function.arguments)
                print(f'{GREY}  → {result}{RESET}')
            messages.append({
                'role': 'tool', 'tool_name': tc.function.name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'


if __name__ == '__main__':
    if not os.environ.get('LANGFUSE_PUBLIC_KEY'):
        sys.exit(
            '找不到 LANGFUSE_PUBLIC_KEY。\n'
            '雲端版請到 https://cloud.langfuse.com 建立專案取得金鑰，\n'
            '或先跑 python examples/tools/fake_langfuse_server.py 再設定：\n'
            '  export LANGFUSE_HOST=http://localhost:3999\n'
            '  export LANGFUSE_PUBLIC_KEY=pk-lf-fake\n'
            '  export LANGFUSE_SECRET_KEY=sk-lf-fake'
        )

    question = ' '.join(sys.argv[1:]) or '請計算 (11434 + 12341) * 412'
    print(f'{CYAN}問題：{question}{RESET}')

    answer = run_agent(question)
    print('\n' + '=' * 60)
    print(answer)

    # 短命的腳本一定要 flush，否則行程結束時 trace 還在緩衝區裡就沒了
    langfuse.flush()
    print(f'\n{GREY}trace 已送出至 {os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")}{RESET}')
