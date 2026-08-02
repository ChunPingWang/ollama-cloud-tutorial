"""方案 B：純雲端 agentic RAG——零 embedding。

【這支教你什麼】不靠向量也能做 RAG，而且檢索是 Agent 的工具、不是固定管線。
【前置知識】第 7 節的 Codebase Agent（同一個模式，換成文件語料）
【下一支】13_rag_hybrid.py：加上本地 embedding 做語意檢索

Ollama Cloud 的 hosted API 沒有可用的 embedding（實測 /api/embed 回 401，
/api/embeddings 與 /v1/embeddings 直接 404，模型清單裡也沒有 embedding 模型）。
所以要做純雲端 RAG，就不能靠向量。

這支程式改用「BM25 關鍵字檢索」，而且不是固定管線，是把檢索做成
Agent 的工具——模型自己決定要不要查、查幾次、換什麼關鍵字。

用法:
    python examples/12_rag_cloud_only.py "資料庫遷移要注意什麼？"
"""

import sys

from _client import MODEL, get_client
from rag_common import BM25, format_hits, load_chunks

client = get_client()

GREY, CYAN, RESET = '\033[90m', '\033[36m', '\033[0m'

CHUNKS = load_chunks()
INDEX = BM25([c['text'] for c in CHUNKS])


# --------------------------------------------------------------------------
# 檢索工具：交給 Agent 自己決定怎麼用
# --------------------------------------------------------------------------

def search_handbook(query: str, top_k: int = 3) -> str:
    """在內部工程手冊中搜尋與問題相關的段落。

    Args:
        query: 搜尋關鍵字或問題。用具體的詞效果較好，例如「資料庫遷移 回滾」
        top_k: 要回傳幾個段落，預設 3，最多 5

    Returns:
        相關段落的內容，每段標明出處與標題
    """
    hits = INDEX.rank(query, max(1, min(int(top_k), 5)))
    if not hits:
        return f'找不到與「{query}」相關的段落，請換個關鍵字再試一次。'
    return format_hits(CHUNKS, hits)


def list_topics() -> str:
    """列出手冊涵蓋哪些主題，用來決定該搜什麼關鍵字。"""
    return '\n'.join(f'- {c["title"]}（{c["source"]}）' for c in CHUNKS)


TOOLS = [search_handbook, list_topics]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

SYSTEM_PROMPT = """你是內部工程手冊的問答助理。

工作方式：
- 不確定該搜什麼時，先用 list_topics 看有哪些主題
- 用 search_handbook 檢索，關鍵字用具體的詞
- 第一次搜不到就換關鍵字再試，不要直接放棄
- **只根據檢索到的內容回答**。手冊沒寫的就說「手冊未涵蓋」，不要用常識補
- 回答時標明依據哪個段落（例如「依據〈資料庫遷移〉」）

回答請用繁體中文，簡潔為主。"""


def run_agent(question: str, max_turns: int = 8, verbose: bool = True) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = client.chat(
            model=MODEL, messages=messages, tools=TOOLS, think=True,
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for tc in response.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments
            if verbose:
                print(f'{CYAN}[第 {turn} 輪檢索] {name}({args}){RESET}')
            fn = AVAILABLE.get(name)
            if fn is None:
                result = f'錯誤：沒有名為 {name} 的工具'
            else:
                try:
                    result = fn(**args)
                except Exception as exc:              # noqa: BLE001
                    result = f'工具執行失敗：{type(exc).__name__}: {exc}'
            if verbose:
                first_line = str(result).splitlines()[0] if result else ''
                print(f'{GREY}  → {first_line[:100]}{RESET}')
            messages.append({
                'role': 'tool', 'tool_name': name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'


if __name__ == '__main__':
    question = ' '.join(sys.argv[1:]) or '資料庫遷移要注意什麼？'
    print(f'{CYAN}語料：{len(CHUNKS)} 個段落（純關鍵字檢索，零 embedding）{RESET}')
    print(f'{CYAN}問題：{question}{RESET}\n')
    print(run_agent(question))
