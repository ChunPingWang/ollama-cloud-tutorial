"""方案 A：混合式 agentic RAG——本地 embedding + 雲端生成。

Ollama Cloud 沒有 embedding，但你本機的 Ollama 有。這支程式同時用兩邊：

    向量化  → 本機 http://localhost:11434（embeddinggemma，768 維）
    生成    → 雲端 https://ollama.com（gpt-oss:120b）

這剛好是第 2 節兩種連線模式併用的實例。embedding 模型很小、CPU 也跑得動，
放本地不但省雲端 GPU 時間，語料也不用送出去——對有合規要求的場景很關鍵。

前置作業：
    ollama pull embeddinggemma

用法:
    python examples/13_rag_hybrid.py "告警一直響但都不是真的問題，該怎麼辦？"
    python examples/13_rag_hybrid.py --compare      # 比較向量檢索與關鍵字檢索
"""

import sys
from pathlib import Path

from ollama import Client

from _client import MODEL, get_client

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_cloud_rag = import_module('12_rag_cloud_only')      # 重用它的切塊與 BM25
CHUNKS, BM25_INDEX = _cloud_rag.CHUNKS, _cloud_rag.INDEX

client = get_client()                                 # 雲端：負責生成
local = Client(host='http://localhost:11434')         # 本機：負責向量化

EMBED_MODEL = 'embeddinggemma'
GREY, CYAN, GREEN, RESET = '\033[90m', '\033[36m', '\033[32m', '\033[0m'


# --------------------------------------------------------------------------
# 向量索引（建在本機）
# --------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    try:
        return local.embed(model=EMBED_MODEL, input=texts)['embeddings']
    except Exception as exc:                          # noqa: BLE001
        sys.exit(
            f'本機 embedding 失敗：{exc}\n'
            f'請確認 Ollama 有在跑，並且已經 `ollama pull {EMBED_MODEL}`。'
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


print(f'{GREY}正在用本機 {EMBED_MODEL} 建立向量索引…{RESET}')
VECTORS = _embed([c['text'] for c in CHUNKS])
print(f'{GREY}完成：{len(VECTORS)} 個向量，維度 {len(VECTORS[0])}{RESET}')


# --------------------------------------------------------------------------
# 檢索工具
# --------------------------------------------------------------------------

def search_handbook(query: str, top_k: int = 3) -> str:
    """在內部工程手冊中以語意相似度搜尋相關段落。

    這是語意檢索，用自然語言描述問題即可，不必猜文件裡用的確切字詞。

    Args:
        query: 要搜尋的問題或描述
        top_k: 要回傳幾個段落，預設 3，最多 5

    Returns:
        相關段落的內容，每段標明出處與標題
    """
    top_k = max(1, min(int(top_k), 5))
    query_vec = _embed([query])[0]
    scored = sorted(
        ((_cosine(query_vec, vec), i) for i, vec in enumerate(VECTORS)),
        reverse=True,
    )[:top_k]

    blocks = []
    for score, i in scored:
        chunk = CHUNKS[i]
        blocks.append(
            f'【{chunk["source"]} — {chunk["title"]}】(相似度 {score:.3f})\n'
            f'{chunk["text"]}'
        )
    return '\n\n---\n\n'.join(blocks)


TOOLS = [search_handbook]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

SYSTEM_PROMPT = """你是內部工程手冊的問答助理。

工作方式：
- 用 search_handbook 檢索，可以直接用自然語言描述問題
- 檢索結果不夠時，換個說法再搜一次
- **只根據檢索到的內容回答**。手冊沒寫的就說「手冊未涵蓋」，不要用常識補
- 回答時標明依據哪個段落

回答請用繁體中文，簡潔為主。"""


def run_agent(question: str, max_turns: int = 8) -> str:
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
            print(f'{CYAN}[第 {turn} 輪檢索] {name}({args}){RESET}')
            fn = AVAILABLE.get(name)
            result = (fn(**args) if fn else f'錯誤：沒有名為 {name} 的工具')
            print(f'{GREY}  → {str(result).splitlines()[0][:100]}{RESET}')
            messages.append({
                'role': 'tool', 'tool_name': name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'


# --------------------------------------------------------------------------
# 兩種檢索的對照
# --------------------------------------------------------------------------

def compare(queries: list[str]) -> None:
    """同一個問題，看向量檢索與 BM25 各自撈到什麼。"""
    for query in queries:
        print(f'\n{CYAN}▸ {query}{RESET}')

        query_vec = _embed([query])[0]
        best_vec = max(range(len(VECTORS)),
                       key=lambda i: _cosine(query_vec, VECTORS[i]))
        vec_score = _cosine(query_vec, VECTORS[best_vec])

        bm25_scores = BM25_INDEX.score(query)
        best_bm25 = max(range(len(bm25_scores)), key=lambda i: bm25_scores[i])

        print(f'  向量  → 〈{CHUNKS[best_vec]["title"]}〉(相似度 {vec_score:.3f})')
        if bm25_scores[best_bm25] > 0:
            print(f'  BM25 → 〈{CHUNKS[best_bm25]["title"]}〉'
                  f'(相關度 {bm25_scores[best_bm25]:.1f})')
        else:
            print(f'  BM25 → {GREY}完全沒撈到（沒有任何關鍵字命中）{RESET}')

        agree = best_vec == best_bm25 and bm25_scores[best_bm25] > 0
        print(f'  {GREEN if agree else GREY}'
              f'{"兩者一致" if agree else "兩者不一致 ← 值得注意"}{RESET}')


COMPARE_QUERIES = [
    '資料庫遷移要注意什麼？',                    # 用詞與文件一致，兩者都該命中
    '告警一直響但都不是真的問題，該怎麼辦？',      # 口語描述，文件用的是「閾值」
    '出事的時候先做什麼？',                      # 完全不同的說法，文件寫「止血」
]


if __name__ == '__main__':
    if '--compare' in sys.argv:
        compare(COMPARE_QUERIES)
        sys.exit(0)

    question = ' '.join(a for a in sys.argv[1:]) or '告警一直響但都不是真的問題，該怎麼辦？'
    print(f'{CYAN}問題：{question}{RESET}\n')
    print('\n' + '=' * 60)
    print(run_agent(question))
