"""方案 A：混合式 agentic RAG——本地 embedding + 雲端生成。

【這支教你什麼】兩種連線模式併用；語意檢索與關鍵字檢索的差別在哪裡
【前置知識】12_rag_cloud_only.py
【下一支】14_rag_eval.py：用自動化方式量出哪種檢索比較準

Ollama Cloud 沒有 embedding，但你本機的 Ollama 有。這支程式同時用兩邊：

    向量化  → 本機 http://localhost:11434（embeddinggemma，768 維）
    生成    → 雲端 https://ollama.com（gpt-oss:120b）

embedding 模型很小、CPU 也跑得動，放本地不但省雲端 GPU 時間，
語料也不用送出去——對有合規要求的場景，這是優點而不是妥協。

前置作業：
    ollama pull embeddinggemma

用法:
    python examples/13_rag_hybrid.py "告警一直響但都不是真的問題，該怎麼辦？"
    python examples/13_rag_hybrid.py --compare      # 比較向量檢索與關鍵字檢索
"""

import sys

from ollama import Client

from _client import MODEL, get_client
from rag_common import BM25, format_hits, load_chunks

client = get_client()                                 # 雲端：負責生成
local = Client(host='http://localhost:11434')         # 本機：負責向量化

EMBED_MODEL = 'embeddinggemma'
GREY, CYAN, GREEN, RESET = '\033[90m', '\033[36m', '\033[32m', '\033[0m'

CHUNKS = load_chunks()
BM25_INDEX = BM25([c['text'] for c in CHUNKS])


# --------------------------------------------------------------------------
# 向量索引（建在本機）
# --------------------------------------------------------------------------

def embed(texts: list[str]) -> list[list[float]]:
    try:
        return local.embed(model=EMBED_MODEL, input=texts)['embeddings']
    except Exception as exc:                          # noqa: BLE001
        sys.exit(
            f'本機 embedding 失敗：{exc}\n'
            f'請確認 Ollama 有在跑，並且已經 `ollama pull {EMBED_MODEL}`。\n'
            f'（若收到 501 not support embeddings，代表你指定的是聊天模型，'
            f'不是 embedding 模型）'
        )


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def build_vectors() -> list[list[float]]:
    print(f'{GREY}正在用本機 {EMBED_MODEL} 建立向量索引…{RESET}')
    vectors = embed([c['text'] for c in CHUNKS])
    print(f'{GREY}完成：{len(vectors)} 個向量，維度 {len(vectors[0])}{RESET}')
    return vectors


def vector_rank(query: str, vectors: list[list[float]],
                top_k: int) -> list[tuple[int, float]]:
    query_vec = embed([query])[0]
    scored = sorted(
        ((cosine(query_vec, vec), i) for i, vec in enumerate(vectors)),
        reverse=True,
    )[:top_k]
    return [(i, score) for score, i in scored]


VECTORS = build_vectors()


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
    hits = vector_rank(query, VECTORS, max(1, min(int(top_k), 5)))
    return format_hits(CHUNKS, hits, score_label='相似度')


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

COMPARE_QUERIES = [
    '資料庫遷移要注意什麼？',                    # 用詞與文件一致，兩者都該命中
    '不小心把密碼寫進程式碼了',                  # 文件用「憑證」，仍有部分字面重疊
    '怎麼避免改壞正式環境',                      # 口語描述，字面幾乎不重疊
    '同事寫的東西太大包看不完',                  # 文件寫「PR 超過四百行就該拆」
    '新人第一天該看什麼',                        # 語料未涵蓋，看兩者怎麼表現
]


def compare(queries: list[str]) -> None:
    """同一個問題，看向量檢索與 BM25 各自撈到什麼。"""
    for query in queries:
        print(f'\n{CYAN}▸ {query}{RESET}')

        vec_hits = vector_rank(query, VECTORS, 1)
        bm25_hits = BM25_INDEX.rank(query, 1)

        vec_i, vec_score = vec_hits[0]
        print(f'  向量  → 〈{CHUNKS[vec_i]["title"]}〉(相似度 {vec_score:.3f})')
        if bm25_hits:
            bm_i, bm_score = bm25_hits[0]
            print(f'  BM25 → 〈{CHUNKS[bm_i]["title"]}〉(相關度 {bm_score:.1f})')
        else:
            bm_i = None
            print(f'  BM25 → {GREY}完全沒撈到（沒有任何關鍵字命中）{RESET}')

        agree = bm_i is not None and vec_i == bm_i
        print(f'  {GREEN if agree else GREY}'
              f'{"兩者一致" if agree else "兩者不一致 ← 值得注意"}{RESET}')


if __name__ == '__main__':
    if '--compare' in sys.argv:
        compare(COMPARE_QUERIES)
        sys.exit(0)

    question = ' '.join(sys.argv[1:]) or '告警一直響但都不是真的問題，該怎麼辦？'
    print(f'{CYAN}問題：{question}{RESET}\n')
    print(run_agent(question))
