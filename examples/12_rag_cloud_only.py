"""方案 B：純雲端 agentic RAG——零 embedding。

Ollama Cloud 的 hosted API 沒有可用的 embedding（實測 /api/embed 回 401，
/api/embeddings 與 /v1/embeddings 直接 404，模型清單裡也沒有 embedding 模型）。
所以要做純雲端 RAG，就不能靠向量。

這支程式改用「關鍵字檢索 + LLM 重排」，而且不是固定管線，是把檢索做成
Agent 的工具——模型自己決定要不要查、查幾次、換什麼關鍵字。
這其實就是第 7 節 Codebase Agent 的模式，只是換成文件語料。

用法:
    python examples/12_rag_cloud_only.py "資料庫遷移要注意什麼？"
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

from _client import MODEL, get_client

client = get_client()

CORPUS_DIR = Path(__file__).parent / 'corpus'
GREY, CYAN, RESET = '\033[90m', '\033[36m', '\033[0m'


# --------------------------------------------------------------------------
# 切塊與 BM25 索引
# --------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    """依 Markdown 標題切塊。

    標題切塊對技術文件效果很好——每個 ## 段落本來就是一個語意單位，
    比固定字元數切塊少很多「一句話被腰斬」的狀況。
    """
    chunks = []
    for path in sorted(CORPUS_DIR.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        # 用 ## 標題切開，保留標題當作 chunk 的一部分
        parts = re.split(r'\n(?=## )', text)
        for part in parts:
            part = part.strip()
            if len(part) < 40:              # 太短的多半是文件標題，沒有檢索價值
                continue
            title = part.splitlines()[0].lstrip('# ').strip()
            chunks.append({'source': path.name, 'title': title, 'text': part})
    return chunks


def _tokenize(text: str) -> list[str]:
    """中英混合的粗略斷詞。

    中文沒有空格，這裡用 bigram（相鄰兩字）當作詞——土砲但夠用，
    而且不需要額外的斷詞套件。英文與數字照原樣切。
    """
    lowered = text.lower()
    latin = re.findall(r'[a-z0-9_]+', lowered)
    han = re.findall(r'[一-鿿]', lowered)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return latin + han + bigrams


class BM25:
    """最小可用的 BM25，不依賴任何套件。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [_tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.freqs = [Counter(t) for t in self.doc_tokens]

        self.idf = {}
        total = len(docs)
        appears = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                appears[token] += 1
        for token, count in appears.items():
            # 加 0.5 平滑，避免出現在所有文件的詞得到負分
            self.idf[token] = math.log(1 + (total - count + 0.5) / (count + 0.5))

    def score(self, query: str) -> list[float]:
        query_tokens = _tokenize(query)
        scores = []
        for i, freq in enumerate(self.freqs):
            score = 0.0
            for token in query_tokens:
                if token not in freq:
                    continue
                tf = freq[token]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                )
                score += self.idf.get(token, 0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


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
    top_k = max(1, min(int(top_k), 5))
    scores = INDEX.score(query)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = [i for i in ranked[:top_k] if scores[i] > 0]
    if not hits:
        return f'找不到與「{query}」相關的段落，請換個關鍵字再試一次。'

    blocks = []
    for i in hits:
        chunk = CHUNKS[i]
        blocks.append(
            f'【{chunk["source"]} — {chunk["title"]}】(相關度 {scores[i]:.1f})\n'
            f'{chunk["text"]}'
        )
    return '\n\n---\n\n'.join(blocks)


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
            if fn is None:
                result = f'錯誤：沒有名為 {name} 的工具'
            else:
                try:
                    result = fn(**args)
                except Exception as exc:              # noqa: BLE001
                    result = f'工具執行失敗：{type(exc).__name__}: {exc}'
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
    print('\n' + '=' * 60)
    print(run_agent(question))
