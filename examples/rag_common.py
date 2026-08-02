"""RAG 範例共用的切塊與 BM25 檢索。

【這支教你什麼】切塊策略與 BM25 的實作（12、13、14 都從這裡取）
【重點】整個檢索層零依賴，純手寫不到 120 行——刻意的，
        目的是讓你看清楚裡面在做什麼，而不是 pip install 一個黑盒子

12（純雲端）與 13（混合式）都從這裡取，14（評估）也用它跑對照。
抽出來是為了讓 13 不用去 import 一個數字開頭的模組——那需要 importlib，
對初學者是沒必要的噪音。
"""

import math
import re
from collections import Counter
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / 'corpus'


def load_chunks(corpus_dir: Path | None = None) -> list[dict]:
    """依 Markdown 標題切塊。

    標題切塊對技術文件效果很好——每個 ## 段落本來就是一個語意單位，
    比固定字元數切塊少很多「一句話被腰斬」的狀況。
    """
    chunks = []
    for path in sorted((corpus_dir or CORPUS_DIR).glob('*.md')):
        text = path.read_text(encoding='utf-8')
        for part in re.split(r'\n(?=## )', text):
            part = part.strip()
            if len(part) < 40:              # 太短的多半是文件標題，沒有檢索價值
                continue
            title = part.splitlines()[0].lstrip('# ').strip()
            chunks.append({'source': path.name, 'title': title, 'text': part})
    return chunks


def tokenize(text: str) -> list[str]:
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
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.freqs = [Counter(t) for t in self.doc_tokens]

        total = len(docs)
        appears = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                appears[token] += 1
        # 加 0.5 平滑，避免出現在所有文件的詞得到負分
        self.idf = {
            token: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for token, count in appears.items()
        }

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
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

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """回傳 (chunk 索引, 分數)，只含分數大於 0 的，最多 top_k 個。"""
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:top_k] if scores[i] > 0]


def format_hits(chunks: list[dict], hits: list[tuple[int, float]],
                score_label: str = '相關度') -> str:
    """把檢索結果排版成給模型看的文字。"""
    if not hits:
        return ''
    blocks = []
    for i, score in hits:
        chunk = chunks[i]
        blocks.append(
            f'【{chunk["source"]} — {chunk["title"]}】({score_label} {score:.3f})\n'
            f'{chunk["text"]}'
        )
    return '\n\n---\n\n'.join(blocks)
