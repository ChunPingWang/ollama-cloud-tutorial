"""自動化評估 RAG 的正確率。

【這支教你什麼】怎麼用數字回答「我的 RAG 到底準不準」，而不是憑感覺
【前置知識】12 與 13
【核心觀念】RAG 有兩個獨立的失敗點，要分開量：
             檢索錯了 → 模型再強也答不對
             檢索對了但答錯 → 是生成或 prompt 的問題

評估分兩層：

  第一層 檢索指標（不花 GPU 時間，秒級跑完）
      Recall@k  正確段落有沒有出現在前 k 名
      MRR       正確段落排第幾名（排越前面分數越高）
      這層可以在每次改切塊策略、換 embedding 模型後立刻重跑。

  第二層 端到端指標（要跑模型，較慢）
      正確率      答案有沒有包含標註的關鍵事實
      拒答正確率  語料沒涵蓋時有沒有正確說「手冊未涵蓋」（抓幻覺）
      忠實度      LLM-as-judge：答案是否只根據檢索內容，沒有自行發揮

用法:
    python examples/14_rag_eval.py                 # 只跑第一層，不花 GPU 時間
    python examples/14_rag_eval.py --end-to-end    # 兩層都跑（會呼叫雲端模型）
    python examples/14_rag_eval.py --end-to-end --limit 5
"""

import json
import sys
from pathlib import Path

from _client import MODEL, get_client
from rag_common import BM25, load_chunks

EVAL_FILE = Path(__file__).parent / 'corpus' / 'eval_set.json'

GREY, CYAN, GREEN, RED, YELLOW, BOLD, RESET = (
    '\033[90m', '\033[36m', '\033[32m', '\033[31m', '\033[33m', '\033[1m', '\033[0m')

CHUNKS = load_chunks()
TITLES = [c['title'] for c in CHUNKS]
BM25_INDEX = BM25([c['text'] for c in CHUNKS])


def load_cases() -> list[dict]:
    data = json.loads(EVAL_FILE.read_text(encoding='utf-8'))
    cases = data['cases']

    # 標註資料本身也要驗——標錯的 expected_chunk 會讓所有指標失去意義
    bad = [c['id'] for c in cases
           if c.get('expected_chunk') and c['expected_chunk'] not in TITLES]
    if bad:
        sys.exit(f'標註資料有誤，這些 expected_chunk 在語料中不存在：{bad}\n'
                 f'語料現有段落：{TITLES}')
    return cases


# --------------------------------------------------------------------------
# 第一層：檢索指標
# --------------------------------------------------------------------------

def _bm25_rank(query: str, top_k: int) -> list[int]:
    return [i for i, _ in BM25_INDEX.rank(query, top_k)]


def _vector_rank_factory():
    """向量檢索需要本機 Ollama，沒有就跳過這個 retriever 而不是整個中斷。"""
    try:
        from ollama import Client
        local = Client(host='http://localhost:11434')
        vectors = local.embed(
            model='embeddinggemma', input=[c['text'] for c in CHUNKS],
        )['embeddings']
    except Exception as exc:                          # noqa: BLE001
        print(f'{YELLOW}略過向量檢索：{str(exc)[:80]}{RESET}')
        print(f'{GREY}（需要本機 Ollama 並 `ollama pull embeddinggemma`）{RESET}\n')
        return None

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def rank(query: str, top_k: int) -> list[int]:
        qv = local.embed(model='embeddinggemma', input=[query])['embeddings'][0]
        scored = sorted(((cosine(qv, v), i) for i, v in enumerate(vectors)),
                        reverse=True)
        return [i for _, i in scored[:top_k]]

    return rank


def eval_retrieval(cases: list[dict], rank_fn, top_k: int = 3) -> dict:
    """算 Recall@1、Recall@k、MRR。只看有正確答案的題目。"""
    answerable = [c for c in cases if c.get('expected_chunk')]
    hit1 = hitk = 0
    reciprocal_ranks = []
    per_phrasing = {}

    for case in answerable:
        ranked_titles = [TITLES[i] for i in rank_fn(case['question'], top_k)]
        expected = case['expected_chunk']

        rank = ranked_titles.index(expected) + 1 if expected in ranked_titles else 0
        hit1 += int(rank == 1)
        hitk += int(rank > 0)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        bucket = per_phrasing.setdefault(case.get('phrasing', '?'),
                                         {'hit': 0, 'total': 0})
        bucket['total'] += 1
        bucket['hit'] += int(rank == 1)

    n = len(answerable) or 1
    return {
        'n': len(answerable),
        'recall@1': hit1 / n,
        f'recall@{top_k}': hitk / n,
        'mrr': sum(reciprocal_ranks) / n,
        'per_phrasing': per_phrasing,
    }


# --------------------------------------------------------------------------
# 第二層：端到端指標
# --------------------------------------------------------------------------

REFUSAL_MARKERS = ['未涵蓋', '沒有涵蓋', '找不到', '未提及', '沒有提到', '無法回答']


def _fact_present(fact, answer: str) -> bool:
    """判斷一項關鍵事實有沒有出現在答案裡。

    fact 可以是字串，也可以是「可接受寫法」的陣列。後者很重要：
    我第一版只用字串比對，結果模型答「15 分鐘」卻被判成缺少「十五分鐘」——
    RAG 是對的，是評估工具製造了假陰性。中文數字與阿拉伯數字的差異
    是這類評估最常見的坑。
    """
    if isinstance(fact, (list, tuple)):
        return any(variant in answer for variant in fact)
    return fact in answer


def _label(fact) -> str:
    return f'{fact[0]}(或其變體)' if isinstance(fact, (list, tuple)) else str(fact)

JUDGE_PROMPT = """你是嚴格的評分員。判斷「答案」是否**完全**能由「檢索內容」支持。

檢索內容：
{context}

答案：
{answer}

只要答案裡有任何一項事實在檢索內容中找不到，就算不忠實。
只回一個字：Y（完全支持）或 N（有無法支持的內容）。"""


def judge_grounded(client, context: str, answer: str) -> bool:
    """LLM-as-judge 忠實度。用 tool calling 逼出結構化結果，
    因為雲端不強制 format（見第 8 節）。"""
    verdict = client.chat(
        model=MODEL, options={'temperature': 0},
        messages=[{'role': 'user',
                   'content': JUDGE_PROMPT.format(context=context[:4000],
                                                  answer=answer)}],
    ).message.content or ''
    return verdict.strip().upper().startswith('Y')


def eval_end_to_end(cases: list[dict], limit: int | None = None) -> dict:
    import importlib
    rag = importlib.import_module('12_rag_cloud_only')
    client = get_client()

    subset = cases[:limit] if limit else cases
    correct = refused_ok = grounded = 0
    answerable_n = unanswerable_n = 0
    failures = []

    for case in subset:
        answer = rag.run_agent(case['question'], verbose=False) or ''

        if case.get('unanswerable'):
            unanswerable_n += 1
            ok = any(marker in answer for marker in REFUSAL_MARKERS)
            refused_ok += int(ok)
            status = f'{GREEN}正確拒答{RESET}' if ok else f'{RED}幻覺！應拒答卻回答了{RESET}'
            if not ok:
                failures.append((case['id'], '應拒答卻回答', answer[:80]))
        else:
            answerable_n += 1
            missing = [_label(fact) for fact in case['answer_must_contain']
                       if not _fact_present(fact, answer)]
            ok = not missing
            correct += int(ok)

            context = rag.search_handbook(case['question'], top_k=3)
            is_grounded = judge_grounded(client, context, answer)
            grounded += int(is_grounded)

            if not ok:
                status = f'{RED}缺少關鍵事實 {missing}{RESET}'
                failures.append((case['id'], f'缺少 {missing}', answer[:80]))
            elif not is_grounded:
                # 答對但講了檢索內容裡沒有的東西——這是最容易被忽略的失敗
                status = f'{YELLOW}正確但不忠實（有自行發揮的內容）{RESET}'
                failures.append((case['id'], '答案超出檢索內容', answer[:100]))
            else:
                status = f'{GREEN}正確{RESET}'

        print(f'  {case["id"]:16s} {status}')

    return {
        'answerable_n': answerable_n,
        'unanswerable_n': unanswerable_n,
        'correctness': correct / (answerable_n or 1),
        'refusal_accuracy': refused_ok / (unanswerable_n or 1),
        'groundedness': grounded / (answerable_n or 1),
        'failures': failures,
    }


# --------------------------------------------------------------------------

def _print_retrieval(label: str, metrics: dict) -> None:
    print(f'{BOLD}{label}{RESET}（{metrics["n"]} 題）')
    print(f'  Recall@1 {metrics["recall@1"]:.0%}   '
          f'Recall@3 {metrics["recall@3"]:.0%}   '
          f'MRR {metrics["mrr"]:.3f}')
    for phrasing, bucket in sorted(metrics['per_phrasing'].items()):
        rate = bucket['hit'] / bucket['total']
        colour = GREEN if rate >= 0.8 else (YELLOW if rate >= 0.5 else RED)
        print(f'    {phrasing:14s} Recall@1 {colour}{rate:.0%}{RESET} '
              f'({bucket["hit"]}/{bucket["total"]})')


if __name__ == '__main__':
    cases = load_cases()
    print(f'{CYAN}語料 {len(CHUNKS)} 段，評估題目 {len(cases)} 題'
          f'（其中 {sum(1 for c in cases if c.get("unanswerable"))} 題語料未涵蓋）{RESET}\n')

    print(f'{BOLD}═══ 第一層：檢索指標 ═══{RESET}\n')
    _print_retrieval('BM25 關鍵字檢索', eval_retrieval(cases, _bm25_rank))

    vector_rank = _vector_rank_factory()
    if vector_rank:
        print()
        _print_retrieval('向量語意檢索', eval_retrieval(cases, vector_rank))

    if '--end-to-end' not in sys.argv:
        print(f'\n{GREY}加上 --end-to-end 可再跑生成層指標（會呼叫雲端模型）{RESET}')
        sys.exit(0)

    limit = None
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])

    print(f'\n{BOLD}═══ 第二層：端到端指標（BM25 版 Agent）═══{RESET}\n')
    result = eval_end_to_end(cases, limit)
    print(f'\n  答案正確率   {result["correctness"]:.0%} '
          f'（{result["answerable_n"]} 題可答）')
    print(f'  拒答正確率   {result["refusal_accuracy"]:.0%} '
          f'（{result["unanswerable_n"]} 題應拒答）')
    print(f'  忠實度       {result["groundedness"]:.0%} '
          f'（LLM-as-judge，答案是否只根據檢索內容）')

    if result['failures']:
        print(f'\n{BOLD}失敗案例（這才是你該去看的）{RESET}')
        for case_id, reason, snippet in result['failures']:
            print(f'  {RED}✗{RESET} {case_id}: {reason}')
            print(f'    {GREY}{snippet}…{RESET}')
