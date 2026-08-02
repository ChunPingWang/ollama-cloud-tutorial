"""實際使用情境：分層路由，讓便宜模型擋掉大部分請求。

【這支教你什麼】GPU 時間計費下的降本：囉嗦比模型大小更花錢
【前置知識】01
【下一支】12_rag_cloud_only.py：讓 Agent 讀你的文件

Ollama Cloud 按 GPU 時間計費，所以省錢的方向跟 token 計費不同：
真正的變數是「模型花多少 GPU 時間」，而 thinking 模型即使回答一個詞
也要燒掉上百個推理 token。

實測（gpt-oss 系列關不掉 thinking，數字是三次的中位數）：

    分類任務「這句話是 bug / feature / question？」
      gemma4:31b     GPU 0.41s   輸出   2 tokens   ✅ 答對
      gpt-oss:120b   GPU 1.41s   輸出 101 tokens   ✅ 答對
      gpt-oss:20b    GPU 1.77s   輸出 100 tokens   ✅ 答對

同樣答對，成本差 3～4 倍。而且注意 20b 比 120b 還貴——
在 GPU 時間計費下，「參數少 = 便宜」這個直覺是錯的。

這支程式的做法：先用便宜的非 thinking 模型判斷難度，
只有真的需要推理的請求才升級到 thinking 模型。

用法:
    python examples/11_model_router.py
"""

import statistics
import time

from _client import get_client

client = get_client()

CHEAP = 'gemma4:31b'        # 非 thinking，回應短、GPU 時間低
SMART = 'gpt-oss:120b'      # thinking 模型，會推理，貴但強

GREY, CYAN, GREEN, YELLOW, RESET = (
    '\033[90m', '\033[36m', '\033[32m', '\033[33m', '\033[0m')

ROUTER_PROMPT = """判斷以下請求需要哪個等級的模型處理，只回一個字：

S = 簡單：分類、抽取、改寫、翻譯、格式轉換、事實查詢
C = 複雜：多步驟推理、數學計算、程式除錯、需要權衡的決策

請求：{task}

只回 S 或 C，不要任何其他文字。"""


def _gpu_seconds(response) -> float:
    """total_duration 是奈秒，且雲端有時會是 None。"""
    return (getattr(response, 'total_duration', None) or 0) / 1e9


# 這句話是整支程式最重要的一行。
# 實測「翻譯成英文」這題：gemma4 沒有這句要花 3.29s / 222 tokens（它會列出
# 五種說法還加註解），加了之後 0.32s / 7 tokens——十倍差距。
# GPU 時間計費下，囉嗦才是成本主因，模型大小反而是次要的。
TERSE = '直接給答案，不要解釋、不要列出多個選項、不要加註解。'


# 雲端 GPU 時間會偶發尖峰：同一題實測過 0.93s 也出現過 3.69s。
# 單次測量會得出完全相反的結論，所以量成本一定要取中位數。
REPS = 3


def ask(model: str, prompt: str, terse: bool = False,
        reps: int = 1) -> tuple[str, float]:
    messages = ([{'role': 'system', 'content': TERSE}] if terse else [])
    messages.append({'role': 'user', 'content': prompt})

    costs, answer = [], ''
    for _ in range(reps):
        response = client.chat(
            model=model, messages=messages, options={'temperature': 0},
        )
        costs.append(_gpu_seconds(response))
        answer = (response.message.content or '').strip()
    return answer, statistics.median(costs)


def route_and_answer(task: str) -> dict:
    """先用便宜模型判難度，再決定要不要升級。"""
    verdict, route_cost = ask(CHEAP, ROUTER_PROMPT.format(task=task))
    is_complex = verdict.upper().startswith('C')
    chosen = SMART if is_complex else CHEAP

    # 簡單任務才壓長度；複雜任務需要模型把推理寫出來，壓了反而會答錯
    answer, answer_cost = ask(chosen, task, terse=not is_complex, reps=REPS)
    return {
        'task': task,
        'verdict': 'C 複雜' if is_complex else 'S 簡單',
        'model': chosen,
        'answer': answer,
        'route_gpu_s': route_cost,
        'answer_gpu_s': answer_cost,
        'total_gpu_s': route_cost + answer_cost,
    }


TASKS = [
    '把這句話分類成 bug/feature/question：「匯出 CSV 中文變亂碼」',
    '把「今天天氣很好」翻譯成英文',
    '一列火車以時速 80 公里行駛 2.5 小時，再以時速 50 公里行駛 1.2 小時，'
    '接著折返以時速 100 公里開了 0.8 小時。它離出發點多遠？',
]


if __name__ == '__main__':
    results = []
    baseline_total = 0.0

    for task in TASKS:
        print(f'\n{CYAN}▸ {task[:50]}…{RESET}')
        wall_start = time.time()
        result = route_and_answer(task)
        result['wall_s'] = time.time() - wall_start
        results.append(result)

        colour = YELLOW if result['verdict'].startswith('C') else GREEN
        print(f'  {colour}路由判定：{result["verdict"]} → {result["model"]}{RESET}')
        print(f'  回答：{result["answer"][:80]}')
        print(f'  {GREY}GPU：路由 {result["route_gpu_s"]:.2f}s + '
              f'回答 {result["answer_gpu_s"]:.2f}s = {result["total_gpu_s"]:.2f}s{RESET}')

        # 對照組：沒有路由、全部都丟給大模型要花多少
        _, baseline_cost = ask(SMART, task, reps=REPS)
        result['baseline_gpu_s'] = baseline_cost
        baseline_total += baseline_cost
        print(f'  {GREY}對照：全用 {SMART} 需 {baseline_cost:.2f}s{RESET}')

    # 只看總計會誤導：複雜題本來就貴，它佔比多少完全取決於你的流量長相。
    # 拆開來看才知道這招對「你的」workload 划不划算。
    print(f'\n{"=" * 60}')
    simple = [r for r in results if r['verdict'].startswith('S')]
    complex_ = [r for r in results if r['verdict'].startswith('C')]

    for label, group in [('簡單任務', simple), ('複雜任務', complex_)]:
        if not group:
            continue
        routed = sum(r['total_gpu_s'] for r in group) / len(group)
        base = sum(r['baseline_gpu_s'] for r in group) / len(group)
        delta = (1 - routed / base) * 100 if base else 0
        verb = '省' if delta >= 0 else '多花'
        print(f'{label}（{len(group)} 題）平均 GPU：'
              f'路由 {routed:.2f}s vs 全用大模型 {base:.2f}s → {verb} {abs(delta):.0f}%')

    if simple:
        # 損益平衡：簡單題省下的量要能蓋過複雜題多付的路由開銷
        gain = sum(r['baseline_gpu_s'] - r['total_gpu_s'] for r in simple) / len(simple)
        overhead = (sum(r['route_gpu_s'] for r in complex_) / len(complex_)
                    if complex_ else 0)
        if gain > 0:
            breakeven = overhead / (gain + overhead) * 100
            print(f'\n損益平衡點：簡單請求需佔總流量 {breakeven:.0f}% 以上才划算')

    print(f'\n{GREY}註一：路由本身每題多一次便宜呼叫，這是固定成本。\n'
          f'註二：複雜題的 GPU 時間波動很大（實測同一題 6.1s～9.1s 都出現過），'
          f'單次測量不可盡信，要看多次的中位數。{RESET}')
