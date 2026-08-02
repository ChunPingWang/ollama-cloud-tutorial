"""多輪對話與 context 管理：讓 Agent 記得住，又不會撐爆。

【這支教你什麼】對話記憶怎麼運作、context 為什麼會爆、三種裁切策略
【前置知識】03_agent_loop.py
【下一支】16_testing_agent.py：怎麼測試 Agent

前面的範例都是「問一次、答一次」。真的做產品時第一個撞到的問題是：
使用者問第二句「那東京呢？」，Agent 完全不知道「那」指的是什麼。

原因是**模型沒有記憶**。它每次只看到你送過去的 messages，
所謂「記得」完全是你自己把歷史一起送過去的結果。

但歷史不能無限長：
  - 超過模型的 context 上限會直接報錯或被截掉
  - Ollama Cloud 按 GPU 時間計費，歷史越長每一輪越貴
  - 太長的歷史會讓模型抓不到重點（大海撈針問題）

所以「記憶」的真正工作不是「存起來」，是**決定丟掉什麼**。

用法:
    python examples/15_conversation_memory.py            # 互動模式
    python examples/15_conversation_memory.py --demo     # 自動跑一段示範
"""

import sys

from _client import MODEL, get_client

client = get_client()

GREY, CYAN, GREEN, YELLOW, RESET = (
    '\033[90m', '\033[36m', '\033[32m', '\033[33m', '\033[0m')

# 真實情況下會設成模型 context 上限的一部分。這裡刻意設很小，
# 才能在短短幾輪對話裡就看到裁切真的發生。
MAX_HISTORY_TOKENS = 400


def get_temperature(city: str) -> str:
    """查詢指定城市的目前氣溫。

    Args:
        city: 城市名稱，例如 Taipei、Tokyo
    """
    return {'Taipei': '31°C', 'Tokyo': '18°C', 'London': '15°C',
            'Seoul': '22°C', 'Osaka': '20°C'}.get(city, '查無資料')


TOOLS = [get_temperature]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

SYSTEM_PROMPT = ('你是氣象助理。使用者可能會用「那邊」「剛才那個」等指代詞，'
                 '請根據對話歷史理解他指的是什麼。回答用繁體中文，簡潔為主。')


def estimate_tokens(messages: list) -> int:
    """粗估 token 數。

    正式做法是用模型回應裡的 prompt_eval_count，但那要先送出去才知道，
    沒辦法拿來「事前」決定要裁掉什麼。所以這裡用字元數估算：
    中文約 1 字 ≈ 1 token，英文約 4 字元 ≈ 1 token，取中間值 2。
    估得不準沒關係，我們只需要一個「該裁了」的訊號。
    """
    total = 0
    for m in messages:
        content = m.get('content') if isinstance(m, dict) else getattr(m, 'content', '')
        total += len(str(content or '')) // 2
    return total


# --------------------------------------------------------------------------
# 三種裁切策略
# --------------------------------------------------------------------------

def trim_sliding_window(messages: list, budget: int) -> list:
    """策略一：滑動視窗——保留 system + 最近的訊息。

    最簡單、最常用。缺點是舊資訊直接消失，使用者問「我一開始說什麼」會失憶。
    """
    system = [m for m in messages[:1] if _role(m) == 'system']
    rest = messages[len(system):]

    while rest and estimate_tokens(system + rest) > budget:
        rest.pop(0)
        # tool 訊息不能沒有對應的 assistant tool_calls，
        # 開頭若是孤兒 tool 訊息就一起丟掉
        while rest and _role(rest[0]) == 'tool':
            rest.pop(0)
    return system + rest


def trim_with_summary(messages: list, budget: int) -> list:
    """策略二：摘要壓縮——把舊訊息交給模型濃縮成一段。

    保留了舊資訊的重點，代價是多一次模型呼叫。
    長對話（客服、助理）值得，短任務不值得。
    """
    system = [m for m in messages[:1] if _role(m) == 'system']
    rest = messages[len(system):]
    if estimate_tokens(system + rest) <= budget:
        return messages

    keep = rest[-4:]                       # 最近兩輪一定要原樣保留
    old = rest[:-4]
    if not old:
        return trim_sliding_window(messages, budget)

    transcript = '\n'.join(
        f'{_role(m)}: {_content(m)}' for m in old if _content(m)
    )
    summary = client.chat(
        model=MODEL, options={'temperature': 0},
        messages=[{'role': 'user',
                   'content': f'把以下對話濃縮成三句話以內的重點，'
                              f'保留使用者提過的具體事實（地點、數字、偏好）：\n\n'
                              f'{transcript}'}],
    ).message.content

    print(f'{YELLOW}  [壓縮] 把 {len(old)} 則舊訊息換成一段摘要{RESET}')
    return system + [{'role': 'user', 'content': f'（先前對話摘要）{summary}'}] + keep


PLACEHOLDER = '（舊工具結果已省略）'


def trim_drop_tool_results(messages: list, budget: int) -> list:
    """策略三：優先丟工具結果——先砍最肥的部分。

    Agent 的 context 通常是被工具輸出撐爆的（想想 read_file 讀進一個大檔），
    對話本身其實很短。先把肥的工具結果換成佔位符，往往就夠了，
    而且對話的來龍去脈完全保留。

    ⚠ 這個策略我第一版寫錯了，錯法很有代表性：
      我無條件把所有舊工具結果換成佔位符，但佔位符比 '31°C' 還長，
      對短結果反而「越裁越肥」。省不下來就靜默退回滑動視窗，
      早期對話被整段砍掉——Agent 於是答錯了「哪個城市最冷」。
    修法有兩個：只換真的比佔位符長的，以及退場時要出聲。
    """
    if estimate_tokens(messages) <= budget:
        return messages

    trimmed = []
    for i, m in enumerate(messages):
        is_old_tool = _role(m) == 'tool' and i < len(messages) - 4
        # 只有「換掉之後真的比較短」才換，否則這步是負優化
        if is_old_tool and len(_content(m)) > len(PLACEHOLDER):
            trimmed.append({**_as_dict(m), 'content': PLACEHOLDER})
        else:
            trimmed.append(m)

    if estimate_tokens(trimmed) > budget:
        # 這裡一定要出聲。靜默退場會讓你以為「記憶好好的」，
        # 直到使用者問了一個需要回顧早期資訊的問題才發現答錯。
        print(f'{YELLOW}  [警告] 光丟工具結果不夠，退回滑動視窗，'
              f'早期對話將遺失{RESET}')
        return trim_sliding_window(trimmed, budget)
    return trimmed


STRATEGIES = {
    'window': ('滑動視窗', trim_sliding_window),
    'summary': ('摘要壓縮', trim_with_summary),
    'tools': ('優先丟工具結果', trim_drop_tool_results),
}


def _role(m) -> str:
    return m.get('role') if isinstance(m, dict) else getattr(m, 'role', '')


def _content(m) -> str:
    c = m.get('content') if isinstance(m, dict) else getattr(m, 'content', '')
    return str(c or '')


def _as_dict(m) -> dict:
    return m if isinstance(m, dict) else {'role': _role(m), 'content': _content(m)}


# --------------------------------------------------------------------------
# 帶記憶的 Agent
# --------------------------------------------------------------------------

class ConversationalAgent:
    """會記得前面講過什麼的 Agent。

    關鍵只有一件事：messages 這個 list 跨輪保留，而不是每次重新建立。
    """

    def __init__(self, strategy: str = 'tools'):
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        self.label, self.trim = STRATEGIES[strategy]

    def ask(self, question: str, max_turns: int = 6) -> str:
        self.messages.append({'role': 'user', 'content': question})

        before = estimate_tokens(self.messages)
        self.messages = self.trim(self.messages, MAX_HISTORY_TOKENS)
        after = estimate_tokens(self.messages)
        if after < before:
            print(f'{YELLOW}  [裁切:{self.label}] {before} → {after} tokens{RESET}')

        for _ in range(max_turns):
            response = client.chat(model=MODEL, messages=self.messages, tools=TOOLS)
            self.messages.append(response.message)

            if not response.message.tool_calls:
                return response.message.content

            for tc in response.message.tool_calls:
                fn = AVAILABLE.get(tc.function.name)
                result = (fn(**tc.function.arguments) if fn
                          else f'錯誤：沒有 {tc.function.name} 這個工具')
                print(f'{GREY}  [工具] {tc.function.name}'
                      f'({tc.function.arguments}) → {result}{RESET}')
                self.messages.append({
                    'role': 'tool', 'tool_name': tc.function.name,
                    'content': str(result),
                })

        return '（已達最大輪數上限）'


DEMO = [
    '台北現在幾度？',
    '那東京呢？',                    # ← 沒有記憶的話，這句會失敗
    '首爾跟大阪呢？',
    '剛才問過的城市裡，哪一個最冷？',   # ← 需要記得前面全部的答案
]


if __name__ == '__main__':
    strategy = 'tools'
    for key in STRATEGIES:
        if f'--{key}' in sys.argv:
            strategy = key

    agent = ConversationalAgent(strategy)
    print(f'{CYAN}裁切策略：{agent.label}　'
          f'（歷史上限 {MAX_HISTORY_TOKENS} tokens，刻意設很小）{RESET}')
    print(f'{GREY}可用 --window / --summary / --tools 切換策略{RESET}\n')

    if '--demo' in sys.argv:
        for question in DEMO:
            print(f'{CYAN}你：{question}{RESET}')
            print(f'{GREEN}助理：{agent.ask(question)}{RESET}\n')
        print(f'{GREY}對話結束，歷史共 {len(agent.messages)} 則訊息、'
              f'約 {estimate_tokens(agent.messages)} tokens{RESET}')
        sys.exit(0)

    print(f'{GREY}輸入問題，Ctrl-C 或空白行結束。試試「台北幾度？」→「那東京呢？」{RESET}\n')
    try:
        while True:
            question = input(f'{CYAN}你：{RESET}').strip()
            if not question:
                break
            print(f'{GREEN}助理：{agent.ask(question)}{RESET}\n')
    except (KeyboardInterrupt, EOFError):
        print()
