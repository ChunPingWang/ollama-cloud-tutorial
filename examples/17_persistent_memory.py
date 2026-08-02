"""跨 session 長期記憶：讓 Agent 記得上禮拜講過的事。

【這支教你什麼】把「記住」做成 Agent 的工具，以及記憶衝突怎麼處理
【前置知識】15_conversation_memory.py、12_rag_cloud_only.py
【下一支】18_deploy_server.py：把 Agent 包成服務

15 講的是「單一對話內」的記憶——程式一關就沒了。
這支講的是跨 session：今天說「我對花生過敏」，下週它還記得。

做法上有兩派：

  A. 全量存對話 + 每次檢索
     簡單，但雜訊多。「嗯」「好喔」也被存進去，檢索品質會被稀釋。

  B. 抽取事實再存                      ← 本檔採用
     只存值得記的結論。存什麼由「模型自己決定」——把 remember 做成
     一個工具，跟第 7 節的 Agent Loop 是同一個模式。

儲存用 SQLite（Python 內建，零依賴），檢索沿用 rag_common 的 BM25。
整套不需要向量資料庫，也不需要本地 Ollama。

用法:
    python examples/17_persistent_memory.py --demo     # 跑一段兩次 session 的示範
    python examples/17_persistent_memory.py            # 互動模式
    python examples/17_persistent_memory.py --dump     # 看看目前記得什麼
    python examples/17_persistent_memory.py --forget   # 清空
"""

import sqlite3
import sys
from pathlib import Path

from _client import MODEL, get_client
from rag_common import BM25

client = get_client()

DB_PATH = Path(__file__).parent / 'memory.db'
GREY, CYAN, GREEN, YELLOW, RESET = (
    '\033[90m', '\033[36m', '\033[32m', '\033[33m', '\033[0m')


# --------------------------------------------------------------------------
# 儲存層
# --------------------------------------------------------------------------

class MemoryStore:
    """長期記憶。用 SQLite 存，用 BM25 檢索。

    刻意不用向量資料庫：記憶通常只有幾百到幾千筆，BM25 就夠了，
    而且不需要 embedding——這樣純雲端部署也能用（見 RAG 那篇的限制）。
    """

    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY,
                subject    TEXT NOT NULL,      -- 主題，用來偵測衝突
                fact       TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                superseded INTEGER NOT NULL DEFAULT 0
            )''')
        self.conn.commit()

    def find_conflicts(self, subject: str, fact: str) -> list[str]:
        """找出「別的主題底下」可能講同一件事的記憶。

        這是實測撞到的問題：使用者說「我改用 Go 了」，模型把它存成
        主題「使用語言」，但舊資訊在主題「身分」底下（"…使用 Python"）。
        主題名稱不同，覆蓋機制就不會觸發，兩筆矛盾的記憶並存。

        單靠「同主題覆蓋」不夠，還要能看見跨主題的衝突。
        """
        rows = [r for r in self.active() if r[1] != subject]
        if not rows:
            return []
        index = BM25([f'{s} {f}' for _, s, f in rows])
        # 門檻靠實測抓，寧可多報幾筆讓模型自己判斷，也不要漏掉真衝突
        return [f'{rows[i][1]}：{rows[i][2]}'
                for i, score in index.rank(fact, top_k=3) if score > 1.0]

    def remember(self, subject: str, fact: str) -> str:
        """存一筆事實。同主題的舊記憶會被標記為過期，而不是刪掉。

        「不刪只標記」很重要：使用者改變偏好時，你會想知道
        他以前的偏好是什麼、什麼時候改的。直接 UPDATE 就丟失了這段歷史。
        """
        old = self.conn.execute(
            'SELECT fact FROM memories WHERE subject = ? AND superseded = 0',
            (subject,)).fetchall()

        self.conn.execute(
            'UPDATE memories SET superseded = 1 WHERE subject = ? AND superseded = 0',
            (subject,))
        self.conn.execute(
            'INSERT INTO memories (subject, fact) VALUES (?, ?)', (subject, fact))
        self.conn.commit()

        if old:
            return (f'已更新「{subject}」：{fact}'
                    f'（先前記錄為「{old[0][0]}」，已標記為過期）')
        return f'已記住「{subject}」：{fact}'

    def active(self) -> list[tuple[int, str, str]]:
        return self.conn.execute(
            'SELECT id, subject, fact FROM memories WHERE superseded = 0 '
            'ORDER BY created_at').fetchall()

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        """檢索與查詢相關的記憶。"""
        rows = self.active()
        if not rows:
            return []
        index = BM25([f'{s} {f}' for _, s, f in rows])
        hits = index.rank(query, top_k)
        return [f'{rows[i][1]}：{rows[i][2]}' for i, _ in hits]

    def all_facts(self) -> list[str]:
        return [f'{s}：{f}' for _, s, f in self.active()]

    def history(self, subject: str) -> list[tuple[str, str, int]]:
        return self.conn.execute(
            'SELECT fact, created_at, superseded FROM memories '
            'WHERE subject = ? ORDER BY created_at', (subject,)).fetchall()

    def forget_all(self) -> None:
        self.conn.execute('DELETE FROM memories')
        self.conn.commit()


STORE = MemoryStore()


# --------------------------------------------------------------------------
# 把「記住」做成 Agent 的工具
# --------------------------------------------------------------------------

def remember(subject: str, fact: str) -> str:
    """記住一件關於使用者的長期事實，之後的對話都會用得到。

    只記真正持久的資訊：偏好、限制、身分、長期目標。
    不要記一次性的問題內容，也不要記你自己剛才的回答。

    Args:
        subject: 這件事的主題，用簡短名詞，例如「飲食限制」「慣用語言」「職稱」。
                 同一個主題再次記錄時會覆蓋舊的，所以主題要取得一致。
        fact: 事實內容，一句話講完

    Returns:
        確認訊息；若偵測到別的主題底下有矛盾資訊，會一併告知
    """
    conflicts = STORE.find_conflicts(subject, fact)
    result = STORE.remember(subject, fact)
    if conflicts:
        # 把衝突丟回給模型，它會自己再呼叫一次 remember 去更新那個主題。
        # 這比在程式裡硬猜「哪兩筆算同一件事」可靠得多。
        listed = '、'.join(f'「{c}」' for c in conflicts)
        result += (f'\n⚠ 注意：其他主題底下有可能矛盾的舊記憶：{listed}。'
                   f'若確實已過時，請用那個主題名稱再呼叫一次 remember 更新它。')
    return result


def recall(query: str) -> str:
    """在長期記憶中搜尋與目前話題相關的事情。

    Args:
        query: 想查什麼，例如「飲食」「工作」

    Returns:
        相關的記憶，找不到就說明沒有
    """
    hits = STORE.recall(query)
    return '\n'.join(f'- {h}' for h in hits) if hits else '長期記憶中沒有相關資訊。'


TOOLS = [remember, recall]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}

SYSTEM_BASE = """你是一個會記住使用者的助理。

關於記憶：
- 使用者透露持久資訊（偏好、限制、身分、長期目標）時，用 remember 記下來
- 主題（subject）要取得一致，例如飲食相關一律用「飲食限制」，
  這樣同一件事更新時才會正確覆蓋而不是變成兩筆
- 不要記一次性的問題，也不要記你自己的回答
- 需要回想時用 recall

回答請用繁體中文，自然簡潔，不要每次都刻意宣告「我記住了」。"""


def build_system_prompt() -> str:
    """把已知記憶放進 system prompt。

    這是「開場注入」策略：記憶不多時直接全部放進去，模型不用先查就知道。
    記憶多到塞不下時，改成只在 system 放摘要，細節靠 recall 工具查。
    """
    facts = STORE.all_facts()
    if not facts:
        return SYSTEM_BASE
    listed = '\n'.join(f'- {f}' for f in facts)
    subjects = sorted({s for _, s, _ in STORE.active()})
    return (f'{SYSTEM_BASE}\n\n你已知關於這位使用者的事：\n{listed}\n\n'
            f'目前已使用的記憶主題：{"、".join(subjects)}。\n'
            f'更新既有資訊時**必須沿用上面已存在的主題名稱**，'
            f'不要另創新主題，否則會產生兩筆互相矛盾的記憶。')


class PersistentAgent:
    """每次啟動都從 SQLite 載入記憶，所以跨 session 有連續性。"""

    def __init__(self):
        self.messages = [{'role': 'system', 'content': build_system_prompt()}]

    def ask(self, question: str, max_turns: int = 6) -> str:
        self.messages.append({'role': 'user', 'content': question})

        for _ in range(max_turns):
            response = client.chat(model=MODEL, messages=self.messages, tools=TOOLS)
            self.messages.append(response.message)

            if not response.message.tool_calls:
                return response.message.content

            for tc in response.message.tool_calls:
                fn = AVAILABLE.get(tc.function.name)
                result = (fn(**tc.function.arguments) if fn
                          else f'錯誤：沒有 {tc.function.name} 這個工具')
                colour = YELLOW if tc.function.name == 'remember' else GREY
                print(f'{colour}  [{tc.function.name}] '
                      f'{tc.function.arguments} → {result}{RESET}')
                self.messages.append({
                    'role': 'tool', 'tool_name': tc.function.name,
                    'content': str(result),
                })

        return '（已達最大輪數上限）'


# --------------------------------------------------------------------------

SESSION_1 = [
    '你好，我叫 Rex，是後端工程師，平常寫 Python。',
    '對了我對花生過敏，蠻嚴重的。',
    '幫我想三個午餐選項。',
]
SESSION_2 = [
    '午餐吃什麼好？',                       # ← 應該自己避開花生，不用再問一次
    '我最近改用 Go 了，Python 比較少寫。',    # ← 應該覆蓋「平常寫 Python」
    '推薦一個適合我的開源專案來貢獻。',        # ← 應該用 Go 而不是 Python
]


def run_session(label: str, questions: list[str]) -> None:
    print(f'\n{CYAN}{"═" * 62}')
    print(f'{label}')
    print(f'{"═" * 62}{RESET}')
    agent = PersistentAgent()             # 全新的 agent，記憶從資料庫來
    known = STORE.all_facts()
    print(f'{GREY}啟動時已知 {len(known)} 筆記憶'
          f'{"：" + "、".join(known) if known else "（空白）"}{RESET}\n')

    for q in questions:
        print(f'{CYAN}你：{q}{RESET}')
        print(f'{GREEN}助理：{agent.ask(q)[:220]}{RESET}\n')


if __name__ == '__main__':
    if '--forget' in sys.argv:
        STORE.forget_all()
        print('記憶已清空。')
        sys.exit(0)

    if '--dump' in sys.argv:
        rows = STORE.active()
        if not rows:
            print('目前沒有任何記憶。')
        for _, subject, fact in rows:
            print(f'  {subject}：{fact}')
            hist = STORE.history(subject)
            for old_fact, when, superseded in hist:
                if superseded:
                    print(f'{GREY}      （舊）{old_fact}  於 {when}{RESET}')
        sys.exit(0)

    if '--demo' in sys.argv:
        STORE.forget_all()                # 從乾淨狀態開始，示範才看得清楚
        run_session('SESSION 1 — 第一次見面', SESSION_1)
        run_session('SESSION 2 — 幾天後，重新啟動程式', SESSION_2)

        print(f'{CYAN}{"═" * 62}\n最終記憶狀態\n{"═" * 62}{RESET}')
        for _, subject, fact in STORE.active():
            print(f'  {subject}：{fact}')
            for old_fact, when, superseded in STORE.history(subject):
                if superseded:
                    print(f'{GREY}      （已過期）{old_fact}{RESET}')
        sys.exit(0)

    agent = PersistentAgent()
    print(f'{GREY}已載入 {len(STORE.all_facts())} 筆長期記憶。'
          f'空白行結束，--dump 可查看記憶。{RESET}\n')
    try:
        while True:
            q = input(f'{CYAN}你：{RESET}').strip()
            if not q:
                break
            print(f'{GREEN}助理：{agent.ask(q)}{RESET}\n')
    except (KeyboardInterrupt, EOFError):
        print()
