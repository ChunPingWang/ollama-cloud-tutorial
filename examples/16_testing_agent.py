"""怎麼測試 Agent：大部分測試根本不用呼叫模型。

【這支教你什麼】用假的 client 測 Agent Loop，不花 GPU 時間也不看運氣
【前置知識】03_agent_loop.py、04_codebase_agent.py
【下一支】—（核心範例結束）

寫 Agent 最常見的說法是「這東西沒辦法測，模型輸出不固定」。
這句話只對了一半。Agent 的組成是：

    工具（純函式）        ← 完全可測，而且大部分 bug 在這裡
    Agent Loop（控制流）  ← 完全可測，把 client 換成假的就好
    模型的判斷             ← 不可測，只能用評估集看趨勢（見 14_rag_eval.py）

前兩層是你自己的程式碼，佔了 bug 的絕大多數。把它們測起來，
你就能安心重構，而不是每次改動都要人工點一遍。

用法:
    python examples/16_testing_agent.py          # 跑全部測試
    python -m unittest discover -s examples -p "16_*.py" -v
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# --------------------------------------------------------------------------
# 假的 client：照劇本回應，完全不碰網路
# --------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, content='', tool_calls=None, thinking=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.thinking = thinking


class FakeToolCall:
    """模仿 ollama SDK 的 tool_call 形狀：tc.function.name / .arguments"""

    def __init__(self, name, arguments):
        self.function = type('F', (), {'name': name, 'arguments': arguments})()


class FakeResponse:
    def __init__(self, message):
        self.message = message
        self.total_duration = 1_000_000_000
        self.prompt_eval_count = 100
        self.eval_count = 50


class FakeClient:
    """照著劇本一句一句回，並記錄每次收到的 messages。

    這是整個測試策略的核心：Agent Loop 只透過 client.chat 與外界互動，
    把它換掉，控制流就變成一段純粹的、可決定的程式。
    """

    def __init__(self, script: list[FakeMessage]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError('Agent 呼叫模型的次數超過劇本預期'
                                 '——可能是終止條件寫錯了')
        return FakeResponse(self.script.pop(0))


# --------------------------------------------------------------------------
# 第一層：測工具（純函式，最容易也最該測）
# --------------------------------------------------------------------------

class TestToolSafety(unittest.TestCase):
    """工具的安全邊界。這類測試最有價值——模型真的會送奇怪的參數進來。"""

    def setUp(self):
        import importlib
        self.agent = importlib.import_module('04_codebase_agent')

    def test_路徑逃逸會被擋下來(self):
        for evil in ['../../etc/passwd', '/etc/passwd', '../../../root/.ssh/id_rsa']:
            with self.subTest(path=evil):
                with self.assertRaises(ValueError):
                    self.agent._safe_path(evil)

    def test_正常路徑可以通過(self):
        self.assertTrue(str(self.agent._safe_path('.')).startswith(str(self.agent.ROOT)))

    def test_讀不存在的檔案回錯誤訊息而不是丟例外(self):
        # 這點很重要：模型讀得懂錯誤訊息，會自己換個做法重試；
        # 丟例外則會讓整個 Agent 中斷。
        result = self.agent.read_file('絕對不存在的檔案.txt')
        self.assertIn('不存在', result)

    def test_搜尋不到時給的訊息要引導模型重試(self):
        # 字串要在執行期組出來，不能直接寫在原始碼裡——
        # 否則 search_code 會掃到這個測試檔案本身而「找到」它。
        # 我第一版就踩了這個自我指涉的坑，測試紅了才發現。
        needle = 'zzq' + 'absent' + 'marker' + 'zzq'
        result = self.agent.search_code(needle)
        self.assertIn('找不到', result)


class TestRetrieval(unittest.TestCase):
    """檢索邏輯是純函式，完全可測——不用等模型也不用聯網。"""

    def setUp(self):
        from rag_common import BM25, load_chunks
        self.chunks = load_chunks()
        self.index = BM25([c['text'] for c in self.chunks])

    def test_切塊有切出東西(self):
        self.assertGreater(len(self.chunks), 1)
        self.assertTrue(all(c['title'] for c in self.chunks))

    def test_關鍵字能命中正確段落(self):
        hits = self.index.rank('資料庫遷移 回滾', top_k=1)
        self.assertTrue(hits)
        self.assertEqual(self.chunks[hits[0][0]]['title'], '資料庫遷移')

    def test_完全沒有字面重疊的查詢應該撈不到(self):
        self.assertEqual(self.index.rank('qqzz wwxx yyvv', top_k=3), [])

    def test_bm25_分數不能當作絕對門檻(self):
        """這是 BM25 相對於向量檢索最關鍵的弱點，值得用測試釘住。

        中文用 bigram 斷詞後，「沒有」「東西」這類常見詞會讓一句
        完全不相干的話也拿到不低的分數。所以你無法用「分數低於 X
        就當作查無資料」——向量的餘弦相似度才有這個性質。
        """
        nonsense = self.index.rank('沒有這種東西', top_k=1)
        real = self.index.rank('資料庫遷移 回滾', top_k=1)
        self.assertTrue(nonsense, '亂查也會有分數，這正是問題所在')
        # 雖然真正相關的分數高很多，但沒有一條「絕對安全」的門檻
        self.assertGreater(real[0][1], nonsense[0][1] * 3)


# --------------------------------------------------------------------------
# 第二層：測 Agent Loop（把 client 換成假的）
# --------------------------------------------------------------------------

def run_loop(client, tools_map, question, max_turns=5):
    """第 7 節那個迴圈，抽出來成為可測的函式。

    重點：它接受 client 當參數，而不是自己去建一個。
    這叫依賴注入，是讓 Agent 可測的唯一關鍵改動。
    """
    messages = [{'role': 'user', 'content': question}]
    for _ in range(max_turns):
        response = client.chat(model='fake', messages=messages,
                               tools=list(tools_map.values()))
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content, messages

        for tc in response.message.tool_calls:
            fn = tools_map.get(tc.function.name)
            if fn is None:
                result = f'錯誤：沒有名為 {tc.function.name} 的工具'
            else:
                try:
                    result = fn(**tc.function.arguments)
                except Exception as exc:              # noqa: BLE001
                    result = f'工具執行失敗：{type(exc).__name__}: {exc}'
            messages.append({'role': 'tool', 'tool_name': tc.function.name,
                             'content': str(result)})
    return '（已達最大輪數上限）', messages


def add(a: int, b: int) -> int:
    """把兩個整數相加"""
    return a + b


def always_fails(x: int) -> int:
    """一定會爆炸的工具，用來測錯誤處理"""
    raise RuntimeError('資料庫連線失敗')


TOOLS_MAP = {'add': add, 'always_fails': always_fails}


class TestAgentLoop(unittest.TestCase):

    def test_沒有工具呼叫時應立刻結束(self):
        client = FakeClient([FakeMessage(content='直接回答')])
        answer, _ = run_loop(client, TOOLS_MAP, '哈囉')
        self.assertEqual(answer, '直接回答')
        self.assertEqual(len(client.calls), 1)

    def test_工具結果會被正確餵回模型(self):
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall('add', {'a': 2, 'b': 3})]),
            FakeMessage(content='答案是 5'),
        ])
        answer, messages = run_loop(client, TOOLS_MAP, '2+3')
        self.assertEqual(answer, '答案是 5')

        tool_msgs = [m for m in messages if isinstance(m, dict)
                     and m.get('role') == 'tool']
        self.assertEqual(tool_msgs[0]['content'], '5')
        # 第二次呼叫模型時，歷史裡必須含有工具結果
        self.assertIn('5', str(client.calls[1]['messages']))

    def test_工具丟例外不會讓_agent_中斷(self):
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall('always_fails', {'x': 1})]),
            FakeMessage(content='我換個方式處理'),
        ])
        answer, messages = run_loop(client, TOOLS_MAP, '試試看')
        self.assertEqual(answer, '我換個方式處理')
        # 錯誤訊息要傳給模型，它才知道發生什麼事
        self.assertIn('資料庫連線失敗', str(messages))

    def test_模型幻覺出不存在的工具要被擋下(self):
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall('不存在的工具', {})]),
            FakeMessage(content='好的'),
        ])
        _, messages = run_loop(client, TOOLS_MAP, '試試看')
        self.assertIn('沒有名為', str(messages))

    def test_模型一直要求工具時要被_max_turns_擋住(self):
        # 沒有這道保險絲，失控的 Agent 會一直燒 GPU 時間
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall('add', {'a': 1, 'b': 1})])
            for _ in range(10)
        ])
        answer, _ = run_loop(client, TOOLS_MAP, '無限迴圈', max_turns=3)
        self.assertIn('上限', answer)
        self.assertEqual(len(client.calls), 3)

    def test_平行工具呼叫每個都要執行(self):
        client = FakeClient([
            FakeMessage(tool_calls=[FakeToolCall('add', {'a': 1, 'b': 1}),
                                    FakeToolCall('add', {'a': 10, 'b': 10})]),
            FakeMessage(content='都算好了'),
        ])
        _, messages = run_loop(client, TOOLS_MAP, '算兩個')
        results = [m['content'] for m in messages
                   if isinstance(m, dict) and m.get('role') == 'tool']
        self.assertEqual(results, ['2', '20'])


# --------------------------------------------------------------------------
# 第三層：測記憶與裁切（同樣不用模型）
# --------------------------------------------------------------------------

class TestMemoryTrimming(unittest.TestCase):

    def setUp(self):
        import importlib
        self.mem = importlib.import_module('15_conversation_memory')

    def _history(self, n=12):
        msgs = [{'role': 'system', 'content': 'S'}]
        for i in range(n):
            msgs += [{'role': 'user', 'content': f'問題 {i} ' + '字' * 40},
                     {'role': 'assistant', 'content': f'回答 {i} ' + '字' * 40}]
        return msgs

    def test_滑動視窗會壓到預算以內(self):
        trimmed = self.mem.trim_sliding_window(self._history(), budget=100)
        self.assertLessEqual(self.mem.estimate_tokens(trimmed), 100)

    def test_system_訊息永遠不能被裁掉(self):
        # 裁掉 system 等於 Agent 突然忘記自己是誰，是最惡劣的失敗
        trimmed = self.mem.trim_sliding_window(self._history(), budget=20)
        self.assertEqual(trimmed[0]['role'], 'system')

    def test_不該留下沒有對應_assistant_的孤兒_tool_訊息(self):
        msgs = [{'role': 'system', 'content': 'S'}]
        for i in range(6):
            msgs += [{'role': 'user', 'content': '問' * 30},
                     {'role': 'assistant', 'content': ''},
                     {'role': 'tool', 'tool_name': 't', 'content': '果' * 30}]
        trimmed = self.mem.trim_sliding_window(msgs, budget=60)
        after_system = trimmed[1:]
        if after_system:
            self.assertNotEqual(after_system[0]['role'], 'tool')

    def test_短工具結果不該被換成更長的佔位符(self):
        # 這是我實際寫錯過的 bug：佔位符比 '31°C' 還長，越裁越肥
        msgs = [{'role': 'system', 'content': 'S'}]
        msgs += [{'role': 'tool', 'tool_name': 't', 'content': '31°C'}] * 8
        msgs += [{'role': 'user', 'content': 'x'}] * 4
        with redirect_stdout(io.StringIO()):
            trimmed = self.mem.trim_drop_tool_results(msgs, budget=10)
        self.assertLessEqual(self.mem.estimate_tokens(trimmed),
                             self.mem.estimate_tokens(msgs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
