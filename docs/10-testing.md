# 測試 Agent：大部分測試根本不用呼叫模型

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 用假的 client 把 Agent Loop 測起來，不花 GPU 時間也不看運氣

[← 多輪對話與 Context 管理](09-memory-context.md)　·　[全部進階主題](README.md)　·　[回到核心路徑 →](../README.md)

---

## 1. 「Agent 沒辦法測」只對了一半

最常聽到的說法是：模型輸出不固定，所以測不了。

拆開來看就知道問題出在哪：

```
工具（純函式）        ← 完全可測，而且大部分 bug 在這裡
Agent Loop（控制流）  ← 完全可測，把 client 換成假的就好
模型的判斷             ← 確實不可測，只能用評估集看趨勢
```

**前兩層是你自己寫的程式碼，佔了 bug 的絕大多數。** 不可測的只有第三層，而那層有[評估集](06-rag.md#9-怎麼自動驗證-rag-的正確率)可以處理。

`examples/16_testing_agent.py` 有 18 個測試，跑完 **0.2 秒**，一次模型都沒呼叫。

## 2. 讓 Agent 可測的唯一關鍵改動

把 `client` 當參數傳進去，而不是在函式裡自己建一個：

```python
def run_loop(client, tools_map, question, max_turns=5):
    """接受 client 當參數，而不是自己去建一個。

    這叫依賴注入，是讓 Agent 可測的唯一關鍵改動。
    """
    messages = [{'role': 'user', 'content': question}]
    for _ in range(max_turns):
        response = client.chat(model='fake', messages=messages, ...)
```

有了這個，就能塞一個照劇本回應的假 client：

```python
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
```

那個 `AssertionError` 本身就是一個測試：**劇本用完還在呼叫，代表你的終止條件有問題。**

## 3. 該測什麼

### 工具的安全邊界（優先度最高）

模型真的會送奇怪的參數進來。這類測試投報率最高：

```python
def test_路徑逃逸會被擋下來(self):
    for evil in ['../../etc/passwd', '/etc/passwd', '../../../root/.ssh/id_rsa']:
        with self.subTest(path=evil):
            with self.assertRaises(ValueError):
                self.agent._safe_path(evil)

def test_讀不存在的檔案回錯誤訊息而不是丟例外(self):
    # 模型讀得懂錯誤訊息，會自己換個做法重試；丟例外則會讓整個 Agent 中斷
    result = self.agent.read_file('絕對不存在的檔案.txt')
    self.assertIn('不存在', result)
```

### Agent Loop 的控制流

```python
def test_模型一直要求工具時要被_max_turns_擋住(self):
    # 沒有這道保險絲，失控的 Agent 會一直燒 GPU 時間
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall('add', {'a': 1, 'b': 1})])
        for _ in range(10)
    ])
    answer, _ = run_loop(client, TOOLS_MAP, '無限迴圈', max_turns=3)
    self.assertIn('上限', answer)
    self.assertEqual(len(client.calls), 3)


def test_工具丟例外不會讓_agent_中斷(self):
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall('always_fails', {'x': 1})]),
        FakeMessage(content='我換個方式處理'),
    ])
    answer, messages = run_loop(client, TOOLS_MAP, '試試看')
    self.assertEqual(answer, '我換個方式處理')
    # 錯誤訊息要傳給模型，它才知道發生什麼事
    self.assertIn('資料庫連線失敗', str(messages))


def test_平行工具呼叫每個都要執行(self):
    # 這是 06_streaming_agent.py 那個 extend / 賦值 bug 的回歸測試
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall('add', {'a': 1, 'b': 1}),
                                FakeToolCall('add', {'a': 10, 'b': 10})]),
        FakeMessage(content='都算好了'),
    ])
    _, messages = run_loop(client, TOOLS_MAP, '算兩個')
    results = [m['content'] for m in messages
               if isinstance(m, dict) and m.get('role') == 'tool']
    self.assertEqual(results, ['2', '20'])
```

### 記憶與裁切

```python
def test_system_訊息永遠不能被裁掉(self):
    # 裁掉 system 等於 Agent 突然忘記自己是誰，是最惡劣的失敗
    trimmed = self.mem.trim_sliding_window(self._history(), budget=20)
    self.assertEqual(trimmed[0]['role'], 'system')

def test_短工具結果不該被換成更長的佔位符(self):
    # 這是我實際寫錯過的 bug（見「多輪對話與 Context 管理」那篇）
    ...
    self.assertLessEqual(self.mem.estimate_tokens(trimmed),
                         self.mem.estimate_tokens(msgs))
```

## 4. 寫這些測試時，測試自己抓到的兩件事

這節是我覺得最值得寫下來的部分——**兩個測試一開始是紅的，而兩次都是我對現實的假設錯了，不是測試寫錯。**

### 一、`search_code` 搜到了測試檔案本身

```python
result = self.agent.search_code('zzz絕不存在的字串zzz')
self.assertIn('找不到', result)     # ❌ FAIL
```

失敗訊息是：

```
AssertionError: '找不到' not found in
  "examples/16_testing_agent.py:103: result = self.agent.search_code('zzz絕不存在的字串zzz')"
```

**它在測試檔案的原始碼裡找到了那個字串。** 自我指涉。修法是在執行期組出來：

```python
needle = 'zzq' + 'absent' + 'marker' + 'zzq'
```

任何會掃描自己專案的工具都有這個坑。

### 二、BM25 對中文亂碼查詢照樣給分

```python
self.assertEqual(self.index.rank('zzzz沒有這種東西zzzz', top_k=3), [])   # ❌ FAIL
# AssertionError: [(5, 1.707), (4, 0.706), (2, 0.675)] != []
```

原因是[中文用 bigram 斷詞](06-rag.md)之後，「沒有」「東西」這些詞真的出現在語料裡。

這不是 bug，是 BM25 的本質，而且它**用測試釘住了[那篇 RAG 文件裡的結論](06-rag.md#8-向量-vs-關鍵字什麼時候值得付這個複雜度)**：向量的餘弦相似度可以當「查無資料」的門檻，BM25 的分數不行。所以我把測試改成斷言這個性質本身：

```python
def test_bm25_分數不能當作絕對門檻(self):
    nonsense = self.index.rank('沒有這種東西', top_k=1)
    real = self.index.rank('資料庫遷移 回滾', top_k=1)
    self.assertTrue(nonsense, '亂查也會有分數，這正是問題所在')
    # 雖然真正相關的分數高很多，但沒有一條「絕對安全」的門檻
    self.assertGreater(real[0][1], nonsense[0][1] * 3)
```

**測試的價值不只是防止退步，是逼你把「我以為的行為」寫下來然後被現實檢驗。** 這兩個假設如果沒寫成測試，我會一直以為它們是對的。

## 5. 三層測試策略總結

| 層次 | 用什麼測 | 花費 | 多久跑一次 |
| --- | --- | --- | --- |
| 工具（純函式） | 一般單元測試 | 零 | 每次存檔 |
| Agent Loop | FakeClient | 零 | 每次存檔 |
| 檢索品質 | [評估集第一層](06-rag.md#9-怎麼自動驗證-rag-的正確率) | 零（不呼叫模型） | 每次改切塊或 embedding |
| 端到端行為 | [評估集第二層](06-rag.md#9-怎麼自動驗證-rag-的正確率) | GPU 時間 | 改 prompt 或換模型時 |

前三層都是零成本，應該綁進 CI。只有最後一層要花錢，而且[有隨機性](06-rag.md#9-怎麼自動驗證-rag-的正確率)，適合看趨勢而不是當通過／不通過的閘門。

```bash
python examples/16_testing_agent.py            # 前兩層，0.2 秒
python examples/14_rag_eval.py                 # 第三層，秒級
python examples/14_rag_eval.py --end-to-end    # 第四層，要花 GPU 時間
```

---

[← 多輪對話與 Context 管理](09-memory-context.md)　·　[全部進階主題](README.md)　·　[回到核心路徑 →](../README.md)
