# 多輪對話與 Context 管理

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 為什麼 Agent 記不住上一句、context 為什麼會爆、三種裁切策略的實測差異

[← 框架與可觀測性](08-frameworks-observability.md)　·　[全部進階主題](README.md)　·　[測試 Agent →](10-testing.md)

---

## 1. 先講一個初學者一定會撞到的事

前面所有範例都是「問一次、答一次」。真的做產品時，第一個撞到的問題是：

```
你：台北現在幾度？
助理：台北目前氣溫約 31°C。

你：那東京呢？
助理：抱歉，我不清楚您指的是什麼。     ← ？？？
```

**模型沒有記憶。** 它每次只看到你送過去的 `messages`，如此而已。所謂「記得」，完全是你自己把歷史一起送過去的結果。

修法簡單到有點反高潮——**把 `messages` 這個 list 跨輪保留，而不是每次重新建立**：

```python
class ConversationalAgent:
    def __init__(self):
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]   # ← 存在 instance 上

    def ask(self, question: str) -> str:
        self.messages.append({'role': 'user', 'content': question})      # ← 累積，不是覆蓋
        ...
```

就這樣。前面每一支範例之所以沒有記憶，只是因為 `messages` 是函式裡的區域變數。

實測（`examples/15_conversation_memory.py --demo`）：

```
你：台北現在幾度？        → 台北目前氣溫約 31°C
你：那東京呢？            → 東京目前氣溫約 18°C          ← 懂「那」是什麼了
你：首爾跟大阪呢？        → 首爾 22°C、大阪 20°C
你：剛才問過的城市裡，哪一個最冷？
                        → 在剛才查詢的四個城市中，東京（18°C）是最冷的
```

最後那題需要模型回顧**全部四輪**，這是驗證記憶有沒有真的在運作最好的問法。

## 2. 但歷史不能無限長

三個理由，由輕到重：

- **會變貴**。Ollama Cloud 按 GPU 時間計費，歷史越長，每一輪的 prompt 處理時間越久
- **會抓不到重點**。太長的 context 會稀釋注意力，這是有名的「大海撈針」問題
- **會直接爆掉**。超過模型 context 上限就報錯

所以「記憶」的真正工作不是「存起來」，**是決定丟掉什麼**。

要決定丟什麼，得先知道現在多長。正式做法是看回應裡的 `prompt_eval_count`，但那要送出去才知道，沒辦法拿來**事前**決定。所以先粗估：

```python
def estimate_tokens(messages: list) -> int:
    """中文約 1 字 ≈ 1 token，英文約 4 字元 ≈ 1 token，取中間值 2。

    估得不準沒關係，我們只需要一個「該裁了」的訊號。
    """
    total = 0
    for m in messages:
        content = m.get('content') if isinstance(m, dict) else getattr(m, 'content', '')
        total += len(str(content or '')) // 2
    return total
```

## 3. 三種裁切策略

### 策略一：滑動視窗

保留 system + 最近的訊息，舊的直接丟。

```python
def trim_sliding_window(messages: list, budget: int) -> list:
    system = [m for m in messages[:1] if _role(m) == 'system']
    rest = messages[len(system):]

    while rest and estimate_tokens(system + rest) > budget:
        rest.pop(0)
        # tool 訊息不能沒有對應的 assistant tool_calls，
        # 開頭若是孤兒 tool 訊息就一起丟掉
        while rest and _role(rest[0]) == 'tool':
            rest.pop(0)
    return system + rest
```

最簡單、最常用。那個內層 while 很容易漏——**孤兒 `tool` 訊息會讓 API 報錯**，因為它沒有對應的 `assistant` tool_calls。

### 策略二：摘要壓縮

把舊訊息交給模型濃縮成一段，接在 system 後面。

```python
summary = client.chat(
    model=MODEL, options={'temperature': 0},
    messages=[{'role': 'user',
               'content': f'把以下對話濃縮成三句話以內的重點，'
                          f'保留使用者提過的具體事實（地點、數字、偏好）：\n\n{transcript}'}],
).message.content

return system + [{'role': 'user', 'content': f'（先前對話摘要）{summary}'}] + keep
```

保留了舊資訊的重點，代價是**多一次模型呼叫**。長對話（客服、助理）值得，短任務不值得。

「保留使用者提過的具體事實（地點、數字、偏好）」這句不能省——不然摘要會變成「使用者詢問了幾個城市的天氣」，具體數字全丟了，等於白壓縮。

### 策略三：優先丟工具結果

Agent 的 context 通常是被工具輸出撐爆的（想想 `read_file` 讀進一個大檔），對話本身其實很短。所以先把肥的工具結果換成佔位符。

```python
PLACEHOLDER = '（舊工具結果已省略）'

is_old_tool = _role(m) == 'tool' and i < len(messages) - 4
# 只有「換掉之後真的比較短」才換，否則這步是負優化
if is_old_tool and len(_content(m)) > len(PLACEHOLDER):
    trimmed.append({**_as_dict(m), 'content': PLACEHOLDER})
```

## 4. 我在這裡寫錯的 bug，值得你避開

第一版我無條件把所有舊工具結果換成佔位符。問題是：

```
'31°C'                    →  '（舊工具結果已省略）'
   5 字元                        11 字元          ← 越裁越肥
```

**佔位符比它取代的內容還長。** 省不下來，於是靜默退回滑動視窗，早期對話被整段砍掉。結果 Agent 答錯了：

| 策略 | 「剛才問過的城市裡，哪一個最冷？」 |
| --- | --- |
| 滑動視窗 | 東京 18°C ✅ |
| 摘要壓縮 | 東京 18°C ✅ |
| 優先丟工具結果（bug 版） | **大阪 20°C ❌** |

實際送給模型的 context 長這樣：

```
system     '你是氣象助理…'
assistant  ''
tool       '20°C'
assistant  '- 首爾：22°C  - 大阪：20°C'
user       '剛才問過的城市裡，哪一個最冷？'
```

台北和東京完全不見了。模型只看得到首爾與大阪，答大阪其實是**正確推理搭配殘缺資料**。

兩個修法：

1. **只換真的比佔位符長的內容**
2. **退場時要出聲**

```python
if estimate_tokens(trimmed) > budget:
    # 靜默退場會讓你以為「記憶好好的」，
    # 直到使用者問了一個需要回顧早期資訊的問題才發現答錯
    print('[警告] 光丟工具結果不夠，退回滑動視窗，早期對話將遺失')
    return trim_sliding_window(trimmed, budget)
```

## 5. 從這件事該學到什麼

**一、裁切策略一定要用「需要回顧早期資訊」的問題測。**

如果我的示範只問到第三句，三種策略看起來都完美。是最後那句「剛才問過的城市裡，哪一個最冷？」才把問題逼出來。你的測試集也該有這種題。

**二、Context 遺失是靜默失敗。**

模型不會說「我忘了」，它會用手上僅有的資料給你一個**讀起來完全合理的錯誤答案**。這比拋例外難查得多，所以裁切發生時一定要留下紀錄——這也是[可觀測性](08-frameworks-observability.md)那篇的價值所在。

**三、選策略要看「資訊住在哪裡」。**

| 你的事實主要在 | 該選 |
| --- | --- |
| 工具結果裡（查詢數值、檔案內容） | 摘要壓縮，或加大預算 |
| 模型的回覆裡（已經被轉述過） | 優先丟工具結果 |
| 最近幾輪就夠（純閒聊、單一任務） | 滑動視窗 |

## 6. 還有什麼可以做

這篇談的是**單一對話內**的記憶。跨 session 的長期記憶（記住使用者上週說過什麼）是另一個題目，通常做法是把結論寫進[向量庫](06-rag.md)，下次對話開始時先檢索一次塞進 system prompt——本質上就是把 RAG 用在「使用者的歷史」而不是「文件」上。

---

[← 框架與可觀測性](08-frameworks-observability.md)　·　[全部進階主題](README.md)　·　[測試 Agent →](10-testing.md)
