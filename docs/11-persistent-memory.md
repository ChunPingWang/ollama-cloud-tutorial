# 跨 Session 長期記憶

> **難度**：進階　|　**前置**：[多輪對話與 Context 管理](09-memory-context.md)、[RAG](06-rag.md)
> 把「記住」做成 Agent 的工具，以及記憶衝突為什麼比你想的難處理

[← 測試 Agent](10-testing.md)　·　[全部進階主題](README.md)　·　[部署上線 →](12-deployment.md)

---

## 1. 這跟上一篇差在哪裡

[上一篇](09-memory-context.md)講的是**單一對話內**的記憶——程式一關就沒了。

這篇是跨 session：今天說「我對花生過敏」，下週重新啟動程式它還記得。

做法上有兩派：

| | 做法 | 優點 | 缺點 |
| --- | --- | --- | --- |
| A | 全量存對話 + 每次檢索 | 實作簡單，不漏資訊 | 雜訊多，「嗯」「好喔」也被存進去稀釋檢索品質 |
| B | **抽取事實再存** | 記憶乾淨、可查、可修正 | 要決定「什麼值得記」 |

本篇用 B。而「什麼值得記」這個判斷——**交給模型自己決定**，把 `remember` 做成一個工具。這跟[核心路徑第 7 節](../README.md#7-手刻-agent-loop)的 Agent Loop 是同一個模式。

儲存用 SQLite（Python 內建），檢索沿用 [`rag_common.py` 的 BM25](06-rag.md)。**整套不需要向量資料庫，也不需要本地 Ollama**——所以純雲端部署也能用。

## 2. 把「記住」做成工具

```python
def remember(subject: str, fact: str) -> str:
    """記住一件關於使用者的長期事實，之後的對話都會用得到。

    只記真正持久的資訊：偏好、限制、身分、長期目標。
    不要記一次性的問題內容，也不要記你自己剛才的回答。

    Args:
        subject: 這件事的主題，用簡短名詞，例如「飲食限制」「慣用語言」「職稱」。
                 同一個主題再次記錄時會覆蓋舊的，所以主題要取得一致。
        fact: 事實內容，一句話講完
    """
```

docstring 就是模型看到的規則。「不要記一次性的問題內容」這句不能省——沒有它，模型會把「使用者問了午餐吃什麼」也存進去，記憶很快就變成垃圾場。

## 3. 儲存：不刪，只標記過期

```sql
CREATE TABLE memories (
    id         INTEGER PRIMARY KEY,
    subject    TEXT NOT NULL,      -- 主題，用來偵測衝突
    fact       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    superseded INTEGER NOT NULL DEFAULT 0
)
```

```python
def remember(self, subject: str, fact: str) -> str:
    """同主題的舊記憶會被標記為過期，而不是刪掉。

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
```

「不刪只標記」在除錯時價值很高：Agent 突然行為改變，你可以直接查那筆記憶什麼時候被改的、改成什麼。

## 4. 記憶怎麼進到對話裡

```python
def build_system_prompt() -> str:
    """把已知記憶放進 system prompt。

    這是「開場注入」策略：記憶不多時直接全部放進去，模型不用先查就知道。
    記憶多到塞不下時，改成只在 system 放摘要，細節靠 recall 工具查。
    """
```

兩種策略的取捨：

| | 開場注入 | 靠 recall 工具查 |
| --- | --- | --- |
| 記憶量 | 少（幾十筆） | 多 |
| 反應 | 模型一開始就知道，不會漏 | 要模型想到去查才會用 |
| 成本 | 每輪都帶著，context 變長 | 只在需要時付費 |

實務上混用：把最關鍵的（過敏、語言偏好）注入，其餘靠工具查。

## 5. 實測：兩個 session 的連續性

```bash
python examples/17_persistent_memory.py --demo
```

**Session 1**（第一次見面）：

```
你：你好，我叫 Rex，是後端工程師，平常寫 Python。
  [remember] {'subject': '姓名', 'fact': '我的名字是 Rex'}
  [remember] {'subject': '職業', 'fact': '我的職業是後端工程師'}
  [remember] {'subject': '慣用語言', 'fact': '我常使用的程式語言是 Python'}

你：對了我對花生過敏，蠻嚴重的。
  [remember] {'subject': '飲食限制', 'fact': '我對花生過敏，且相當嚴重'}
```

**Session 2**（重新啟動程式，全新的 agent 物件）：

```
啟動時已知 4 筆記憶

你：午餐吃什麼好？
  [recall] {'query': '食物過敏'} → - 飲食限制：我對花生過敏，且嚴重
助理：…以下幾個選項可以參考（都不含花生）…      ← 沒有再問一次
```

最後那題是真正的驗收：**它沒有再問「你有什麼飲食限制嗎」，直接用了上個 session 的資訊。**

## 6. 我撞到的 bug：記憶衝突比想像中難

第一版跑完，最終記憶長這樣：

```
身分：Rex, 後端工程師, 使用 Python          ← 舊的，還活著
使用語言：Rex 最近改用 Go，Python 較少寫     ← 新的
```

**兩筆互相矛盾的記憶並存。**

原因是我的覆蓋機制建立在「同一個 subject」上，但模型第一次用了主題 `身分`（把語言寫在裡面），第二次用了主題 `使用語言`。**主題名稱不同，覆蓋就不會觸發。**

這是「讓模型自由決定 subject」的必然代價。兩個修法我都做了：

**修法一：把現有主題列給模型看**

```python
subjects = sorted({s for _, s, _ in STORE.active()})
return (f'{SYSTEM_BASE}\n\n你已知關於這位使用者的事：\n{listed}\n\n'
        f'目前已使用的記憶主題：{"、".join(subjects)}。\n'
        f'更新既有資訊時**必須沿用上面已存在的主題名稱**，'
        f'不要另創新主題，否則會產生兩筆互相矛盾的記憶。')
```

**修法二：寫入時偵測跨主題衝突，回饋給模型**

```python
def find_conflicts(self, subject: str, fact: str) -> list[str]:
    """找出「別的主題底下」可能講同一件事的記憶。

    單靠「同主題覆蓋」不夠，還要能看見跨主題的衝突。
    """
    rows = [r for r in self.active() if r[1] != subject]
    index = BM25([f'{s} {f}' for _, s, f in rows])
    return [f'{rows[i][1]}：{rows[i][2]}'
            for i, score in index.rank(fact, top_k=3) if score > 1.0]
```

偵測到就把訊息附在工具結果裡：

```
已記住「慣用語言」：我主要使用的程式語言是 Go
⚠ 注意：其他主題底下有可能矛盾的舊記憶：「職業：我的職業是後端工程師」。
   若確實已過時，請用那個主題名稱再呼叫一次 remember 更新它。
```

**重點是「回饋給模型」而不是在程式裡硬判斷。** 程式無法可靠地判斷「後端工程師」跟「改用 Go」算不算矛盾——但模型可以。這又回到同一個模式：**把判斷交給模型，把機制交給程式**。

修完之後的最終狀態：

```
姓名：我的名字是 Rex
職業：我的職業是後端工程師
飲食限制：我對花生過敏，且相當嚴重
慣用語言：我主要使用的程式語言是 Go
    （已過期）我常使用的程式語言是 Python
```

乾淨，而且保留了變更歷史。

## 7. 上線前還要想的三件事

**一、記憶會膨脹。** 幾個月後可能有幾百筆。做法：加 `last_used_at`，長期沒被檢索到的降權或歸檔。

**二、記憶是個人資料。** 這個 SQLite 裡面是使用者的過敏史、職業、偏好。要有刪除機制（`--forget` 只是最陽春的版本），多使用者時要按 user 隔離，並且想清楚保留期限。

**三、記憶污染。** 使用者（或被注入的內容）可以讓 Agent 記住任意事情，之後每次對話都會帶著。這是 prompt injection 的持久化版本，比單次注入嚴重得多。至少要做到：只記使用者**直接說的**，不要記從網頁或文件讀來的內容。

---

[← 測試 Agent](10-testing.md)　·　[全部進階主題](README.md)　·　[部署上線 →](12-deployment.md)
