# 正式上線前要處理的事

> **難度**：中階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 重試、成本、可觀測性、安全邊界

[← RAG](06-rag.md)　·　[全部進階主題](README.md)　·　[接框架與可觀測性 →](08-frameworks-observability.md)

---

前面的程式碼是為了讀起來清楚。真的要上線，這幾件事跑不掉：

### 重試與逾時

網路會抖，雲端服務會偶發 5xx。至少包一層指數退避：

```python
import time
from ollama import ResponseError

def chat_with_retry(client, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return client.chat(**kwargs)
        except ResponseError as exc:
            if exc.status_code and exc.status_code < 500:
                raise                       # 4xx 是你的問題，重試沒用
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

注意 4xx 不要重試——認證錯誤、模型名稱打錯、參數格式不對，重試一百次也一樣。

### 成本控制

[〈實際使用情境〉](05-cost.md)整節都在講這件事，這裡只補上線相關的兩點：

- **配額與 rate limit**。免費方案除了模型受限（見[核心路徑的環境準備](../README.md#4-環境準備)），也有速率限制。正式跑之前看一下 [ollama.com/pricing](https://ollama.com/pricing) 的方案配額，並且在程式裡把 429 當成需要退避的錯誤處理。
- **設預算告警**。GPU 時間計費的好處是可預估，前提是你真的有在看。把[〈接框架與可觀測性〉](08-frameworks-observability.md)的 trace 接起來，至少能回答「這個月哪個 Agent 吃掉最多時間」。

### 可觀測性

至少把每輪的 `tool_calls` 和 `thinking` 記下來。Agent 出錯時你要回答的問題是「它為什麼決定做這件事」，而那個答案只在 thinking 裡。

### 安全邊界

- API key 用環境變數或 secret manager，不要進版控
- **能寫入的工具（刪檔、發 API、送信）一律加人工確認關卡**，別讓模型直接觸發不可逆操作
- 路徑、SQL、shell 指令這類參數，一律當成不可信輸入來驗證——模型的輸出本質上就是使用者可以間接控制的內容

### 資料落地

雲端模式下對話內容會送到 Ollama 的伺服器。有合規要求的資料，該過濾就過濾，或改跑本地模型。

---

---

[← RAG](06-rag.md)　·　[全部進階主題](README.md)　·　[接框架與可觀測性 →](08-frameworks-observability.md)
