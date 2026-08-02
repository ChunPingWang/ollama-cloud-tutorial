# 進階主題

核心路徑（[README](../README.md) 第 1–8 節）讀完之後，這些主題各自獨立，需要哪塊看哪塊，不用照順序。

| 主題 | 難度 | 內容 |
| --- | --- | --- |
| [結構化輸出：讓 Agent 吐出可以直接用的資料](01-structured-output.md) | 中階 | 雲端不強制 format 的坑，以及兩種實測可行的替代方案 |
| [串流與 Thinking：把思考過程秀出來](02-streaming.md) | 中階 | 邊跑邊顯示，以及串流時累積 tool_calls 的正確寫法 |
| [用 OpenAI SDK 相容層接既有生態系](03-openai-compat.md) | 中階 | 把既有以 OpenAI 為介面的程式接到 Ollama Cloud |
| [接上 MCP：不用自己寫工具](04-mcp.md) | 進階 | 30 行橋接，把整個 MCP 生態系的工具接進你的 Agent |
| [實際使用情境：GPU 時間計費下的成本控制](05-cost.md) | 中階 | 三個反直覺的實測數字，以及分層路由的損益平衡點 |
| [RAG：為什麼雲端做不了，以及兩條可走的路](06-rag.md) | 進階 | embedding 的角色、雲端的限制、兩條驗證過的路徑、自動化評估 |
| [正式上線前要處理的事](07-production.md) | 中階 | 重試、成本、可觀測性、安全邊界 |
| [接框架與可觀測性：LangChain / Langfuse / LangSmith](08-frameworks-observability.md) | 進階 | 什麼時候該用框架，以及怎麼看見 Agent 在想什麼 |
| [多輪對話與 Context 管理](09-memory-context.md) | 中階 | 為什麼 Agent 記不住上一句、三種裁切策略的實測差異 |
| [測試 Agent](10-testing.md) | 中階 | 用假 client 把 Agent Loop 測起來，0.2 秒跑完 18 個測試 |

---

[← 回到核心路徑](../README.md)
