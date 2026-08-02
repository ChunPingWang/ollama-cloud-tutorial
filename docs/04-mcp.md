# 接上 MCP：不用自己寫工具

> **難度**：進階　|　**前置**：[核心路徑](../README.md#三條學習路徑)第 1–8 節
> 30 行橋接，把整個 MCP 生態系的工具接進你的 Agent

[← 用 OpenAI SDK 相容層接既有生態系](03-openai-compat.md)　·　[全部進階主題](README.md)　·　[實際使用情境 →](05-cost.md)

---

[〈實戰〉](../README.md#8-實戰一個會讀專案的-codebase-agent)那三個工具是我們自己刻的。但如果 Agent 要能查 GitHub、讀 Slack、打資料庫、操作瀏覽器呢？一個一個手寫，寫到天荒地老。

**MCP（Model Context Protocol）解決的就是這件事**：它把「工具」標準化成一個協定，任何人寫的 MCP Server 都能被任何 MCP Client 使用。現在 GitHub、Notion、Playwright、PostgreSQL、檔案系統……都有現成的 Server。接上去，你的 Agent 立刻多幾十個工具。

## 1. 先講清楚：Ollama 沒有內建 MCP

這點要先說，因為很多文章講得很含糊。**截至目前，Ollama 本身不支援 MCP**（官方 repo 的 [issue #7865](https://github.com/ollama/ollama/issues/7865) 還開著）。你不能像設定 Claude Desktop 那樣丟一個 `mcp.json` 給 Ollama 就完事。

但這其實沒什麼大不了，因為橋接的工作量小到有點好笑：

| | MCP | Ollama |
| --- | --- | --- |
| 工具清單 | `client.list_tools()` → `tool.name` / `tool.description` / `tool.input_schema` | `tools=[{'type':'function','function':{'name','description','parameters'}}]` |
| 執行工具 | `await client.call_tool(name, args)` | 你自己執行，結果以 `role: 'tool'` 塞回去 |

兩邊的參數描述**都是 JSON Schema**。所以橋接層的本質只是搬欄位——大概 30 行。

架構長這樣：

```
                 ┌──────────────────────────────┐
  你的 Agent ───→│  MCPToolBridge（30 行）       │
   (Ollama)      │  list_tools() → tools schema │
       ↑         │  tool_calls   → call_tool()  │
       └─────────│                              │
   tool 結果      └───────┬──────────────────────┘
                         │ MCP 協定 (stdio / HTTP)
                 ┌───────┴───────┬──────────────┐
              工單系統        GitHub        檔案系統
             (自己寫的)      (官方 Server)   (官方 Server)
```

## 2. 準備一台 MCP Server

為了讓範例能獨立跑，先自己寫一台。用 MCP Python SDK 寫 Server 跟寫普通函式差不多——`@mcp.tool()` 會從型別標註和 docstring 自動生出 schema，跟[〈Agent 的心臟〉](../README.md#6-agent-的心臟tool-calling) Ollama SDK 的做法是同一個思路。

`examples/mcp_server_demo.py`（節錄）：

```python
from mcp.server import MCPServer

mcp = MCPServer('TicketSystem')

TICKETS = {
    'T-101': {'title': '結帳頁在 Safari 會卡住', 'status': 'open',
              'priority': 'high', 'assignee': 'alice',
              'comments': ['使用者回報只有 iOS 18 會發生']},
    # ...
}


@mcp.tool()
def list_tickets(status: str = 'all') -> str:
    """列出工單，可依狀態篩選。

    Args:
        status: 篩選條件，可填 open、closed 或 all（預設）
    """
    rows = []
    for tid, t in TICKETS.items():
        if status != 'all' and t['status'] != status:
            continue
        rows.append(f"{tid} [{t['status']}/{t['priority']}] {t['title']}")
    return '\n'.join(rows) or f'沒有狀態為 {status} 的工單'


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """取得單一工單的完整內容，包含所有留言。

    Args:
        ticket_id: 工單編號，例如 T-101
    """
    t = TICKETS.get(ticket_id)
    if t is None:
        raise ValueError(f'找不到工單 {ticket_id}，現有的有 {list(TICKETS)}')
    return f"編號：{ticket_id}\n標題：{t['title']}\n狀態：{t['status']}\n..."


if __name__ == '__main__':
    mcp.run()   # 預設走 stdio，由呼叫端以子行程啟動
```

注意 `get_ticket` 裡那個 `raise ValueError`。**MCP 的錯誤處理很聰明：Server 端丟例外不會讓 Client 也炸掉**，而是變成一個 `is_error=True` 的正常回應。這正是我們要的——模型讀得到錯誤訊息，就能自己修正參數重試。

## 3. 橋接層

核心就這個類別（完整版在 `examples/08_mcp_agent.py`）：

```python
from contextlib import AsyncExitStack
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolBridge:
    """連上一或多台 MCP Server，把它們的工具攤平成 Ollama 的 tools 陣列。

    工具名稱會加上 server 前綴（例如 tickets__get_ticket），避免多台
    server 提供同名工具時撞在一起。
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._routes: dict[str, tuple[Client, str]] = {}
        self.tools: list[dict] = []

    async def connect(self, server_name: str, transport) -> list[str]:
        client = await self._stack.enter_async_context(Client(transport))
        listing = await client.list_tools()

        for tool in listing.tools:
            qualified = f'{server_name}__{tool.name}'
            self._routes[qualified] = (client, tool.name)
            self.tools.append({
                'type': 'function',
                'function': {
                    'name': qualified,
                    'description': tool.description or '',
                    'parameters': tool.input_schema,   # MCP 給的就是 JSON Schema
                },
            })
        return [t.name for t in listing.tools]

    async def call(self, qualified_name: str, arguments: dict) -> str:
        route = self._routes.get(qualified_name)
        if route is None:
            return f'錯誤：沒有名為 {qualified_name} 的工具，可用的有 {list(self._routes)}'

        client, mcp_name = route
        result = await client.call_tool(mcp_name, arguments)

        # 工具內部丟例外不會在這裡 raise，而是回傳 is_error=True。
        # 把錯誤訊息原樣交給模型，它通常會自己換個做法重試。
        text = self._blocks_to_text(result)
        return f'工具執行失敗：{text}' if result.is_error else text

    @staticmethod
    def _blocks_to_text(result) -> str:
        """CallToolResult.content 是一串 block，這裡只取得出文字的部分。"""
        parts = []
        for block in result.content:
            text = getattr(block, 'text', None)
            parts.append(text if text is not None
                         else f'[{getattr(block, "type", "unknown")} 類型的內容，此 Agent 未處理]')
        if not parts and getattr(result, 'structured_content', None) is not None:
            return str(result.structured_content)
        return '\n'.join(parts) or '(工具沒有回傳內容)'

    async def aclose(self) -> None:
        await self._stack.aclose()
```

三個設計重點：

- **`AsyncExitStack`**：MCP 連線是 async context manager，每台 Server 的生命週期都得管。用 `AsyncExitStack` 就能連任意多台，最後一次 `aclose()` 全部收乾淨。
- **工具名稱加前綴**（`tickets__get_ticket`）：接兩台以上 Server 時，同名工具幾乎一定會撞——`search`、`list`、`get` 這種名字太常見了。前綴是必須的，不是潔癖。
- **`content` 是 block 陣列**：MCP 的回傳可能是文字、圖片、embedded resource。這裡只處理文字，其他型別回一個佔位訊息，至少模型知道「有東西但我看不到」，而不是靜默吃掉。

## 4. Agent 主體

因為 MCP SDK 是 async 的，Agent Loop 也要換成 `AsyncClient`。除此之外，結構跟[〈實戰〉](../README.md#8-實戰一個會讀專案的-codebase-agent)一模一樣：

```python
from ollama import AsyncClient

async def run_agent(bridge: MCPToolBridge, question: str, max_turns: int = 12) -> str:
    client = AsyncClient(
        host='https://ollama.com',
        headers={'Authorization': 'Bearer ' + os.environ['OLLAMA_API_KEY']},
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = await client.chat(
            model=MODEL, messages=messages, tools=bridge.tools, think=True,
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        # 同一輪的多個工具呼叫可以平行送出，省下往返時間
        calls = list(response.message.tool_calls)
        results = await asyncio.gather(*(
            bridge.call(tc.function.name, tc.function.arguments) for tc in calls
        ))

        for tc, result in zip(calls, results):
            messages.append({
                'role': 'tool', 'tool_name': tc.function.name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'
```

**`asyncio.gather` 是換成 async 之後白賺的好處。** 模型一輪要求查三張工單時，同步版本會一張一張跑，async 版本一次打完。工具是 I/O bound（HTTP、資料庫、子行程）時差距很明顯。

接上 Server 並啟動：

```python
bridge = MCPToolBridge()
try:
    names = await bridge.connect(
        'tickets',
        stdio_client(StdioServerParameters(
            command=sys.executable, args=['examples/mcp_server_demo.py'],
        )),
    )
    print(f'已連上 MCP server「tickets」，工具：{names}')
    answer = await run_agent(bridge, question)
    print(answer)
finally:
    await bridge.aclose()
```

跑跑看：

```bash
pip install mcp
python examples/08_mcp_agent.py "有哪些還沒關掉的工單？最急的那張在講什麼？"
```

輸出大致是：

```
已連上 MCP server「tickets」，工具：['list_tickets', 'get_ticket', 'add_comment']
[MCP] tickets__list_tickets({'status': 'open'})
  → T-101 [open/high] 結帳頁在 Safari 會卡住 — 負責人 alice ...
[MCP] tickets__get_ticket({'ticket_id': 'T-101'})
  → 編號：T-101 標題：結帳頁在 Safari 會卡住 ...
============================================================
目前有兩張未關閉的工單...
```

**注意這裡沒有任何一行是我們寫的工具邏輯。** Agent 的工具全部來自 MCP Server，而那台 Server 可以是別人寫的、跑在別台機器上的。

## 5. 接現成的 MCP Server

自己寫的能接，別人的當然也能。`connect()` 的第二個參數換掉就好。

**本地 stdio server**（官方的檔案系統 Server，用 npx 跑）：

```python
await bridge.connect('fs', stdio_client(StdioServerParameters(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-filesystem', '/path/to/project'],
)))
```

**需要金鑰的 server**：透過 `env` 傳，不要寫進 args：

```python
await bridge.connect('github', stdio_client(StdioServerParameters(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-github'],
    env={'GITHUB_PERSONAL_ACCESS_TOKEN': os.environ['GITHUB_TOKEN']},
)))
```

**遠端 HTTP server**：`Client` 會從字串型別自動判斷用 streamable HTTP transport，直接給 URL 就行：

```python
await bridge.connect('notion', 'https://your-mcp-host/mcp')
```

要帶認證 header 的話，就自己組 transport：

```python
import httpx2
from mcp.client.streamable_http import streamable_http_client

async with httpx2.AsyncClient(
    headers={'Authorization': f'Bearer {token}'},
    timeout=httpx2.Timeout(30.0, read=300.0),
) as http_client:
    transport = streamable_http_client('https://your-mcp-host/mcp', http_client=http_client)
    await bridge.connect('notion', transport)
```

## 6. 接 MCP 之後才會遇到的坑

**工具數量會爆炸，而且要花錢。** 一台 GitHub Server 可能就給你 30 個工具，接三台就快 100 個。這些工具的完整 schema **每一輪**都要送給模型，是實打實的 context 成本，而且太多選項會讓模型挑錯工具。實務做法是**篩選**——只註冊這個 Agent 真正需要的：

```python
ALLOWED = {'tickets': {'list_tickets', 'get_ticket'}}   # 唯讀，不給 add_comment

for tool in listing.tools:
    if tool.name not in ALLOWED.get(server_name, {tool.name}):
        continue
    # ... 註冊
```

**寫入型工具要有關卡。** MCP Server 提供什麼工具是它決定的，裡面很可能有 `delete_file`、`close_issue`、`send_message`。在 `bridge.call()` 加一道確認：

```python
DESTRUCTIVE = {'fs__write_file', 'github__create_issue', 'tickets__add_comment'}

async def call(self, qualified_name, arguments):
    if qualified_name in DESTRUCTIVE:
        answer = input(f'\n⚠ Agent 要執行 {qualified_name}({arguments})，允許嗎？[y/N] ')
        if answer.strip().lower() != 'y':
            return '使用者拒絕了這次操作，請改用其他方式或直接回報無法完成。'
    ...
```

被拒絕時**要回一句話給模型**，不要回空字串。模型收到「使用者拒絕了」會轉而說明狀況；收到空字串則常常會傻傻地再試一次。

**第三方 Server 的描述文字是不可信輸入。** 工具的 `description` 會原封不動進入你的 prompt。如果那台 Server 不是你的，等於讓外部來源往你的 context 裡塞字——這是 prompt injection 的標準入口。接不熟的 Server 前先自己 `list_tools()` 看一遍它寫了什麼。

**stdio server 是子行程。** 它會繼承你的環境變數、以你的權限跑。`npx -y` 那行是在下載並執行別人的程式碼，該有的警覺不能少。

---

---

[← 用 OpenAI SDK 相容層接既有生態系](03-openai-compat.md)　·　[全部進階主題](README.md)　·　[實際使用情境 →](05-cost.md)
