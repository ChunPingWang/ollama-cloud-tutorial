"""把 MCP Server 的工具接進 Ollama Cloud Agent。

Ollama 沒有內建 MCP 支援，所以中間需要一層橋接：
  MCP 的 list_tools() → 轉成 Ollama 的 tools schema → 模型決定呼叫 →
  轉回 MCP 的 call_tool() → 結果塞回 messages。

兩邊的參數描述都是 JSON Schema，所以轉換其實只是搬欄位。

用法:
    python examples/08_mcp_agent.py "有哪些還沒關掉的工單？最急的那張在講什麼？"
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

load_dotenv()

MODEL = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')
GREY, CYAN, RED, RESET = '\033[90m', '\033[36m', '\033[31m', '\033[0m'


# --------------------------------------------------------------------------
# 橋接層
# --------------------------------------------------------------------------

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
        """連上一台 MCP Server 並註冊它的工具。

        transport 可以是 stdio_client(...) 的結果，也可以直接給
        streamable HTTP 的 URL 字串。
        """
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
        """執行一個工具呼叫，回傳給模型看的純文字結果。"""
        route = self._routes.get(qualified_name)
        if route is None:
            return f'錯誤：沒有名為 {qualified_name} 的工具，可用的有 {list(self._routes)}'

        client, mcp_name = route
        result = await client.call_tool(mcp_name, arguments)

        # MCP 的慣例：工具內部丟例外不會在這裡 raise，而是回傳 is_error=True。
        # 把錯誤訊息原樣交給模型，它通常會自己換個做法重試。
        text = self._blocks_to_text(result)
        return f'工具執行失敗：{text}' if result.is_error else text

    @staticmethod
    def _blocks_to_text(result) -> str:
        """CallToolResult.content 是一串 block，這裡只取得出文字的部分。"""
        parts = []
        for block in result.content:
            text = getattr(block, 'text', None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(f'[{getattr(block, "type", "unknown")} 類型的內容，此 Agent 未處理]')
        if not parts and getattr(result, 'structured_content', None) is not None:
            return str(result.structured_content)
        return '\n'.join(parts) or '(工具沒有回傳內容)'

    async def aclose(self) -> None:
        await self._stack.aclose()


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一個工單系統助理，只能透過提供的工具存取資料。

工作方式：
- 需要總覽時先用 list 類工具，需要細節再針對特定項目查詢
- 只根據工具回傳的內容回答，不要臆測沒查到的資料
- 資訊足夠時直接給結論，不要再繼續呼叫工具
- 除非使用者明確要求，否則不要呼叫會修改資料的工具

回答請用繁體中文。"""


def _get_ollama_client() -> AsyncClient:
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        sys.exit('找不到 OLLAMA_API_KEY，請先到 https://ollama.com/settings/keys 建立。')
    return AsyncClient(
        host='https://ollama.com',
        headers={'Authorization': 'Bearer ' + api_key},
    )


async def run_agent(bridge: MCPToolBridge, question: str, max_turns: int = 12) -> str:
    client = _get_ollama_client()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = await client.chat(
            model=MODEL, messages=messages, tools=bridge.tools, think=True,
        )
        messages.append(response.message)

        if response.message.thinking:
            print(f'\n{GREY}[第 {turn} 輪思考] '
                  f'{response.message.thinking.strip()[:300]}{RESET}')

        if not response.message.tool_calls:
            return response.message.content

        # 同一輪的多個工具呼叫可以平行送出，省下往返時間
        calls = list(response.message.tool_calls)
        for tc in calls:
            print(f'{CYAN}[MCP] {tc.function.name}({tc.function.arguments}){RESET}')

        results = await asyncio.gather(*(
            bridge.call(tc.function.name, tc.function.arguments) for tc in calls
        ))

        for tc, result in zip(calls, results):
            colour = RED if result.startswith('工具執行失敗') else GREY
            print(f'{colour}  → {result.replace(chr(10), " ")[:120]}{RESET}')
            messages.append({
                'role': 'tool', 'tool_name': tc.function.name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'


async def main() -> None:
    question = ' '.join(sys.argv[1:]) or '有哪些還沒關掉的工單？最急的那張在講什麼？'

    bridge = MCPToolBridge()
    try:
        # 本地 stdio server：由我們自己啟動成子行程
        server_script = os.path.join(os.path.dirname(__file__), 'mcp_server_demo.py')
        names = await bridge.connect(
            'tickets',
            stdio_client(StdioServerParameters(
                command=sys.executable, args=[server_script],
            )),
        )
        print(f'{CYAN}已連上 MCP server「tickets」，工具：{names}{RESET}')

        # 要再接第二台就多呼叫一次 connect，遠端 HTTP server 直接給 URL：
        # await bridge.connect('github', 'https://your-mcp-host/mcp')

        print(f'{CYAN}問題：{question}{RESET}')
        answer = await run_agent(bridge, question)
        print('\n' + '=' * 60)
        print(answer)
    finally:
        await bridge.aclose()


if __name__ == '__main__':
    asyncio.run(main())
