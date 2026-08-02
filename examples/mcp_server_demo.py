"""示範用的 MCP Server：一個假的工單系統。

【這支教你什麼】MCP Server 怎麼寫（給 08 當子行程用，不單獨執行）
【前置知識】—
【下一支】08_mcp_agent.py

這支程式不是給人直接跑的，是給 08_mcp_agent.py 當成子行程啟動。
想單獨檢查它有哪些工具，可以用官方 Inspector：

    uv run mcp dev examples/mcp_server_demo.py
"""

from mcp.server import MCPServer

mcp = MCPServer('TicketSystem')

# 假資料，實務上這裡會是資料庫或內部 API
TICKETS = {
    'T-101': {
        'title': '結帳頁在 Safari 會卡住',
        'status': 'open',
        'priority': 'high',
        'assignee': 'alice',
        'comments': ['使用者回報只有 iOS 18 會發生'],
    },
    'T-102': {
        'title': '匯出 CSV 中文變亂碼',
        'status': 'open',
        'priority': 'medium',
        'assignee': 'bob',
        'comments': [],
    },
    'T-103': {
        'title': '登入頁的忘記密碼連結壞掉',
        'status': 'closed',
        'priority': 'low',
        'assignee': 'alice',
        'comments': ['已於 v2.3.1 修復'],
    },
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
        rows.append(f"{tid} [{t['status']}/{t['priority']}] {t['title']} — 負責人 {t['assignee']}")
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
    comments = '\n'.join(f'  - {c}' for c in t['comments']) or '  (無留言)'
    return (
        f"編號：{ticket_id}\n"
        f"標題：{t['title']}\n"
        f"狀態：{t['status']}\n"
        f"優先度：{t['priority']}\n"
        f"負責人：{t['assignee']}\n"
        f"留言：\n{comments}"
    )


@mcp.tool()
def add_comment(ticket_id: str, comment: str) -> str:
    """在指定工單上新增一則留言。

    Args:
        ticket_id: 工單編號，例如 T-101
        comment: 留言內容
    """
    t = TICKETS.get(ticket_id)
    if t is None:
        raise ValueError(f'找不到工單 {ticket_id}')
    t['comments'].append(comment)
    return f'已在 {ticket_id} 新增留言，目前共 {len(t["comments"])} 則'


if __name__ == '__main__':
    mcp.run()   # 預設走 stdio，由呼叫端以子行程啟動
