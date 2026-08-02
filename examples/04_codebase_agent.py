"""實戰：會自己翻專案、讀檔、搜尋，然後回答問題的 Codebase Agent。

【這支教你什麼】真實 Agent 需要的防護：路徑邊界、輸出截斷、工具例外處理
【前置知識】03
【下一支】05_structured_output.py：讓輸出變成下游能用的資料

用法:
    export AGENT_ROOT=/path/to/your/project
    python examples/04_codebase_agent.py "這個專案的進入點在哪裡？"
"""

import os
import sys
from pathlib import Path

from _client import MODEL, get_client

client = get_client()

ROOT = Path(os.environ.get('AGENT_ROOT', '.')).resolve()
MAX_FILE_BYTES = 40_000
MAX_SEARCH_HITS = 50

GREY, CYAN, RESET = '\033[90m', '\033[36m', '\033[0m'


# --------------------------------------------------------------------------
# 工具定義
# --------------------------------------------------------------------------

def _safe_path(relative_path: str) -> Path:
    """把相對路徑解析成絕對路徑，並確保沒有跳出 ROOT。"""
    target = (ROOT / relative_path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f'路徑超出允許範圍：{relative_path}')
    return target


def list_files(relative_path: str = '.') -> str:
    """列出專案中某個目錄底下的檔案與子目錄。

    Args:
        relative_path: 相對於專案根目錄的路徑，預設為根目錄本身

    Returns:
        每行一個項目，目錄結尾帶 /
    """
    target = _safe_path(relative_path)
    if not target.is_dir():
        return f'{relative_path} 不是目錄'
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith('.'):
            continue
        entries.append(f'{p.name}/' if p.is_dir() else p.name)
    return '\n'.join(entries) or '(空目錄)'


def read_file(relative_path: str) -> str:
    """讀取專案中某個檔案的完整內容。

    Args:
        relative_path: 相對於專案根目錄的檔案路徑

    Returns:
        檔案內容，過長時會被截斷
    """
    target = _safe_path(relative_path)
    if not target.is_file():
        return f'{relative_path} 不存在或不是檔案'
    data = target.read_text(encoding='utf-8', errors='replace')
    if len(data) > MAX_FILE_BYTES:
        data = data[:MAX_FILE_BYTES] + '\n... (內容過長已截斷)'
    return data


def search_code(pattern: str, relative_path: str = '.') -> str:
    """在專案中以純文字關鍵字搜尋，回傳符合的檔案與行號。

    Args:
        pattern: 要搜尋的關鍵字（不分大小寫）
        relative_path: 搜尋範圍，預設整個專案

    Returns:
        每行格式為 檔案路徑:行號: 該行內容，最多 50 筆
    """
    root = _safe_path(relative_path)
    needle = pattern.lower()
    hits = []
    for p in root.rglob('*'):
        if not p.is_file() or any(part.startswith('.') for part in p.parts):
            continue
        try:
            for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
                if needle in line.lower():
                    hits.append(f'{p.relative_to(ROOT)}:{i}: {line.strip()[:160]}')
                    if len(hits) >= MAX_SEARCH_HITS:
                        return '\n'.join(hits) + '\n... (結果過多已截斷)'
        except (UnicodeDecodeError, PermissionError, OSError):
            continue
    return '\n'.join(hits) or f'找不到符合 "{pattern}" 的內容'


TOOLS = [list_files, read_file, search_code]
AVAILABLE = {fn.__name__: fn for fn in TOOLS}


# --------------------------------------------------------------------------
# Agent 主體
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一個程式碼庫分析助手。

工作方式：
- 先用 list_files 建立對專案結構的認識，再決定要讀哪些檔案
- 需要找特定符號或設定時，用 search_code 比逐檔讀取有效率
- 只根據實際讀到的檔案內容回答，不要臆測沒看過的程式碼
- 蒐集到足夠資訊後，直接給出結論，不要再繼續呼叫工具

回答時請用繁體中文，並在提及程式碼時附上 檔案路徑:行號。"""


def run_agent(question: str, max_turns: int = 12) -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]

    for turn in range(1, max_turns + 1):
        response = client.chat(
            model=MODEL, messages=messages, tools=TOOLS, think=True,
        )
        messages.append(response.message)

        if response.message.thinking:
            print(f'\n{GREY}[第 {turn} 輪思考] '
                  f'{response.message.thinking.strip()[:300]}{RESET}')

        if not response.message.tool_calls:
            return response.message.content

        for tc in response.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments
            print(f'{CYAN}[工具] {name}({args}){RESET}')
            fn = AVAILABLE.get(name)
            if fn is None:
                result = f'錯誤：沒有名為 {name} 的工具，可用的有 {list(AVAILABLE)}'
            else:
                try:
                    result = fn(**args)
                except Exception as exc:                  # noqa: BLE001
                    result = f'工具執行失敗：{type(exc).__name__}: {exc}'
            preview = str(result).replace('\n', ' ')[:120]
            print(f'{GREY}  → {preview}{RESET}')
            messages.append({
                'role': 'tool', 'tool_name': name, 'content': str(result),
            })

    return '（已達最大輪數上限，任務未完成）'


if __name__ == '__main__':
    question = (
        ' '.join(sys.argv[1:])
        or '這個專案是做什麼的？進入點在哪裡？主要模組怎麼切的？'
    )
    print(f'{CYAN}分析目錄：{ROOT}{RESET}')
    print(f'{CYAN}問題：{question}{RESET}')
    answer = run_agent(question)
    print('\n' + '=' * 60)
    print(answer)
