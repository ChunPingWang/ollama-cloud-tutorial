"""用 LangChain 接 Ollama Cloud，並用 Langfuse 的 CallbackHandler 追蹤。

重點在 ChatOllama 的兩個參數：
    base_url      指到 https://ollama.com
    client_kwargs 塞 Authorization header

⚠ 已知地雷：with_structured_output() 在雲端會壞掉，原因跟第 8 節一樣
   （雲端不強制 format）。本檔第三段示範問題與解法。

用法:
    python examples/09_langchain_agent.py

要一併送 trace 到 Langfuse 的話，先設好 LANGFUSE_* 三個環境變數
（沒有帳號可用 examples/tools/fake_langfuse_server.py 驗證）。
"""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel

load_dotenv()

MODEL = os.environ.get('OLLAMA_MODEL', 'gpt-oss:120b')
CYAN, GREY, RESET = '\033[36m', '\033[90m', '\033[0m'


def build_llm() -> ChatOllama:
    """LangChain 接 Ollama Cloud 的關鍵就這三行。"""
    api_key = os.environ.get('OLLAMA_API_KEY')
    if not api_key:
        sys.exit('找不到 OLLAMA_API_KEY，請先到 https://ollama.com/settings/keys 建立。')
    return ChatOllama(
        model=MODEL,
        base_url='https://ollama.com',
        client_kwargs={'headers': {'Authorization': f'Bearer {api_key}'}},
    )


def build_callbacks() -> list:
    """有設 Langfuse 金鑰就掛上 CallbackHandler，沒設就跳過。"""
    if not os.environ.get('LANGFUSE_PUBLIC_KEY'):
        print(f'{GREY}(未設定 LANGFUSE_PUBLIC_KEY，略過 trace){RESET}')
        return []
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler()]


@tool
def get_temperature(city: str) -> str:
    """查詢指定城市的目前氣溫。"""
    return {'Taipei': '31°C', 'Tokyo': '18°C', 'London': '15°C'}.get(city, '查無資料')


@tool
def get_conditions(city: str) -> str:
    """查詢指定城市的天氣狀況。"""
    return {'Taipei': '晴時多雲', 'Tokyo': '陰', 'London': '下雨'}.get(city, '查無資料')


TOOLS = [get_temperature, get_conditions]


# --------------------------------------------------------------------------

if __name__ == '__main__':
    llm = build_llm()
    callbacks = build_callbacks()
    config = {'callbacks': callbacks} if callbacks else {}

    print(f'{CYAN}=== 1. 基本呼叫 ==={RESET}')
    print(llm.invoke('用一句話說明什麼是 AI Agent', config=config).content)

    print(f'\n{CYAN}=== 2. bind_tools（工具呼叫在雲端可正常運作）==={RESET}')
    reply = llm.bind_tools(TOOLS).invoke('台北現在幾度？', config=config)
    print('tool_calls:', reply.tool_calls)

    print(f'\n{CYAN}=== 3. create_agent（LangChain 幫你跑完 Agent Loop）==={RESET}')
    agent = create_agent(
        model=llm, tools=TOOLS,
        system_prompt='你是氣象助理，必須透過工具查詢，不可自行編造數據。回答用繁體中文。',
    )
    # Ollama Cloud 實測會偶發 500，Agent 跑越多輪撞上的機率越高。
    # ChatOllama 沒有 max_retries，但整個 agent 是 Runnable，可以直接包 with_retry。
    resilient_agent = agent.with_retry(stop_after_attempt=3)
    result = resilient_agent.invoke(
        {'messages': [{'role': 'user', 'content': '台北和東京現在的氣溫與天氣如何？哪個比較適合出門？'}]},
        config=config,
    )
    # 注意：中途的 AIMessage 只帶 tool_calls、content 是空字串，
    # 真正的答案在最後一則 content 非空的 AIMessage 上。
    for message in result['messages']:
        kind = type(message).__name__
        if getattr(message, 'tool_calls', None):
            print(f'{GREY}  [{kind}] → 呼叫 {[tc["name"] for tc in message.tool_calls]}{RESET}')
        elif kind == 'ToolMessage':
            print(f'{GREY}  [{kind}] ← {message.content}{RESET}')

    final = next((m.content for m in reversed(result['messages'])
                  if type(m).__name__ == 'AIMessage' and m.content), '(沒有產生最終回覆)')
    print('\n' + final)

    print(f'\n{CYAN}=== 4. with_structured_output 的雲端地雷 ==={RESET}')

    class Country(BaseModel):
        name: str
        capital: str

    # 預設走 JSON mode，而雲端不強制 format → OutputParserException
    try:
        print('預設模式：', llm.with_structured_output(Country).invoke('Tell me about Canada.'))
    except Exception as exc:                          # noqa: BLE001
        print(f'預設模式：❌ {type(exc).__name__}: {str(exc)[:90]}…')

    # 解法：明講走 function_calling，借 tool calling 做約束（同第 8 節方案 A）
    structured = llm.with_structured_output(Country, method='function_calling')
    print('function_calling：',
          structured.invoke('Tell me about Canada. 請務必呼叫工具回報結果。'))

    if callbacks:
        from langfuse import get_client
        get_client().flush()
        print(f'\n{GREY}trace 已送出{RESET}')
