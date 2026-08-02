"""結構化輸出：在 Ollama Cloud 上真正可行的做法。

⚠ 重要：Ollama Cloud 的 hosted API 目前「不強制」結構化輸出。
   實測結果（2026-08，gpt-oss:120b / gemma4:31b）：
     - 原生 chat(format=<json schema>)         → 被忽略，回傳散文
     - 原生 chat(format='json')                 → 被忽略，回傳 ```json 圍籬
     - OpenAI 相容層 response_format=json_schema → 被忽略，回傳散文
   本地 ollama serve 支援 format，但雲端這條路走不通。

   可行的替代方案有兩個，本檔都示範：
     方案 A（推薦）：把 schema 包成一個工具，用 tool calling 拿結構化參數。
                    tool calling 在雲端是確實生效的，等於借它來做 schema 約束。
     方案 B（備援）：prompt 要求 JSON + 去除 ``` 圍籬 + Pydantic 驗證 + 重試。

用法:
    python examples/05_structured_output.py
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from _client import MODEL, get_client

client = get_client()

SAMPLE_CODE = '''
def transfer(from_acc, to_acc, amount):
    balance = db.query(f"SELECT balance FROM accounts WHERE id = {from_acc}")
    if balance >= amount:
        db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE id = {from_acc}")
        db.execute(f"UPDATE accounts SET balance = balance + {amount} WHERE id = {to_acc}")
    return True
'''


class Finding(BaseModel):
    file: str = Field(description='檔案路徑')
    line: int = Field(description='行號')
    severity: str = Field(description='嚴重程度，只能是 high、medium 或 low')
    issue: str = Field(description='問題描述')


class ReviewReport(BaseModel):
    summary: str = Field(description='整體評估，兩句話以內')
    findings: list[Finding] = Field(description='發現的問題清單')


# --------------------------------------------------------------------------
# 方案 A：用 tool calling 當結構化輸出（推薦）
# --------------------------------------------------------------------------

def structured_via_tool(prompt: str, schema_model: type[BaseModel],
                        attempts: int = 3) -> BaseModel:
    """把 Pydantic model 包成一個工具，逼模型以工具參數的形式交出結構化資料。"""
    tool = {
        'type': 'function',
        'function': {
            'name': 'submit',
            'description': f'提交最終結果。這是唯一的作答方式，必須呼叫。'
                           f'欄位說明見 parameters。',
            'parameters': schema_model.model_json_schema(),
        },
    }
    messages = [{'role': 'user', 'content': prompt}]

    for attempt in range(1, attempts + 1):
        response = client.chat(
            model=MODEL, messages=messages, tools=[tool],
            options={'temperature': 0},          # 結構化輸出不需要創意
        )

        if response.message.tool_calls:
            args = response.message.tool_calls[0].function.arguments
            try:
                return schema_model.model_validate(args)
            except ValidationError as exc:
                # 把驗證錯誤回饋給模型，讓它自己修
                messages.append(response.message)
                messages.append({
                    'role': 'tool', 'tool_name': 'submit',
                    'content': f'資料格式不符，請修正後重新呼叫 submit：{exc}',
                })
        else:
            messages.append(response.message)
            messages.append({
                'role': 'user',
                'content': '請不要用文字回答，必須呼叫 submit 工具提交結果。',
            })
        print(f'  (第 {attempt} 次未通過，重試中…)')

    raise RuntimeError(f'嘗試 {attempts} 次仍拿不到合法的結構化輸出')


# --------------------------------------------------------------------------
# 方案 B：prompt + 去圍籬 + 驗證（備援）
# --------------------------------------------------------------------------

def _strip_fence(text: str) -> str:
    """模型很愛把 JSON 包在 ```json ... ``` 裡，先拆掉。"""
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    return (match.group(1) if match else text).strip()


def structured_via_prompt(prompt: str, schema_model: type[BaseModel],
                          attempts: int = 3) -> BaseModel:
    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
    messages = [{
        'role': 'user',
        'content': f'{prompt}\n\n'
                   f'只輸出符合以下 JSON Schema 的 JSON，不要任何說明文字：\n{schema}',
    }]

    for attempt in range(1, attempts + 1):
        response = client.chat(model=MODEL, messages=messages,
                               options={'temperature': 0})
        raw = _strip_fence(response.message.content)
        try:
            return schema_model.model_validate_json(raw)
        except ValidationError as exc:
            messages.append(response.message)
            messages.append({
                'role': 'user',
                'content': f'剛才的輸出無法解析：{exc}\n請只輸出合法 JSON。',
            })
            print(f'  (第 {attempt} 次解析失敗，重試中…)')

    raise RuntimeError(f'嘗試 {attempts} 次仍拿不到合法 JSON')


# --------------------------------------------------------------------------

if __name__ == '__main__':
    # 實務上這段脈絡會來自 Agent Loop（04_codebase_agent.py）蒐集到的檔案內容。
    # 這裡為了範例可獨立執行，直接用寫死的程式碼。
    prompt = f'請審查以下程式碼並列出問題：\n\n檔案 payment.py：\n{SAMPLE_CODE}'

    for label, fn in [('方案 A：tool calling', structured_via_tool),
                      ('方案 B：prompt + 驗證', structured_via_prompt)]:
        print(f'\n===== {label} =====')
        report = fn(prompt, ReviewReport)
        print(f'摘要：{report.summary}\n')
        for f in report.findings:
            print(f'[{f.severity:>6}] {f.file}:{f.line} — {f.issue}')
