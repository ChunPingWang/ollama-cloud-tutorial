"""結構化輸出：兩階段模式。

階段一用 tools 蒐集資訊，階段二單獨呼叫一次帶 format 的請求做總結。
format 和 tools 不要同時開，會互相干擾。

用法:
    python examples/05_structured_output.py
"""

from pydantic import BaseModel, Field

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
    severity: str = Field(description='嚴重程度：high / medium / low')
    issue: str = Field(description='問題描述')


class ReviewReport(BaseModel):
    summary: str = Field(description='整體評估，兩句話以內')
    findings: list[Finding] = Field(description='發現的問題清單')


# --- 階段一：（示意）用 tools 蒐集資訊 -------------------------------------
# 實務上這裡會是 04_codebase_agent.py 的 run_agent()，把讀到的檔案內容
# 累積成 collected 這段脈絡。這裡為了範例可獨立執行，直接用寫死的程式碼。
collected = f'檔案 payment.py 的內容：\n{SAMPLE_CODE}'


# --- 階段二：單獨呼叫，用 format 約束輸出 ----------------------------------
response = client.chat(
    model=MODEL,
    messages=[{
        'role': 'user',
        'content': f'請審查以下程式碼並列出問題：\n\n{collected}',
    }],
    format=ReviewReport.model_json_schema(),
    options={'temperature': 0},          # 結構化輸出不需要創意
)

report = ReviewReport.model_validate_json(response.message.content)

print(f'摘要：{report.summary}\n')
for f in report.findings:
    print(f'[{f.severity:>6}] {f.file}:{f.line} — {f.issue}')
