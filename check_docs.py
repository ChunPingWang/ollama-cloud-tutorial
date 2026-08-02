"""文件一致性檢查：連結、錨點、檔案引用、範例檔頭。

拆成 README + docs/ 之後，跨檔連結變多，手動檢查不可靠。
每次改文件後跑一次：

    python check_docs.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
MD_FILES = [ROOT / 'README.md'] + sorted((ROOT / 'docs').glob('*.md'))

GREEN, RED, GREY, RESET = '\033[32m', '\033[31m', '\033[90m', '\033[0m'


def strip_code(text: str) -> str:
    """把程式碼區塊挖掉。

    不這樣做的話，Python 的 AVAILABLE[name](**kwargs) 會被
    Markdown 的連結正規式誤判成 [文字](網址)。
    """
    return re.sub(r'```.*?```', '', text, flags=re.S)


def anchors_of(text: str) -> set[str]:
    """GitHub 的錨點規則：小寫、空白換連字號、去掉標點。"""
    found = set()
    for heading in re.findall(r'^#{1,6} (.+)$', text, re.M):
        a = heading.strip().lower().replace(' ', '-')
        a = re.sub(r'[^\w一-鿿\-]', '', a)
        found.add(a)
    return found


def main() -> int:
    problems: list[str] = []
    contents = {f: f.read_text(encoding='utf-8') for f in MD_FILES}
    prose = {f: strip_code(t) for f, t in contents.items()}
    anchors = {f: anchors_of(t) for f, t in contents.items()}

    # 1) 內部連結（含跨檔 + 錨點）
    for f, text in prose.items():
        for link in re.findall(r'\]\(([^)]+)\)', text):
            if link.startswith(('http://', 'https://', 'mailto:')):
                continue
            path_part, _, frag = link.partition('#')
            target = f if not path_part else (f.parent / path_part).resolve()
            if not target.exists():
                problems.append(f'{f.name}: 連結指向不存在的檔案 → {link}')
                continue
            if frag and target in anchors and frag not in anchors[target]:
                problems.append(f'{f.name}: 錨點失效 → {link}')

    # 2) 文中提到的範例檔案是否存在
    for f, text in contents.items():
        for ref in set(re.findall(r'examples/[\w/]+\.(?:py|md|json)', text)):
            if not (ROOT / ref).exists():
                problems.append(f'{f.name}: 範例檔不存在 → {ref}')

    # 3) 每支範例都要有學習檔頭
    for py in sorted((ROOT / 'examples').rglob('*.py')):
        text = py.read_text(encoding='utf-8')
        if '【這支教你什麼' not in text and '【不用單獨執行' not in text \
           and 'fake_langfuse' not in py.name:
            problems.append(f'範例缺學習檔頭 → {py.relative_to(ROOT)}')

    # 4) 每支範例都要被文件提到
    all_md = '\n'.join(contents.values())
    for py in sorted((ROOT / 'examples').rglob('*.py')):
        if py.name not in all_md:
            problems.append(f'範例未被任何文件提及 → {py.relative_to(ROOT)}')

    # 5) 進階文件都要有導覽列
    for f in MD_FILES:
        if f.parent.name == 'docs' and f.name != 'README.md':
            if '全部進階主題' not in contents[f]:
                problems.append(f'{f.name}: 缺導覽列')

    # 6) 不該再有舊的「第 N 節」殘留（拆檔後應該都是連結）
    for f, text in contents.items():
        for m in re.findall(r'第 (\d+) 節', text):
            n = int(m)
            if n > 8:
                problems.append(f'{f.name}: 殘留舊章節引用「第 {n} 節」')

    total = sum(len(t.splitlines()) for t in contents.values())
    print(f'{len(MD_FILES)} 個 Markdown 檔，共 {total} 行')
    print(f'{GREY}  README {len(contents[ROOT / "README.md"].splitlines())} 行'
          f'　docs/ {len(MD_FILES) - 1} 篇{RESET}')

    if problems:
        print(f'\n{RED}發現 {len(problems)} 個問題：{RESET}')
        for p in problems:
            print(f'  ✗ {p}')
        return 1

    print(f'\n{GREEN}✅ 全部檢查通過{RESET}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
