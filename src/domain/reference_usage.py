"""Conservative, read-only citation checks before deleting library entries."""
from pathlib import Path
import re
from .citation_syntax import PROTECTED_CITATION, citation_keys


def _without_comments(text):
    '''删除未转义的 TeX 行注释，保留行数。

    paras:
        text: 待解析或扫描的文本。
    return: 去除注释后的文本。
    '''
    lines = []
    for line in text.splitlines(keepends=True):
        for i, char in enumerate(line):
            if char == '%' and (len(line[:i]) - len(line[:i].rstrip('\\'))) % 2 == 0:
                line = line[:i] + ('\n' if line.endswith('\n') else '')
                break
        lines.append(line)
    return ''.join(lines)


def assert_unreferenced(cite_key, main_tex_path='', output_dir='', sections=None):
    '''确认文献未被正文及章节引用；发现引用或无法可靠读取正文时抛出 ValueError。

    检查常见 cite、nocite 和递归包含文件，不展开自定义宏或条件分支。

    paras:
        cite_key: 准备删除的文献引用键。
        main_tex_path: 主 TeX 路径；为空时尝试 output_dir 下已有的 main.tex。
        output_dir: 查找主文件及解析相对包含路径的备用目录。
        sections: 章节名到正文的映射；None 表示没有额外章节。
    '''
    used, visited = [], set()
    include = re.compile(r'\\(?:input|include|subfile)\s*(?:\{([^}]+)\}|([^\s%{}]+))')

    def scan(text, label, base):
        '''扫描文本中的引用并递归检查包含文件，累积文献使用位置。

        paras:
            text: 待解析或扫描的文本。
            label: 用于标记引用位置的文件名或章节标识。
            base: 解析 TeX 包含文件路径的基准目录。
        '''
        text = _without_comments(text)
        for match in PROTECTED_CITATION.finditer(text):
            keys = set(citation_keys(match))
            if cite_key in keys or ('*' in keys and match[0].startswith('\\nocite')):
                used.append(f"{label}:{text.count(chr(10), 0, match.start()) + 1}")
        for match in include.finditer(text):
            target = (match[1] or match[2]).strip()
            if base is None or any(c in target for c in ('\\', '#', '$')):
                raise ValueError(f"无法确定被引用文件路径，拒绝删除文献：{label} -> {target}")
            path = base / target
            if not path.suffix:
                path = path.with_suffix('.tex')
            read(path, base)

    def read(path, base):
        '''读取尚未访问的 TeX 文件并继续扫描，限制递归文件数且拒绝不可读输入。

        paras:
            path: 目标文件或配置项路径。
            base: 解析 TeX 包含文件路径的基准目录。
        '''
        path = path.resolve()
        if path in visited:
            return
        if len(visited) >= 100:
            raise ValueError('引用检查文件数超过100，拒绝删除文献')
        visited.add(path)
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"无法读取正文，拒绝删除文献：{path} ({exc})") from exc
        scan(text, str(path), base)

    main = Path(main_tex_path) if main_tex_path else None
    if main is None and output_dir:
        candidate = Path(output_dir) / 'main.tex'
        if candidate.exists():
            main = candidate
    base = main.resolve().parent if main is not None else (Path(output_dir).resolve() if output_dir else None)
    if main is not None:
        read(main, base)
    for name, text in (sections or {}).items():
        scan(text, f'section[{name}]', base)
    if used:
        raise ValueError(f"文献 {cite_key} 仍被引用，禁止删除。请先修改正文引用。位置：" + '；'.join(used))
