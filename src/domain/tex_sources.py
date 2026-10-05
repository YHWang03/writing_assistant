"""读取论文的静态 TeX 文件树，避免遗漏章节中的未知引用。"""
import re
from pathlib import Path


def read_tex_sources(tex_path):
    '''读取主文件及静态 input/include/subfile，拒绝越界、动态路径和循环包含。

    paras:
        tex_path: 主 TeX 文件路径；子文件须位于主文件目录树内。
    return: 绝对路径到去除行注释后文本的映射；无法完整读取时抛出异常。
    '''
    root = Path(tex_path).resolve().parent
    sources, active = {}, set()
    include = re.compile(r'\\(?:input|include|subfile)(?![a-zA-Z])\s*(?:\{([^}]+)\}|([^\s%{}]+))')

    def read(path):
        '''递归读取单个 TeX 文件并检查其包含关系。

        paras:
            path: 待读取文件的 Path。
        '''
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'章节路径越界：{path}')
        if path in active:
            raise ValueError(f'循环包含章节：{path}')
        if str(path) in sources:
            return
        if len(sources) >= 100:
            raise ValueError('TeX 包含文件超过 100 个，无法完成检查')
        active.add(path)
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text(encoding='utf-8'))
        sources[str(path)] = text
        for match in include.finditer(text):
            target = (match[1] or match[2]).strip()
            if any(char in target for char in ('\\', '#', '$')):
                raise ValueError(f'无法解析动态章节路径：{target}')
            child = root / target
            if not child.suffix:
                child = child.with_suffix('.tex')
            read(child)
        active.remove(path)

    read(Path(tex_path))
    return sources
