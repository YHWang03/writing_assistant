"""文件读写工具
- ReadFileTool: 读取文件文本
- WriteFileTool: 写入文本文件
- ListFilesTool: 列出目录内容
- DeleteFileTool: 按 glob 模式删除文件（如清理 LaTeX 辅助文件）
"""

import glob as _glob
from ..base import Tool
from ._safe_path import safe_resolve, _PROJECT_ROOT

# 进程级文件内容缓存，避免同一文件被多个 Agent 重复读取
_file_cache: dict[str, str] = {}

# .tex 论文单次写入上限（字符）：超过则拒绝一次性覆盖，强制分段（先 write 再 append）
_MAX_TEX_WRITE_CHARS = 8000

# 二进制/不可按 UTF-8 读取的文件后缀（read_file 只读文本，这些交给专职工具）
_BINARY_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".gz", ".tar", ".7z", ".pyc", ".bin",
}


class ReadFileTool(Tool):
    """读取文件文本"""

    def __init__(self):
        super().__init__(
            name="read_file",
            description="读取文件内容。支持两种模式：head（默认，从开头读取）和 tail（从末尾向前读取）。"
                        "tail 模式适合读取文件末尾内容（如检查 LaTeX 编译日志、确认文件完整性），无需手动计算 offset。"
        )
        self._agent_name: str = ""

    def set_agent_name(self, name: str):
        """设置调用此工具的 Agent 名称，用于拦截 .bib 读取"""
        self._agent_name = name

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
                "max_chars": {"type": "integer", "description": "最大字符数", "default": 50000},
                "offset": {
                    "type": "integer",
                    "description": "起始偏移量（字符数）。仅在 mode='head' 下生效，默认从文件开头读取",
                    "default": 0,
                },
                "mode": {
                    "type": "string",
                    "enum": ["head", "tail"],
                    "description": "读取模式：head=从开头/offset 读取（默认），tail=从文件末尾向前读取 max_chars 字符。tail 模式下 offset 参数无效",
                    "default": "head",
                },
            },
            "required": ["file_path"],
        }

    def execute(self, file_path: str, max_chars: int = 50000, offset: int = 0,
                mode: str = "head") -> str:
        try:
            path = safe_resolve(file_path)
        except ValueError as e:
            return f"Error: {e}"

        # soft warning：提示但仍允许读取 .bib 文件
        warning = ""
        if path.suffix == ".bib" and self._agent_name == "WritingAgent":
            warning = (
                "提示：文献数据可通过 read_context 获取，但你仍可读取此文件。\n\n"
            )

        if not path.exists():
            return f"Error: 文件不存在: {file_path}"
        if path.is_dir():
            return f"Error: '{file_path}' 是一个目录，请指定具体文件路径"

        # 二进制文件不能按文本读取，直接指路到专职工具，避免抛原始编码异常
        if path.suffix.lower() in _BINARY_SUFFIXES:
            if path.suffix.lower() == ".pdf":
                return (
                    f"提示：{file_path} 是 PDF 文件，read_file 只读文本。\n"
                    "PDF 正文请用 get_paper_text（读取原文），元数据请用 parse_pdf / parse_and_store。"
                )
            return (
                f"提示：{file_path} 是二进制文件（{path.suffix}），read_file 只读文本，无法读取。"
            )

        try:
            cache_key = str(path.resolve())
            if cache_key in _file_cache:
                full_content = _file_cache[cache_key]
            else:
                full_content = path.read_text(encoding="utf-8")
                _file_cache[cache_key] = full_content
            total_len = len(full_content)

            if mode == "tail":
                start = max(0, total_len - max_chars)
                content = full_content[start:]
                if start > 0:
                    content = f"... [skipped first {start} chars, total {total_len} chars]\n\n" + content
                return warning + content
            else:
                content = full_content[offset:offset + max_chars]
                if total_len > offset + max_chars:
                    content += f"\n\n... [truncated, total {total_len} chars]"
                return warning + content
        except UnicodeDecodeError:
            return (
                f"提示：{file_path} 不是 UTF-8 文本文件（可能是二进制或非 UTF-8 编码）。\n"
                "PDF 请用 get_paper_text / parse_pdf；其他文本文件请确认编码后改用对应工具。"
            )
        except Exception as e:
            return f"Error: 读取失败 — {e}"


class WriteFileTool(Tool):
    """写入文本文件 — 支持三种模式: write(覆盖), append(追加), replace(局部替换)"""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="写入文本文件。支持三种模式："
                        "write=覆盖写入（默认）, append=追加到末尾, replace=查找old_text并替换为content。"
                        "局部修改时请优先使用 mode='replace'，避免重写整个文件。"
                        "注意：只有 LiteratureAgent 可以写入 .bib 文件，其他 Agent 请使用 add_reference 或请求 MasterAgent 调度 LiteratureAgent。"
        )
        self._agent_name: str = ""

    def set_agent_name(self, name: str):
        """设置调用此工具的 Agent 名称，用于拦截非 LiteratureAgent 写 .bib"""
        self._agent_name = name

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入或替换的内容"},
                "mode": {
                    "type": "string",
                    "enum": ["write", "append", "replace"],
                    "description": "写入模式: write=覆盖写入（默认）, append=追加到末尾, replace=查找old_text并替换为content",
                    "default": "write",
                },
                "old_text": {
                    "type": "string",
                    "description": "要替换的原文。仅 mode=replace 时需要，精确匹配后替换为 content",
                },
                "append": {
                    "type": "boolean",
                    "description": "是否追加模式（已弃用，请使用 mode='append'）。true=追加，false=覆盖写入",
                    "default": False,
                },
            },
            "required": ["file_path", "content"],
        }

    def execute(self, **kwargs) -> str:
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")
        mode = kwargs.get("mode", "write")
        old_text = kwargs.get("old_text")
        append = kwargs.get("append", False)

        # 向后兼容：append=True → mode="append"
        if append and mode == "write":
            mode = "append"

        missing = []
        if not file_path:
            missing.append("file_path（要写入的文件路径）")
        if content is None:
            missing.append("content（要写入/替换的内容）")
        if mode == "replace" and not old_text:
            missing.append("old_text（replace 模式要替换的原文）")
        if missing:
            return "Error: 缺少参数：" + "、".join(missing) + "。"

        try:
            path = safe_resolve(file_path)
        except ValueError as e:
            return f"Error: {e}"

        # 拦截非 LiteratureAgent 写 .bib 文件
        if path.suffix == ".bib" and self._agent_name != "LiteratureAgent":
            return (
                "Error: 只有 LiteratureAgent 可以写入 .bib 文件。\n"
                "请将缺失的文献信息报告给 MasterAgent，由 MasterAgent 调度 LiteratureAgent 处理。"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if mode == "replace":
                if not path.exists():
                    return f"Error: 文件不存在，无法替换: {file_path}"
                existing = path.read_text(encoding="utf-8")
                if old_text not in existing:
                    return (
                        f"Error: 未找到匹配文本。\n"
                        f"查找内容前150字符: {old_text[:150]}..."
                    )
                result = existing.replace(old_text, content, 1)
                path.write_text(result, encoding="utf-8")
                _file_cache.pop(str(path.resolve()), None)  # 写后清缓存
                return (
                    f"文件已替换: {file_path} ({len(result)} 字符, "
                    f"替换了 {len(old_text)} → {len(content)} 字符)"
                )

            elif mode == "append":
                if path.exists():
                    existing = path.read_text(encoding="utf-8")
                    content = existing + content
                path.write_text(content, encoding="utf-8")
                _file_cache.pop(str(path.resolve()), None)  # 写后清缓存
                return f"文件已追加: {file_path} ({len(content)} 字符)"

            else:  # mode == "write"
                # .tex 论文禁止一次性覆盖写入整篇：强制分段（先 write 再 append），
                # 避免模型用单个超大 write_file 反复失败（内容超输出预算被截断后重试同一件事）
                if path.suffix == ".tex" and len(content) > _MAX_TEX_WRITE_CHARS:
                    return (
                        f"Error: 内容过大（{len(content)} 字符），拒绝一次性覆盖写入整篇 .tex。\n"
                        f"请分段写入：先用 mode='write' 写文档头部（preamble + 第一个 section），"
                        f"再用 mode='append' 逐节追加后续内容；修改已有段落用 mode='replace'。"
                        f"单次写入请控制在 {_MAX_TEX_WRITE_CHARS} 字符以内。"
                    )
                path.write_text(content, encoding="utf-8")
                _file_cache.pop(str(path.resolve()), None)  # 写后清缓存
                return f"文件已写入: {file_path} ({len(content)} 字符)"

        except Exception as e:
            return f"Error: 写入失败 — {e}"


class ListFilesTool(Tool):
    """列出目录内容"""

    def __init__(self):
        super().__init__(
            name="ls",
            description="列出指定目录下的文件和子目录。输入目录路径，"
                        "返回文件列表（含文件大小）。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "目录路径。默认为项目根目录。",
                },
            },
            "required": [],
        }

    def execute(self, dir_path: str = "") -> str:
        if not dir_path:
            dir_path = str(_PROJECT_ROOT)
        try:
            path = safe_resolve(dir_path)
        except ValueError as e:
            return f"Error: {e}"
        if not path.exists():
            return f"Error: 目录不存在: {dir_path}"
        if not path.is_dir():
            return f"Error: '{dir_path}' 不是目录"
        try:
            items = []
            for entry in sorted(path.iterdir()):
                tag = "DIR" if entry.is_dir() else "FILE"
                size = ""
                if entry.is_file():
                    try:
                        s = entry.stat().st_size
                        if s < 1024:
                            size = f" ({s}B)"
                        elif s < 1024 * 1024:
                            size = f" ({s / 1024:.1f}KB)"
                        else:
                            size = f" ({s / 1024 / 1024:.1f}MB)"
                    except OSError:
                        pass
                items.append(f"  [{tag}] {entry.name}{size}")
            if not items:
                return f"目录为空: {dir_path}"
            return f"{dir_path} ({len(items)} 项):\n" + "\n".join(items)
        except Exception as e:
            return f"Error: 列出目录失败 — {e}"


# 默认 LaTeX 辅助文件 glob 模式
_LATEX_AUX_PATTERNS = [
    "*.aux", "*.dvi", "*.log", "*.toc", "*.bbl", "*.blg",
    "*.out", "*.fff", "*.lof", "*~",
]


class DeleteFileTool(Tool):
    """按 glob 模式删除文件，常用于清理 LaTeX 辅助文件"""

    def __init__(self):
        super().__init__(
            name="delete_files",
            description="按 glob 模式删除指定目录下的文件。"
                        "如果不指定 patterns，默认删除所有 LaTeX 辅助文件"
                        "（*.aux, *.dvi, *.log, *.toc, *.bbl, *.blg, *.out, *.fff, *.lof, *~）。"
                        "支持自定义 patterns 列表。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "要清理的目录路径。",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要删除的 glob 模式列表。"
                                   "不指定则使用默认 LaTeX 辅助文件模式。",
                },
            },
            "required": ["dir_path"],
        }

    def execute(self, dir_path: str, patterns: list[str] | None = None) -> str:
        try:
            path = safe_resolve(dir_path)
        except ValueError as e:
            return f"Error: {e}"
        if not path.exists():
            return f"Error: 目录不存在: {dir_path}"
        if not path.is_dir():
            return f"Error: '{dir_path}' 不是目录"

        patterns = patterns or _LATEX_AUX_PATTERNS

        deleted = []
        errors = []
        for pattern in patterns:
            for f in path.glob(pattern):
                if f.is_file():
                    try:
                        f.unlink()
                        deleted.append(f.name)
                        # 清除文件缓存（如果存在）
                        _file_cache.pop(str(f.resolve()), None)
                    except OSError as e:
                        errors.append(f"{f.name}: {e}")

        if not deleted and not errors:
            return f"未找到匹配的文件（patterns: {patterns})"

        result = f"已删除 {len(deleted)} 个文件:"
        for name in deleted:
            result += f"\n  ✓ {name}"
        if errors:
            result += f"\n{len(errors)} 个失败:"
            for err in errors:
                result += f"\n  ✗ {err}"
        return result