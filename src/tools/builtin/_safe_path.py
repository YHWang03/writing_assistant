"""共享路径安全工具 — 将文件操作限制在项目目录内。"""

from pathlib import Path

# 项目根目录，所有文件操作限定在此目录下
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def safe_resolve(file_path: str) -> Path:
    """解析路径并校验未越出项目根目录。

    paras:
        file_path: 文件路径
    return: 解析后的绝对 Path；越界抛 ValueError
    """
    path = Path(file_path).resolve()
    if not str(path).startswith(str(_PROJECT_ROOT.resolve())):
        raise ValueError(f"路径越界: {file_path}（仅允许操作项目目录内的文件）")
    return path
