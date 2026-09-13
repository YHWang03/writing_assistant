"""共享路径安全工具 — 供所有文件操作工具使用
- safe_resolve: 安全解析路径，防止路径遍历攻击
"""

from pathlib import Path

# 项目根目录，所有文件操作限定在此目录下
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def safe_resolve(file_path: str) -> Path:
    """安全解析路径，防止路径遍历攻击"""
    path = Path(file_path).resolve()
    if not str(path).startswith(str(_PROJECT_ROOT.resolve())):
        raise ValueError(f"路径越界: {file_path}（仅允许操作项目目录内的文件）")
    return path