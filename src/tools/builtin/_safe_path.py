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
    if not path.is_relative_to(_PROJECT_ROOT.resolve()):
        raise ValueError(f"路径越界: {file_path}（仅允许操作项目目录内的文件）")
    return path


class FileReadPolicy:
    """精确文件白名单；仅允许列出其直接父目录中的授权文件。"""

    def __init__(self, agent_name: str, files=()):
        '''规范化精确文件白名单，并派生允许列出的直接父目录集合。

        paras:
            agent_name: 所属 Agent 的名称。
            files: 允许读取的精确文件路径序列。
        '''
        self.agent_name = agent_name
        # 配置阶段只规范化；执行时仍须同时通过项目边界与白名单检查。
        self.files = frozenset(Path(p).resolve() for p in files)
        self.directories = frozenset(p.parent for p in self.files)

    def resolve(self, value: str, *, directory: bool = False) -> Path:
        '''校验项目边界和文件或目录白名单，拒绝越权路径。

        paras:
            value: 请求访问的文件或目录路径。
            directory: 是否按可列出的目录校验；False 表示校验文件。
        return: 允许访问的规范化绝对路径；拒绝时抛出 ValueError。
        '''
        path = safe_resolve(value)
        if path not in (self.directories if directory else self.files):
            raise ValueError(
                f"{self.agent_name} 无权访问: {value}。仅允许配置中的文献文件；"
                "研究主题和检索要求应由 Master 提供，请继续文献处理任务。")
        return path

    def visible(self, path: Path) -> bool:
        '''检查目录条目解析后的目标是否为授权文件。

        paras:
            path: 待展示的目录条目路径。
        return: 允许展示返回 True，越界或未授权返回 False。
        '''
        try:
            return safe_resolve(str(path)) in self.files
        except ValueError:
            return False


class ScopedFileRead:
    """工具执行层的可选读取权限；未配置的其他 Agent 保持原行为。"""

    read_policy: FileReadPolicy | None = None

    def resolve_read(self, value: str, *, directory: bool = False) -> Path:
        '''使用绑定的读取策略校验路径，无策略时仅检查项目边界。

        paras:
            value: 请求读取的文件或列出的目录路径。
            directory: 是否按可列出的目录校验；False 表示校验文件。
        return: 允许读取的规范化绝对路径。
        '''
        if self.read_policy is not None:
            return self.read_policy.resolve(value, directory=directory)
        return safe_resolve(value)
