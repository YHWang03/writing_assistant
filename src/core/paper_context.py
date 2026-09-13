"""
PaperContext — 共享上下文 + 权限隔离

所有 Agent 通过 PaperContext 共享状态，通过 AgentContextView 实现权限隔离。

用一个 PaperContext 实例来维持上下文， 对于每一个agent通过view方法来建立带有权限隔离的上下文
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging

from .paper import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperContext:
    """论文写作的共享上下文，所有 Agent 通过 View 访问"""

    # ---- 路径配置 ----
    seed_pdf_paths: list[str] = field(default_factory=list)
    ref_pdf_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    template_dir: str = ""
    library_dir: str = ""

    # ---- 论文配置 ----
    innovation_points: str = ""
    experiment_description: str = ""
    experiment_images: list[str] = field(default_factory=list)
    formula_manuscript: str = ""
    user_prompt: str = ""

    # ---- 模板 ----
    template_validated: bool = False
    template_info: str = ""

    # ---- 文献 ----
    reference_library: list[Paper] = field(default_factory=list)
    related_work_draft: str = ""

    # ---- 写作 ----
    sections: dict[str, str] = field(default_factory=dict)
    main_tex_path: str = ""

    # ---- 修改日志 ----
    modification_log: list[dict] = field(default_factory=list)

    # ---- 公共方法 ----

    def add_reference(self, ref: Paper):
        """添加文献到文献库"""
        if not ref.cite_key:
            return
        for existing in self.reference_library:
            if existing.cite_key == ref.cite_key:
                return
        self.reference_library.append(ref)

    def set_section(self, name: str, content: str):
        """设置章节内容"""
        self.sections[name] = content

    def get_section(self, name: str) -> str:
        """获取章节内容"""
        return self.sections.get(name, "")

    def get_all_cite_keys(self) -> list[str]:
        """获取所有文献的 cite_key"""
        return [ref.cite_key for ref in self.reference_library if ref.cite_key]

    def log_modification(self, agent: str, action: str, detail: str):
        """记录修改日志"""
        self.modification_log.append({
            "agent": agent,
            "action": action,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })

    def view(self, readable: set[str], writable: set[str]) -> "AgentContextView":
        """创建带权限隔离的视图"""
        return AgentContextView(self, readable, writable)


class AgentContextView:
    """
    Agent 上下文视图 — 权限隔离代理

    每个 Agent 只能读/写被允许的字段，其他字段访问会报错。
    所有写操作自动委托给底层 PaperContext。
    """

    def __init__(self, context: PaperContext, readable: set[str], writable: set[str]):
        self._context = context
        self._readable = readable
        self._writable = writable

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._readable:
            raise AttributeError(
                f"AgentContextView: 字段 '{name}' 不可读（当前 Agent 无权读取）"
            )
        return getattr(self._context, name)

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if name not in self._writable:
            raise AttributeError(
                f"AgentContextView: 字段 '{name}' 不可写（当前 Agent 无权修改）"
            )
        setattr(self._context, name, value)

    def add_reference(self, ref: Paper):
        if "reference_library" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 reference_library")
        self._context.add_reference(ref)

    def set_section(self, name: str, content: str):
        if "sections" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 sections")
        self._context.set_section(name, content)

    def get_section(self, name: str) -> str:
        if "sections" not in self._readable:
            raise AttributeError("AgentContextView: 无权读取 sections")
        return self._context.get_section(name)

    def get_all_cite_keys(self) -> list[str]:
        if "reference_library" not in self._readable:
            raise AttributeError("AgentContextView: 无权读取 reference_library")
        return self._context.get_all_cite_keys()

    def log_modification(self, agent: str, action: str, detail: str):
        if "modification_log" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 modification_log")
        self._context.log_modification(agent, action, detail)

    def get_readable_summary(self) -> str:
        """返回可读字段的摘要，注入到 Agent 的 system prompt 中"""
        lines = ["## Your Context Access"]
        lines.append(f"You can read: {sorted(self._readable)}")
        lines.append(f"You can write: {sorted(self._writable)}")
        lines.append("")

        # 展示已有数据的字段值
        for field in sorted(self._readable):
            try:
                value = getattr(self._context, field)
                if value is None:
                    continue
                if isinstance(value, str) and not value:
                    continue
                if isinstance(value, (list, dict)) and not value:
                    continue

                if isinstance(value, str):
                    display = value[:80] + "..." if len(value) > 80 else value
                    lines.append(f"  {field}: {display}")
                elif isinstance(value, list):
                    lines.append(f"  {field}: [{len(value)} 个元素]")
                    if field.endswith("_paths") and value:
                        # 展开路径列表的前几个
                        preview = [str(p) for p in value[:5]]
                        lines.append(f"    前几个: {preview}")
                elif isinstance(value, dict):
                    lines.append(f"  {field}: {{{len(value)} 个键}}")
                elif isinstance(value, bool):
                    lines.append(f"  {field}: {value}")
            except AttributeError:
                pass
        return "\n".join(lines)