"""PaperContext — 共享上下文与权限隔离，各 Agent 经 view() 获得受控视图"""

from dataclasses import dataclass, field
from datetime import datetime
import logging

from .paper import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperContext:
    """论文写作的共享上下文，所有 Agent 通过 View 访问"""

    seed_pdf_paths: list[str] = field(default_factory=list)
    ref_pdf_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    template_dir: str = ""
    library_dir: str = ""

    innovation_points: str = ""
    experiment_description: str = ""
    experiment_images: list[str] = field(default_factory=list)
    formula_manuscript: str = ""
    user_prompt: str = ""

    template_validated: bool = False
    template_info: str = ""

    reference_library: list[Paper] = field(default_factory=list)
    related_work_draft: str = ""

    sections: dict[str, str] = field(default_factory=dict)
    main_tex_path: str = ""

    modification_log: list[dict] = field(default_factory=list)

    def add_reference(self, ref: Paper):
        """按 cite_key 去重添加文献。

        paras:
            ref: Paper 对象
        return: 无
        """
        if not ref.cite_key:
            return
        for existing in self.reference_library:
            if existing.cite_key == ref.cite_key:
                return
        self.reference_library.append(ref)

    def set_section(self, name: str, content: str):
        """设置章节内容。

        paras:
            name: 章节名
            content: 章节内容
        return: 无
        """
        self.sections[name] = content

    def get_section(self, name: str) -> str:
        """获取章节内容。

        paras:
            name: 章节名
        return: 章节内容；不存在返回空串
        """
        return self.sections.get(name, "")

    def get_all_cite_keys(self) -> list[str]:
        """获取全部文献 cite_key。

        paras: 无
        return: cite_key 列表
        """
        return [ref.cite_key for ref in self.reference_library if ref.cite_key]

    def log_modification(self, agent: str, action: str, detail: str):
        """追加一条修改日志。

        paras:
            agent: Agent 名
            action: 动作名
            detail: 详情
        return: 无
        """
        self.modification_log.append({
            "agent": agent,
            "action": action,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })

    def view(self, readable: set[str], writable: set[str]) -> "AgentContextView":
        """创建带权限隔离的视图。

        paras:
            readable: 可读字段名集合
            writable: 可写字段名集合
        return: AgentContextView 实例
        """
        return AgentContextView(self, readable, writable)


class AgentContextView:
    """Agent 上下文视图 — 权限隔离代理，越权读写抛 AttributeError"""

    def __init__(self, context: PaperContext, readable: set[str], writable: set[str]):
        """构造视图。

        paras:
            context: 底层 PaperContext
            readable: 可读字段名集合
            writable: 可写字段名集合
        return: 无
        """
        self._context = context
        self._readable = readable
        self._writable = writable

    def __getattr__(self, name: str):
        """字段读取代理，越权抛错。

        paras:
            name: 字段名
        return: 底层 context 的字段值
        """
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._readable:
            raise AttributeError(
                f"AgentContextView: 字段 '{name}' 不可读（当前 Agent 无权读取）"
            )
        return getattr(self._context, name)

    def __setattr__(self, name: str, value):
        """字段写入代理，越权抛错。

        paras:
            name: 字段名
            value: 写入值
        return: 无
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        if name not in self._writable:
            raise AttributeError(
                f"AgentContextView: 字段 '{name}' 不可写（当前 Agent 无权修改）"
            )
        setattr(self._context, name, value)

    def add_reference(self, ref: Paper):
        """权限校验后添加文献。

        paras:
            ref: Paper 对象
        return: 无
        """
        if "reference_library" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 reference_library")
        self._context.add_reference(ref)

    def set_section(self, name: str, content: str):
        """权限校验后设置章节。

        paras:
            name: 章节名
            content: 章节内容
        return: 无
        """
        if "sections" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 sections")
        self._context.set_section(name, content)

    def get_section(self, name: str) -> str:
        """权限校验后读取章节。

        paras:
            name: 章节名
        return: 章节内容
        """
        if "sections" not in self._readable:
            raise AttributeError("AgentContextView: 无权读取 sections")
        return self._context.get_section(name)

    def get_all_cite_keys(self) -> list[str]:
        """权限校验后读取全部 cite_key。

        paras: 无
        return: cite_key 列表
        """
        if "reference_library" not in self._readable:
            raise AttributeError("AgentContextView: 无权读取 reference_library")
        return self._context.get_all_cite_keys()

    def log_modification(self, agent: str, action: str, detail: str):
        """权限校验后追加修改日志。

        paras:
            agent: Agent 名
            action: 动作名
            detail: 详情
        return: 无
        """
        if "modification_log" not in self._writable:
            raise AttributeError("AgentContextView: 无权修改 modification_log")
        self._context.log_modification(agent, action, detail)

    def get_readable_summary(self) -> str:
        """生成可读字段摘要（注入 Agent system prompt）。

        paras: 无
        return: 逐字段摘要文本
        """
        lines = ["## Your Context Access"]
        lines.append(f"You can read: {sorted(self._readable)}")
        lines.append(f"You can write: {sorted(self._writable)}")
        lines.append("")

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
                        preview = [str(p) for p in value[:5]]
                        lines.append(f"    前几个: {preview}")
                elif isinstance(value, dict):
                    lines.append(f"  {field}: {{{len(value)} 个键}}")
                elif isinstance(value, bool):
                    lines.append(f"  {field}: {value}")
            except AttributeError:
                pass
        return "\n".join(lines)
