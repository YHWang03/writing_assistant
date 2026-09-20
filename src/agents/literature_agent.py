"""LiteratureAgent — 文献处理 Agent：解析 PDF、检索文献、生成 BibTeX 并维护文献库。"""

import logging

from ..core.agent import Agent
from ..core.llm import LLM
from ..core.library import save_library
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    GetPaperTextTool, SearchPapersTool, VerifyPaperTool,
    GenerateBibtexTool, SummarizePaperTool, WriteBibFileTool, WriteFileTool,
    ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool,
    GenerateBibFromRefLibTool, FinishTool, ParseAndStoreTool,
    ListPaperFilesTool, FindRelevantPapersTool, WriteLibraryTool,
)
from ..hooks.builtin import (
    MemoryRecallHook, DeadlineNudgeHook, CollectWrittenPathsHook,
    FinishNudgeHook, OutputGateHook, MemoryExtractHook,
)


class LiteratureAgent(Agent):
    """文献处理 Agent"""

    def __init__(self, llm: LLM,
                 max_steps: int = 15, max_tokens: int = 8192,
                 run_mode: str = "react", min_relevant: int = 15):
        """初始化 LiteratureAgent，注册工具集并设置检索阈值与产出闸门。

        paras:
            llm: LLM 实例
            max_steps: 最大执行步数
            max_tokens: 单次生成最大 token 数
            run_mode: 运行模式（react/plan_execute）
            min_relevant: 相关文献检索的最低数量阈值
        return: 无
        """
        super().__init__(
            name="LiteratureAgent", llm=llm,
            system_prompt=load_prompt("literature_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()
        self._setup_hooks()
        self.min_relevant = min_relevant
        self.required_output_exts = [".bib"]

    def _setup_tools(self):
        """注册本 Agent 的工具集。

        paras: 无
        return: 无
        """
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(ParseAndStoreTool())
        self.tool_registry.register(GetPaperTextTool())
        self.tool_registry.register(SearchPapersTool())
        self.tool_registry.register(VerifyPaperTool())
        self.tool_registry.register(GenerateBibtexTool())
        self.tool_registry.register(SummarizePaperTool())
        self.tool_registry.register(WriteBibFileTool())
        write_file = WriteFileTool()
        write_file.set_agent_name("LiteratureAgent")
        self.tool_registry.register(write_file)
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(ListFilesTool())
        self.tool_registry.register(AddReferenceTool())
        self._read_context = ReadContextTool()
        self.tool_registry.register(self._read_context)
        self.tool_registry.register(GenerateBibFromRefLibTool())
        self.tool_registry.register(ListPaperFilesTool())
        self.tool_registry.register(FindRelevantPapersTool())
        self.tool_registry.register(WriteLibraryTool())
        self.tool_registry.register(FinishTool())

    def _setup_hooks(self):
        """注册本 Agent 的 hooks。

        paras: 无
        return: 无
        """
        self.hooks.register(MemoryRecallHook())
        self.hooks.register(DeadlineNudgeHook())
        self.hooks.register(CollectWrittenPathsHook())
        self.hooks.register(FinishNudgeHook())
        self.hooks.register(OutputGateHook())
        self.hooks.register(MemoryExtractHook())

    def _sync_context_to_tools(self):
        """重置搜索计数，并将 context 中的引用函数与文献库注入到各工具。

        paras: 无
        return: 无
        """
        try:
            search_tool = self.tool_registry.get_tool("search_papers")
            if search_tool is not None and hasattr(search_tool, "reset"):
                search_tool.reset()
        except Exception:
            pass

        if self.context is not None:
            try:
                add_ref = self.tool_registry.get_tool("add_reference")
                if add_ref is not None:
                    add_ref.set_add_func(self.context.add_reference)
            except Exception:
                pass
            try:
                parse_store = self.tool_registry.get_tool("parse_and_store")
                if parse_store is not None:
                    parse_store.set_add_func(self.context.add_reference)
            except Exception:
                pass
            try:
                bib_tool = self.tool_registry.get_tool("generate_bib_from_ref_library")
                if bib_tool is not None:
                    bib_tool.set_reference_library(self.context.reference_library)
            except Exception:
                pass
            try:
                lp_tool = self.tool_registry.get_tool("list_paper_files")
                if lp_tool is not None:
                    lp_tool.set_pdf_paths(
                        self.context.seed_pdf_paths, self.context.ref_pdf_paths)
            except Exception:
                pass
            try:
                fr_tool = self.tool_registry.get_tool("find_relevant_papers")
                if fr_tool is not None:
                    fr_tool.set_library_path(self.context.library_dir)
                    fr_tool.set_add_func(self.context.add_reference)
                    fr_tool.set_min_relevant(self.min_relevant)
            except Exception:
                pass
            try:
                wl_tool = self.tool_registry.get_tool("write_library")
                if wl_tool is not None:
                    wl_tool.set_library_path(self.context.library_dir)
                    wl_tool.set_reference_library(self.context.reference_library)
            except Exception:
                pass
            try:
                self._read_context.set_context(self.context)
            except Exception:
                pass

    def run(self, input_text: str) -> str:
        """运行 LiteratureAgent 执行循环，结束后兜底落盘文献库。

        paras:
            input_text: 任务输入文本
        return: 执行结果文本
        """
        result = self._run_loop(input_text, verbose=True)
        self._persist_library()
        return result

    def _persist_library(self):
        """任务结束后兜底落盘 reference_library，不依赖 agent 是否调用 write_library。

        paras: 无
        return: 无
        """
        if self.context is None:
            return
        try:
            lib_dir = self.context.library_dir
            refs = self.context.reference_library
        except Exception:
            return
        if not lib_dir or not refs:
            return
        try:
            n = len(save_library(lib_dir, refs))
            logging.getLogger(__name__).info(
                f"[LiteratureAgent] 已落盘文献库 {n} 条到 {lib_dir}")
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"[LiteratureAgent] 落盘文献库失败: {e}")
