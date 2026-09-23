"""LiteratureAgent — 文献处理 Agent：解析 PDF、检索文献、生成 BibTeX 并维护文献库。"""

import logging
from pathlib import Path

from ..core.agent import Agent
from ..core.llm import LLM
from ..domain.library import save_library
from ..prompts import load_prompt
from ..tools.builtin import (
    GetPaperTextTool, SearchPapersTool, VerifyPaperTool,
    GenerateBibtexTool, SummarizePaperTool,
    ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool,
    GenerateBibFromRefLibTool, FinishTool, ParseAndStoreTool,
    ListPaperFilesTool, FindRelevantPapersTool, WriteLibraryTool,
)
from ._common import STANDARD_HOOKS
from ..tools.builtin.citations.manage import UpdateReferenceTool, RemoveReferenceTool
from ..domain.reference_provenance import ReferenceProvenance, title_key


class LiteratureAgent(Agent):
    """文献处理 Agent"""

    tool_types = (
        ParseAndStoreTool, GetPaperTextTool, SearchPapersTool, VerifyPaperTool,
        GenerateBibtexTool, SummarizePaperTool, UpdateReferenceTool, RemoveReferenceTool,
        ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool,
        GenerateBibFromRefLibTool, ListPaperFilesTool, FindRelevantPapersTool,
        WriteLibraryTool, FinishTool,
    )
    hook_types = STANDARD_HOOKS
    declared_output_exts = (".bib",)

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
        self.min_relevant = min_relevant
        self._setup_declared_components()
        self._provenance = ReferenceProvenance()

    def _sync_context_to_tools(self):
        """重置搜索计数，并将 context 中的引用函数与文献库注入到各工具。

        paras: 无
        return: 无
        """
        self.require_tool("search_papers").reset()
        if self.context is None:
            return
        self.require_tool("add_reference").set_add_func(self.context.add_reference)
        for ref in self.context.get_references():
            if ref.provenance_kind in ("pdf", "search", "verified_title") and ref.provenance_title == title_key(ref.title):
                self._provenance.record(ref.title, ref.provenance_kind)
        verifier = self.require_tool("verify_paper")
        verifier.provenance = self._provenance
        self.require_tool("search_papers").provenance = self._provenance
        adder = self.require_tool("add_reference")
        adder.provenance, adder.verifier = self._provenance, verifier
        updater = self.require_tool("update_reference")
        updater.provenance, updater.verifier = self._provenance, verifier
        self.require_tool("update_reference").context = self.context
        self.require_tool("remove_reference").context = self.context
        self.require_tool("parse_and_store").set_add_func(self.context.add_reference)
        self.require_tool("parse_and_store").provenance = self._provenance
        self.require_tool("parse_and_store").allowed_pdf_paths = {
            str(Path(p).resolve()) for p in
            (*self.context.read("seed_pdf_paths"), *self.context.read("ref_pdf_paths"))}
        self.require_tool("generate_bib_from_ref_library").set_reference_provider(self.context.get_references)
        self.require_tool("generate_bib_from_ref_library").set_persist_callback(lambda: save_library(
            self.context.library_dir, self.context.get_references(), replace=True))
        self.require_tool("list_paper_files").set_pdf_paths(
            self.context.read("seed_pdf_paths"), self.context.read("ref_pdf_paths"))
        relevant = self.require_tool("find_relevant_papers")
        relevant.set_library_path(self.context.library_dir)
        relevant.set_add_func(self.context.add_reference)
        relevant.set_min_relevant(self.min_relevant)
        writer = self.require_tool("write_library")
        writer.set_library_path(self.context.library_dir)
        writer.set_reference_provider(self.context.get_references)
        self.require_tool("read_context").set_context(self.context)

    def run(self, input_text: str) -> str:
        """运行 LiteratureAgent 执行循环，结束后兜底落盘文献库。

        paras:
            input_text: 任务输入文本
        return: 执行结果文本
        """
        try:
            return self._run_loop(input_text, verbose=True)
        finally:
            self._persist_library()

    def _persist_library(self):
        """任务结束后兜底落盘 reference_library，不依赖 agent 是否调用 write_library。

        paras: 无
        return: 无
        """
        if self.context is None:
            return
        lib_dir = self.context.library_dir
        refs = self.context.get_references()
        if not lib_dir:
            return
        try:
            n = len(save_library(lib_dir, refs, replace=True))
            logging.getLogger(__name__).info(
                f"[LiteratureAgent] 已落盘文献库 {n} 条到 {lib_dir}")
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"[LiteratureAgent] 落盘文献库失败: {e}")
