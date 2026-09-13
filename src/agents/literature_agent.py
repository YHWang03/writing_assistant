"""
LiteratureAgent — 文献处理 Agent

负责：
- 解析用户提供的 PDF 论文
- 提取元数据（标题、作者、摘要、期刊等）
- 总结核心贡献
- 在线搜索补充文献
- 生成 BibTeX 条目
- 写入 references.bib 文件
"""

import logging

from ..core.agent import Agent
from ..core.llm import LLM
from ..core.library import save_library
from ..prompts import load_prompt
from ..tools.registry import ToolRegistry
from ..tools.builtin import (
    ParsePDFTool, GetPaperTextTool, SearchPapersTool, VerifyPaperTool,
    GenerateBibtexTool, SummarizePaperTool, WriteBibFileTool, WriteFileTool,
    ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool, MemoryTool,
    GenerateBibFromRefLibTool, FinishTool, ParseAndStoreTool,
    ListPaperFilesTool, FindRelevantPapersTool, WriteLibraryTool,
)


class LiteratureAgent(Agent):
    """文献处理 Agent"""

    def __init__(self, llm: LLM,
                 max_steps: int = 15, max_tokens: int = 8192,
                 run_mode: str = "react", min_relevant: int = 15):
        super().__init__(
            name="LiteratureAgent", llm=llm,
            system_prompt=load_prompt("literature_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self._setup_tools()
        self.min_relevant = min_relevant
        # 确定性产出闸门：完成前必须写出 .bib 文件（存在且非空）才允许 finish
        self.required_output_exts = [".bib"]

    def _setup_tools(self):
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
        self.tool_registry.register(MemoryTool())
        self.tool_registry.register(GenerateBibFromRefLibTool())
        self.tool_registry.register(ListPaperFilesTool())
        self.tool_registry.register(FindRelevantPapersTool())
        self.tool_registry.register(WriteLibraryTool())
        self.tool_registry.register(FinishTool())

    def _sync_context_to_tools(self):
        """将 context.add_reference 注入到 AddReferenceTool，
        将 reference_library 注入到 GenerateBibFromRefLibTool，
        并将 memory_manager 注入到 MemoryTool"""
        # 每个任务开始重置搜索计数，防止跨 dispatch 累积（配合 search_papers 硬上限）
        try:
            search_tool = self.tool_registry.get_tool("search_papers")
            if search_tool is not None and hasattr(search_tool, "reset"):
                search_tool.reset()
        except Exception:
            pass

        try:
            mem_tool = self.tool_registry.get_tool("memory")
            if mem_tool is not None and self.memory_manager is not None:
                mem_tool.set_memory_manager(self.memory_manager)
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
        result = self._run_loop(input_text, verbose=True)
        self._persist_library()
        return result

    def _persist_library(self):
        """任务结束后兜底落盘 reference_library，不依赖 agent 是否调用 write_library。"""
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