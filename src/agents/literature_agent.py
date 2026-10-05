"""LiteratureAgent — 文献处理 Agent：解析 PDF、检索文献、生成 BibTeX 并维护文献库。"""

import logging
import json
from pathlib import Path

from ..core.agent import Agent
from ..core.llm import LLM
from ..domain.library import save_library
from ..prompts import load_prompt
from ..tools.builtin import (
    SearchPapersTool,
    GenerateBibtexTool,
    ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool,
    GenerateBibFromRefLibTool, FinishTool, ParseAndStoreTool,
    ListPaperFilesTool, FindRelevantPapersTool, WriteLibraryTool,
)
from ._common import STANDARD_HOOKS
from ..tools.builtin.citations.manage import UpdateReferenceTool, RemoveReferenceTool
from ..tools.builtin._safe_path import FileReadPolicy
from ..domain.library import LIBRARY_FILENAME
from ..domain.library import reference_snapshot
from ..domain.bibliography import render_bib
from ..hooks.builtin.literature_finish import LiteratureFinishHook


class LiteratureAgent(Agent):
    """文献处理 Agent"""

    tool_types = (
        ParseAndStoreTool, SearchPapersTool,
        GenerateBibtexTool, UpdateReferenceTool, RemoveReferenceTool,
        ReadFileTool, ListFilesTool, AddReferenceTool, ReadContextTool,
        GenerateBibFromRefLibTool, ListPaperFilesTool, FindRelevantPapersTool,
        WriteLibraryTool, FinishTool,
    )
    hook_types = (LiteratureFinishHook,) + STANDARD_HOOKS
    declared_output_exts = (".bib",)
    scoped_read_tools = ("read_file", "ls", "parse_and_store")

    def __init__(self, llm: LLM,
                 max_steps: int = 15, max_tokens: int = 8192,
                 run_mode: str = "react", min_relevant: int = 15,
                 finish_reserve_steps: int = 3):
        """初始化 LiteratureAgent，注册工具集并设置检索阈值与产出闸门。

        paras:
            llm: LLM 实例
            max_steps: 最大执行步数
            max_tokens: 单次生成最大 token 数
            run_mode: 运行模式（react/plan_execute）
            min_relevant: 相关文献检索的最低数量阈值
            finish_reserve_steps: 预留用于保存、导出和汇报的轮数
        return: 无
        """
        super().__init__(
            name="LiteratureAgent", llm=llm,
            system_prompt=load_prompt("literature_agent.md"),
            max_steps=max_steps, max_tokens=max_tokens,
            run_mode=run_mode,
        )
        self.min_relevant = min_relevant
        self.finish_reserve_steps = finish_reserve_steps
        self._finish_confirmed = False
        self.last_run_status = ''
        self._setup_declared_components()
        self._bind_read_policy()  # 上下文尚未注入时默认拒绝读取。

    def _bind_read_policy(self):
        '''根据当前上下文为文献工具绑定精确文件读取白名单，无上下文时默认拒绝。
        '''
        files = []
        if self.context is not None:
            files.extend((*self.context.read("seed_pdf_paths"),
                          *self.context.read("ref_pdf_paths")))
            library_dir = self.context.library_dir
            if library_dir:
                files.append(Path(library_dir) / LIBRARY_FILENAME)
            # 精简上下文可不授权 output_dir，此时不授予对应文件权限。
            for directory in (library_dir, getattr(self.context, "output_dir", "")):
                if directory:
                    files.append(Path(directory) / "references.bib")
        policy = FileReadPolicy(self.name, files)
        for name in self.scoped_read_tools:
            self.require_tool(name).read_policy = policy

    def _sync_context_to_tools(self):
        """重置搜索计数，并将 context 中的引用函数与文献库注入到各工具。"""
        self.require_tool("search_papers").reset()
        self._bind_read_policy()
        if self.context is None:
            return
        self.require_tool("add_reference").set_add_func(self.context.add_reference)
        self.require_tool('add_reference').search_results = self.require_tool('search_papers').results
        self.require_tool("update_reference").context = self.context
        self.require_tool("remove_reference").context = self.context
        self.require_tool("parse_and_store").set_add_func(self.context.add_reference)
        self.require_tool("parse_and_store").set_reference_provider(self.context.get_references)
        self.require_tool("parse_and_store").allowed_pdf_paths = {
            str(Path(p).resolve()) for p in
            (*self.context.read("seed_pdf_paths"), *self.context.read("ref_pdf_paths"))}
        self.require_tool("add_reference").allowed_pdf_paths = self.require_tool("parse_and_store").allowed_pdf_paths
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
        """运行文献任务并核对完成状态；异常仅保存已有库，不自动导出。

        paras:
            input_text: 任务输入文本
        return: 程序核验状态和交付摘要；执行异常继续抛出并记录保留状态
        """
        self.require_tool("generate_bib_from_ref_library").exports.clear()
        self.require_tool('parse_and_store').reset_task()
        self._finish_confirmed = False
        self.last_run_status = ''
        interrupted = True
        try:
            result = self._run_loop(input_text, verbose=True)
            interrupted = False
        finally:
            self._persist_library()
            issues = self.completion_issues()
            if interrupted:
                issues.insert(0, '执行异常或中断')
            elif not self._finish_confirmed:
                issues.insert(0, '未通过正常结束检查，可能已耗尽步数')
            count = len(self.context.get_references()) if self.context is not None else 0
            self.last_run_status = (f'任务未完成（当前库 {count} 条）：' + '；'.join(issues)
                                    if issues else f'文献产物已核对同步（{count} 条）；语义质量及任务覆盖范围以未完成项说明为准。')
            logging.getLogger(__name__).info(self.last_run_status,
                extra={'event': 'literature_task_status', 'agent': self.name,
                       'completed': not issues, 'reference_count': count, 'issues': issues})
        return '[程序核验状态]\n' + self.last_run_status + '\n\n[交付摘要]\n' + result

    def completion_issues(self):
        '''核对当前库已保存、本次导出记录及磁盘 .bib 是否仍与当前库一致。

        return: 未完成项列表；空列表只表示文献产物已同步，不证明检索充分。
        '''
        if self.context is None:
            return ['尚未绑定文献上下文']
        references = self.context.get_references()
        current = reference_snapshot(references)
        issues = []
        try:
            if not self.context.library_dir:
                raise ValueError('未配置文献库目录')
            saved = json.loads((Path(self.context.library_dir) / LIBRARY_FILENAME).read_text(encoding='utf-8'))
            if not isinstance(saved, list) or sorted(saved, key=lambda item: item['cite_key']) != current:
                issues.append('持久文献库与当前库不同步，请 write_library')
        except (OSError, ValueError, TypeError, KeyError):
            issues.append('持久文献库尚未成功保存或无法读取，请 write_library')
        exports = self.require_tool('generate_bib_from_ref_library').exports
        output_dir = getattr(self.context, 'output_dir', '')
        required = str((Path(output_dir) / 'references.bib').resolve()) if output_dir else None
        if not exports or (required and required not in exports):
            issues.append('本次尚未成功导出输出目录的 references.bib' if required else '本次尚未成功导出 .bib')
        try:
            content = render_bib(references)
            for path, version in exports.items():
                if version != current or Path(path).read_text(encoding='utf-8') != content:
                    issues.append(f'导出已过期，请重新导出：{path}')
        except (OSError, ValueError, TypeError):
            issues.append('无法核对 .bib，请修复并重新导出')
        return issues

    def _output_satisfied(self, written_paths):
        '''用文献库和导出版本一致性替代仅检查非空文件的通用闸门。

        paras:
            written_paths: 通用接口的已写路径集合，本实现使用实际导出记录。
        return: 保存和导出检查全部通过时为 True。
        '''
        return not self.completion_issues()

    def _persist_library(self):
        """任务结束后兜底落盘 reference_library，不依赖 agent 是否调用 write_library。"""
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
