"""Application lifecycle, task execution, and durable shutdown."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from .domain.library import save_library
from .domain.paper_context import PaperContext
from .observability.telemetry import task_usage

logger = logging.getLogger(__name__)


@dataclass
class Application:
    context: PaperContext
    agents: dict[str, object]
    _closed: bool = False

    def run_task(self, prompt: str) -> str:
        with task_usage():
            result = self.agents["master"].run(prompt)
            if self.context.main_tex_path:
                from .domain.citation_evidence import report_status
                status = report_status(self.context.main_tex_path, self.context.reference_library, mark_stale=True)
                result = "[程序核验状态，以此为准]\n" + status + "\n\n[Agent 总结]\n" + result
            return result

    def close(self) -> None:
        """Persist durable state exactly once."""
        if self._closed:
            return
        self._closed = True
        if self.context.library_dir:
            try:
                count = len(save_library(self.context.library_dir, self.context.reference_library, replace=True))
                logger.info("已保存文献库 %d 条到 %s", count, self.context.library_dir)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("保存文献库失败: %s", exc)
        for name, agent in self.agents.items():
            if agent.memory:
                try:
                    agent.memory.save()
                    logger.info("已保存 %s 长期记忆", name)
                except OSError as exc:
                    logger.warning("保存 %s 长期记忆失败: %s", name, exc)

    def __enter__(self) -> "Application":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
