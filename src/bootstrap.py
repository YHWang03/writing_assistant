"""Build the application object graph from validated configuration."""

from __future__ import annotations

import logging

from .agents import (
    BuildAgent, CitationAgent, LiteratureAgent, MasterAgent, ReviewAgent, WritingAgent,
)
from .config import AppConfig
from .core.llm import LLM, configure_tool_llm
from .domain.library import load_library
from .domain.paper_context import PaperContext
from .lifecycle import Application
from .memory import AgentMemory

logger = logging.getLogger(__name__)


CONTEXT_POLICIES = {
    "master": ({
        "seed_pdf_paths", "ref_pdf_paths", "output_dir", "template_dir",
        "innovation_points", "experiment_description", "experiment_images",
        "formula_manuscript", "user_prompt", "reference_library", "sections",
        "main_tex_path", "modification_log",
    }, set()),
    "literature": ({
        "seed_pdf_paths", "ref_pdf_paths", "output_dir", "reference_library", "library_dir",
    }, {"reference_library", "modification_log"}),
    "writing": ({
        "output_dir", "template_dir", "innovation_points", "experiment_description",
        "experiment_images", "formula_manuscript", "user_prompt", "reference_library",
        "sections", "main_tex_path",
    }, {"sections", "main_tex_path", "modification_log"}),
    "citation": ({"main_tex_path", "reference_library", "output_dir"}, {"modification_log"}),
    "review": ({"main_tex_path", "output_dir", "sections", "reference_library"}, {"modification_log"}),
    "build": ({"template_dir", "main_tex_path", "output_dir"}, {"template_validated", "modification_log"}),
}


def _read_text(config: AppConfig, value: str) -> str:
    if not value:
        return ""
    path = config.resolve(value)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    logger.warning("文件不存在: %s", path)
    return ""


def _image_paths(config: AppConfig, value: str) -> list[str]:
    if not value:
        return []
    directory = config.resolve(value)
    if not directory.is_dir():
        logger.warning("图片目录不存在: %s", directory)
        return []
    extensions = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}
    return sorted(str(path) for path in directory.iterdir() if path.suffix.lower() in extensions)


def build_context(config: AppConfig) -> PaperContext:
    paths, paper = config.paths, config.paper
    template_dir = config.resolve(paths.template_dir)
    if not template_dir.is_dir() or not any(template_dir.glob("*.tex")):
        fallback = config.project_dir / "resources" / "templates" / "default"
        if fallback.is_dir():
            logger.warning("模板目录不可用（%s），回退到: %s", template_dir, fallback)
            template_dir = fallback
    seed_dir = config.resolve(paths.seed_dir)
    refs_dir = config.resolve(paths.refs_dir)
    library_dir = str(config.resolve(paths.library_dir))
    return PaperContext(
        output_dir=str(config.resolve(paths.output_dir)),
        template_dir=str(template_dir),
        library_dir=library_dir,
        seed_pdf_paths=sorted(str(path) for path in seed_dir.glob("*.pdf")) if seed_dir.is_dir() else [],
        ref_pdf_paths=sorted(str(path) for path in refs_dir.glob("*.pdf")) if refs_dir.is_dir() else [],
        innovation_points=_read_text(config, paper.innovation_points),
        experiment_description=_read_text(config, paper.experiment_description),
        experiment_images=_image_paths(config, paper.experiment_images),
        formula_manuscript=_read_text(config, paper.formula_manuscript),
        user_prompt=_read_text(config, paper.user_prompt),
        reference_library=load_library(library_dir),
    )


def _memory(config: AppConfig, name: str) -> AgentMemory:
    memory = config.memory
    return AgentMemory(
        path=str(config.resolve(memory.path.format(agent_name=name))),
        max_records=memory.max_records,
        recall_limit=memory.recall_limit,
        consolidate_threshold=memory.consolidate_threshold,
    )


def build_agents(config: AppConfig, llm: LLM) -> dict[str, object]:
    cfg = config.agents
    if cfg is None:
        raise ValueError("agents 配置尚未初始化")
    agents = {
        "master": MasterAgent(llm, cfg.master.max_steps, cfg.master.max_tokens, cfg.run_mode_for("MasterAgent")),
        "literature": LiteratureAgent(
            llm, cfg.literature.max_steps, cfg.literature.max_tokens,
            cfg.run_mode_for("LiteratureAgent"), cfg.literature.min_relevant or 15,
        ),
        "writing": WritingAgent(llm, cfg.writing.max_steps, cfg.writing.max_tokens, cfg.run_mode_for("WritingAgent")),
        "citation": CitationAgent(llm, cfg.citation.max_steps, cfg.citation.max_tokens, cfg.run_mode_for("CitationAgent")),
        "review": ReviewAgent(llm, cfg.review.max_steps, cfg.review.max_tokens, cfg.run_mode_for("ReviewAgent")),
        "build": BuildAgent(llm, cfg.build.max_steps, cfg.build.max_tokens, cfg.run_mode_for("BuildAgent")),
    }
    for name, agent in agents.items():
        agent.memory = _memory(config, name) if config.memory.enabled.get(name, False) else None
    master = agents["master"]
    for name in ("literature", "writing", "citation", "review", "build"):
        master.register_sub_agent(agents[name].name, agents[name])
    return agents


def create_application(config: AppConfig) -> Application:
    configure_tool_llm(config.llm.flash_model)
    llm = LLM(model=config.llm.model)
    context = build_context(config)
    agents = build_agents(config, llm)
    for name, agent in agents.items():
        readable, writable = CONTEXT_POLICIES[name]
        agent.context = context.view(readable=readable, writable=writable)
    return Application(context=context, agents=agents)
