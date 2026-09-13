"""
writing_assistant 入口程序

用法：
    python main.py                          # 交互模式
    python main.py --config config.yaml     # 指定配置文件
"""

import sys
import logging
import re
from datetime import date
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

# 项目根目录（main.py 所在目录），用于解析相对路径
_PROJECT_DIR = Path(__file__).parent.resolve()

from src.core.llm import LLM, configure_tool_llm
from src.core.logging_setup import setup_structured_logging
from src.core.paper_context import PaperContext
from src.core.library import save_library
from src.memory import MemoryManager
from src.agents.master_agent import MasterAgent
from src.agents.literature_agent import LiteratureAgent
from src.agents.writing_agent import WritingAgent
from src.agents.citation_agent import CitationAgent
from src.agents.review_agent import ReviewAgent
from src.agents.build_agent import BuildAgent


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    # 相对路径基于项目根目录
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_context(config: dict) -> PaperContext:
    """根据配置构建 PaperContext"""
    paths = config.get("paths", {})
    paper = config.get("paper", {})

    def _resolve(p: str) -> Path:
        """解析路径：绝对路径直接用，相对路径基于项目根目录"""
        path = Path(p)
        if path.is_absolute():
            return path
        return _PROJECT_DIR / path

    def _read_file(path_str: str) -> str:
        """读取文件内容，路径不存在或为空时返回空字符串"""
        if not path_str:
            return ""
        p = _resolve(path_str)
        if p.exists():
            return p.read_text(encoding="utf-8")
        logger.warning(f"文件不存在: {p}")
        return ""

    def _read_image_dir(path_str: str) -> list[str]:
        """读取图片目录，返回所有图片文件的路径列表"""
        if not path_str:
            return []
        p = _resolve(path_str)
        if not p.exists() or not p.is_dir():
            logger.warning(f"图片目录不存在: {p}")
            return []
        image_exts = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}
        return [str(f) for f in p.iterdir() if f.suffix.lower() in image_exts]

    seed_dir = _resolve(paths.get("seed_dir", "refs"))
    refs_dir = _resolve(paths.get("refs_dir", "reference"))
    library_dir = _resolve(paths.get("library_dir", "example/library"))
    return PaperContext(
        output_dir=str(_resolve(paths.get("output_dir", "output"))),
        template_dir=str(_resolve(paths.get("template_dir", "template"))),
        library_dir=str(library_dir),
        seed_pdf_paths=[str(p) for p in seed_dir.glob("*.pdf")] if seed_dir.exists() else [],
        ref_pdf_paths=[str(p) for p in refs_dir.glob("*.pdf")] if refs_dir.exists() else [],
        innovation_points=_read_file(paper.get("innovation_points", "")),
        experiment_description=_read_file(paper.get("experiment_description", "")),
        experiment_images=_read_image_dir(paper.get("experiment_images", "")),
        formula_manuscript=_read_file(paper.get("formula_manuscript", "")),
        user_prompt=_read_file(paper.get("user_prompt", "")),
    )


def _build_agent_memory(agent_name: str, config: dict) -> MemoryManager:
    """为单个 Agent 构建 MemoryManager"""
    mem_cfg = config.get("memory", {})
    long_term_path = mem_cfg.get("long_term_path", f"memory/{agent_name}_long_term.json")
    # 相对路径基于项目根目录
    lt_path = Path(long_term_path)
    if not lt_path.is_absolute():
        lt_path = _PROJECT_DIR / lt_path
    return MemoryManager(
        long_term_max_size=mem_cfg.get("long_term_max_size", 500),
        episodic_max_size=mem_cfg.get("episodic_max_size", 200),
        long_term_path=str(lt_path),
    )


def build_agents(config: dict, llm: LLM) -> dict[str, object]:
    """构建所有 Agent 实例，注入记忆系统"""
    agent_cfg = config.get("agents", {})

    # 读取 run_mode：支持全局配置或按 Agent 分别配置
    run_mode_cfg = agent_cfg.get("run_mode", "react")
    if isinstance(run_mode_cfg, dict):
        # 按 Agent 分别配置，如 {"default": "react", "LiteratureAgent": "plan_execute"}
        default_run_mode = run_mode_cfg.get("default", "react")
    else:
        default_run_mode = run_mode_cfg

    def _get_run_mode(agent_name: str) -> str:
        if isinstance(run_mode_cfg, dict):
            return run_mode_cfg.get(agent_name, default_run_mode)
        return default_run_mode

    # --- MasterAgent ---
    master = MasterAgent(
        llm=llm,
        max_steps=agent_cfg.get("master", {}).get("max_steps", 12),
        max_tokens=agent_cfg.get("master", {}).get("max_tokens", 4096),
        run_mode=_get_run_mode("MasterAgent"),
    )
    master.memory_manager = _build_agent_memory("master", config)

    # --- LiteratureAgent ---
    literature = LiteratureAgent(
        llm=llm,
        max_steps=agent_cfg.get("literature", {}).get("max_steps", 15),
        max_tokens=agent_cfg.get("literature", {}).get("max_tokens", 4096),
        run_mode=_get_run_mode("LiteratureAgent"),
        min_relevant=agent_cfg.get("literature", {}).get("min_relevant", 15),
    )
    literature.memory_manager = _build_agent_memory("literature", config)

    # --- WritingAgent ---
    writing = WritingAgent(
        llm=llm,
        max_steps=agent_cfg.get("writing", {}).get("max_steps", 10),
        max_tokens=agent_cfg.get("writing", {}).get("max_tokens", 8192),
        run_mode=_get_run_mode("WritingAgent"),
    )
    writing.memory_manager = _build_agent_memory("writing", config)

    # --- CitationAgent ---
    citation = CitationAgent(
        llm=llm,
        max_steps=agent_cfg.get("citation", {}).get("max_steps", 12),
        max_tokens=agent_cfg.get("citation", {}).get("max_tokens", 4096),
        run_mode=_get_run_mode("CitationAgent"),
    )
    citation.memory_manager = _build_agent_memory("citation", config)

    # --- ReviewAgent ---
    review = ReviewAgent(
        llm=llm,
        max_steps=agent_cfg.get("review", {}).get("max_steps", 8),
        max_tokens=agent_cfg.get("review", {}).get("max_tokens", 4096),
        run_mode=_get_run_mode("ReviewAgent"),
    )
    review.memory_manager = _build_agent_memory("review", config)

    # --- BuildAgent ---
    build = BuildAgent(
        llm=llm,
        max_steps=agent_cfg.get("build", {}).get("max_steps", 8),
        max_tokens=agent_cfg.get("build", {}).get("max_tokens", 4096),
        run_mode=_get_run_mode("BuildAgent"),
    )
    build.memory_manager = _build_agent_memory("build", config)

    # 注册子 Agent 到主 Agent
    master.register_sub_agent("LiteratureAgent", literature)
    master.register_sub_agent("WritingAgent", writing)
    master.register_sub_agent("CitationAgent", citation)
    master.register_sub_agent("ReviewAgent", review)
    master.register_sub_agent("BuildAgent", build)

    return {
        "master": master,
        "literature": literature,
        "writing": writing,
        "citation": citation,
        "review": review,
        "build": build,
    }


def _setup_logging() -> Path:
    """配置日志（控制台 + 文件），返回日志文件路径"""
    logs_dir = _PROJECT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    today = date.today().strftime("%Y_%m_%d")
    # 找到今天第 N 次运行
    existing = [int(m.group(1)) for f in logs_dir.iterdir()
                if (m := re.match(rf"{today}_(\d+)\.log", f.name))]
    run_no = max(existing) + 1 if existing else 1
    log_path = logs_dir / f"{today}_{run_no}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    # 并行设置结构化 JSONL 日志
    structured_path = setup_structured_logging(log_path)
    logger.info(f"日志文件: {log_path} | 结构化日志: {structured_path}",
                extra={"event": "startup", "agent": "root"})
    return log_path


def main():
    log_path = _setup_logging()
    logger.info(f"日志文件: {log_path}")

    # 解析命令行参数
    config_path = "config.yaml"
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]

    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path)

    # 初始化 LLM
    llm_cfg = config.get("llm", {})
    configure_tool_llm(llm_cfg.get("flash_model"))
    llm = LLM(model=llm_cfg.get("model"))

    # 构建上下文
    context = build_context(config)

    # 构建 Agent（含记忆注入）
    agents = build_agents(config, llm)

    # 注入 context 到各 Agent
    master = agents["master"]
    master.context = context.view(
        readable={
            "seed_pdf_paths", "ref_pdf_paths", "output_dir",
            "template_dir", "innovation_points",
            "experiment_description", "experiment_images", "formula_manuscript",
            "user_prompt", "reference_library", "sections", "main_tex_path",
            "modification_log",
        },
        writable={},
    )

    literature = agents["literature"]
    literature.context = context.view(
        readable={"seed_pdf_paths", "ref_pdf_paths", "output_dir",
                  "reference_library", "library_dir"},
        writable={"reference_library", "modification_log"},
    )

    writing = agents["writing"]
    writing.context = context.view(
        readable={
            "output_dir", "template_dir", "innovation_points",
            "experiment_description", "experiment_images", "formula_manuscript",
            "user_prompt", "reference_library", "sections", "main_tex_path",
        },
        writable={"sections", "main_tex_path", "modification_log"},
    )

    citation = agents["citation"]
    citation.context = context.view(
        readable={"main_tex_path", "reference_library", "output_dir"},
        writable={"modification_log"},
    )

    review = agents["review"]
    review.context = context.view(
        readable={"main_tex_path", "output_dir", "sections", "reference_library"},
        writable={"modification_log"},
    )

    build = agents["build"]
    build.context = context.view(
        readable={"template_dir", "main_tex_path", "output_dir"},
        writable={"template_validated", "modification_log"},
    )

    # 交互循环
    logger.info("=" * 60)
    logger.info("  writing_assistant — 论文写作助手")
    logger.info("  输入 'quit' 退出，输入需求开始写作")
    logger.info("=" * 60)

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("Goodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            logger.info("Goodbye.")
            break

        try:
            result = master.run(user_input)
            logger.info(f"[MasterAgent] {result}")
        except KeyboardInterrupt:
            logger.info("任务已中断。")
            continue

    # 退出前保存文献库（跨任务累积；即使中途异常也尽量落盘）
    if getattr(context, "library_dir", "") and context.reference_library:
        try:
            n = len(save_library(context.library_dir, context.reference_library))
            logger.info(f"已保存文献库 {n} 条到 {context.library_dir}")
        except Exception as e:
            logger.warning(f"保存文献库失败: {e}")

    # 退出前保存长期记忆
    for agent_name, agent in agents.items():
        if agent.memory_manager:
            agent.memory_manager.save()
            logger.info(f"已保存 {agent_name} 长期记忆")


if __name__ == "__main__":
    main()