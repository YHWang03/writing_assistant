"""Typed application configuration and project-relative path handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_MODES = frozenset({"react", "plan_execute"})
AGENT_NAMES = ("master", "literature", "writing", "citation", "review", "build")


def _memory_switches(value=None) -> dict[str, bool]:
    '''校验各 Agent 的记忆开关，默认仅开启 Master。

    paras:
        value: 待处理的配置值。
    return: 合并默认值和覆盖值后的开关字典。
    '''
    switches = {name: name == "master" for name in AGENT_NAMES}
    overrides = _mapping(value, "memory.enabled")
    for name, enabled in overrides.items():
        if name not in switches:
            raise ConfigError(f"memory.enabled 未知 Agent: {name}")
        if not isinstance(enabled, bool):
            raise ConfigError(f"memory.enabled.{name} 必须是布尔值 true/false")
        switches[name] = enabled
    return switches


class ConfigError(ValueError):
    """Raised when the YAML configuration is structurally invalid."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    '''校验配置项是否为映射，空值按空配置处理。

    paras:
        value: 待处理的配置值。
        path: 配置项名称，用于错误信息定位。
    return: 映射对象；None 转为空字典，类型错误抛出 ConfigError。
    '''
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} 必须是映射，实际为 {type(value).__name__}")
    return value


def _positive_int(value: Any, path: str, default: int) -> int:
    '''读取正整数配置并拒绝布尔值、非整数及非正数。

    paras:
        value: 待处理的配置值。
        path: 配置项名称，用于错误信息定位。
        default: 未提供值时使用的默认值。
    return: 校验后的整数，未配置时使用默认值。
    '''
    result = default if value is None else value
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise ConfigError(f"{path} 必须是正整数")
    return result


@dataclass(frozen=True)
class PathConfig:
    output_dir: str = "output"
    template_dir: str = "template"
    seed_dir: str = "refs"
    refs_dir: str = "reference"
    library_dir: str = "example/library"


@dataclass(frozen=True)
class PaperConfig:
    innovation_points: str = ""
    experiment_description: str = ""
    experiment_images: str = ""
    formula_manuscript: str = ""
    user_prompt: str = ""


@dataclass(frozen=True)
class LLMConfig:
    model: str | None = None
    flash_model: str | None = None


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int
    max_tokens: int
    min_relevant: int | None = None
    finish_reserve_steps: int = 3


@dataclass(frozen=True)
class AgentsConfig:
    run_modes: Mapping[str, str]
    master: AgentLimits
    literature: AgentLimits
    writing: AgentLimits
    citation: AgentLimits
    review: AgentLimits
    build: AgentLimits

    def run_mode_for(self, agent_name: str) -> str:
        '''取得指定 Agent 的运行模式，未覆盖时使用默认模式。

        paras:
            agent_name: 所属 Agent 的名称。
        return: 运行模式名称。
        '''
        return self.run_modes.get(agent_name, self.run_modes["default"])


@dataclass(frozen=True)
class MemoryConfig:
    enabled: Mapping[str, bool] = field(default_factory=_memory_switches)
    path: str = "memory/{agent_name}.json"
    max_records: int = 200
    recall_limit: int = 5
    consolidate_threshold: int = 10


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: AgentsConfig | None = None
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    project_dir: Path = PROJECT_DIR

    def resolve(self, value: str) -> Path:
        '''将相对配置路径拼接到项目目录，绝对路径保持不变。

        paras:
            value: 绝对路径或相对项目目录的路径字符串。
        return: 解析后的 Path。
        '''
        path = Path(value)
        return path if path.is_absolute() else self.project_dir / path


def _agent_limits(raw: Mapping[str, Any], name: str, steps: int,
                  tokens: int, *, literature: bool = False) -> AgentLimits:
    '''校验并构建单个 Agent 的步数、输出预算和检索阈值。

    paras:
        raw: 原始配置映射。
        name: 目标名称。
        steps: 默认最大执行步数。
        tokens: 默认最大输出 token 数。
        literature: 是否解析文献 Agent 专用的最少相关文献数。
    return: 对应 Agent 的限制配置。
    '''
    section = _mapping(raw.get(name), f"agents.{name}")
    min_relevant = None
    if literature:
        min_relevant = _positive_int(
            section.get("min_relevant"), "agents.literature.min_relevant", 15)
    return AgentLimits(
        max_steps=_positive_int(section.get("max_steps"), f"agents.{name}.max_steps", steps),
        max_tokens=_positive_int(section.get("max_tokens"), f"agents.{name}.max_tokens", tokens),
        min_relevant=min_relevant,
        finish_reserve_steps=_positive_int(section.get('finish_reserve_steps'),
                                          f'agents.{name}.finish_reserve_steps', 3),
    )


def _agents_config(raw_value: Any) -> AgentsConfig:
    '''解析运行模式和所有 Agent 的执行限制。

    paras:
        raw_value: 待解析的 agents 配置值。
    return: 校验后的 AgentsConfig。
    '''
    raw = _mapping(raw_value, "agents")
    run_mode_value = raw.get("run_mode", "react")
    if isinstance(run_mode_value, str):
        run_modes = {"default": run_mode_value}
    else:
        run_modes = dict(_mapping(run_mode_value, "agents.run_mode"))
        run_modes.setdefault("default", "react")
    invalid = {name: mode for name, mode in run_modes.items() if mode not in RUN_MODES}
    if invalid:
        raise ConfigError(f"不支持的运行模式: {invalid}；可选值: {sorted(RUN_MODES)}")
    return AgentsConfig(
        run_modes=run_modes,
        master=_agent_limits(raw, "master", 12, 4096),
        literature=_agent_limits(raw, "literature", 15, 4096, literature=True),
        writing=_agent_limits(raw, "writing", 10, 8192),
        citation=_agent_limits(raw, "citation", 12, 4096),
        review=_agent_limits(raw, "review", 8, 4096),
        build=_agent_limits(raw, "build", 8, 4096),
    )


def load_config(config_path: str | Path) -> AppConfig:
    '''读取并校验 YAML 配置，补充默认值；配置缺失或非法时抛出异常。

    paras:
        config_path: YAML 配置路径；相对路径以 PROJECT_DIR 为基准。
    return: 校验后的 AppConfig，project_dir 为配置文件所在目录。
    '''
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as stream:
        root = _mapping(yaml.safe_load(stream) or {}, "config")
    paths = _mapping(root.get("paths"), "paths")
    paper = _mapping(root.get("paper"), "paper")
    llm = _mapping(root.get("llm"), "llm")
    memory = _mapping(root.get("memory"), "memory")
    memory_path = str(memory.get("path", "memory/{agent_name}.json"))
    if "{agent_name}" not in memory_path:
        raise ConfigError("memory.path 必须包含 {agent_name} 占位符")
    try:
        path_config = PathConfig(**{k: str(v) for k, v in paths.items()})
        paper_config = PaperConfig(**{k: str(v) for k, v in paper.items()})
        llm_config = LLMConfig(**llm)
    except TypeError as exc:
        raise ConfigError(f"存在未知配置项: {exc}") from exc
    return AppConfig(
        paths=path_config,
        paper=paper_config,
        llm=llm_config,
        agents=_agents_config(root.get("agents")),
        memory=MemoryConfig(
            enabled=_memory_switches(memory.get("enabled")),
            path=memory_path,
            max_records=_positive_int(memory.get("max_records"), "memory.max_records", 200),
            recall_limit=_positive_int(memory.get("recall_limit"), "memory.recall_limit", 5),
            consolidate_threshold=_positive_int(
                memory.get("consolidate_threshold"), "memory.consolidate_threshold", 10),
        ),
        project_dir=path.parent.resolve(),
    )
