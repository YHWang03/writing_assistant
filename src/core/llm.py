"""
LLM 调用层 — DeepSeek（Anthropic 兼容接口）

提供两个核心方法：
  - chat: 纯文本调用
  - chat_with_tools: 带工具调用（返回原始 response，含 tool_use blocks）
"""

import os
from pathlib import Path
from anthropic import Anthropic


def _find_and_load_dotenv():
    """从当前目录向上查找 .env 文件并加载环境变量。"""
    if getattr(_find_and_load_dotenv, "_loaded", False):
        # if _find_and_load_dotenv._loaded is True
        return

    search_dir = Path.cwd()
    for _ in range(6):
        env_path = search_dir / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key, val = key.strip(), val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
            break
        search_dir = search_dir.parent

    _find_and_load_dotenv._loaded = True


class LLM:
    """
    DeepSeek LLM 客户端（Anthropic 兼容接口）

    使用：
        llm = LLM()                                    # 自动读 .env
        llm = LLM(model="deepseek-v4-pro")             # 手动指定模型
        llm = LLM(api_key="sk-xxx", base_url="...")    # 手动指定连接
    """

    def __init__(self, model: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None):
        _find_and_load_dotenv()
        self.model = model or os.getenv("LLM_MODEL_NAME", "deepseek-v4-pro")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_URL")

        if not api_key or not base_url:
            raise ValueError(
                "LLM_API_KEY 和 LLM_URL 必须在 .env 中配置或作为参数传入。"
            )

        self._client = Anthropic(api_key=api_key, base_url=base_url)

    # ---- 核心调用方法 ----

    def chat(self, messages: list[dict], system: str = "",
             max_tokens: int = 4096) -> str:
        """纯文本调用（不带工具），返回模型文本回答。"""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        parts = [b.text for b in response.content if b.type == "text"]
        return "\n".join(parts)

    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        system: str = "", max_tokens: int = 4096):
        """
        带工具调用，返回原始的 Messages API response 对象。

        调用方需自行处理 response.content 中的：
          - thinking blocks
          - text blocks
          - tool_use blocks
        """
        return self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

    def __repr__(self) -> str:
        return f"LLM(model={self.model})"


# ---- 工具 LLM 工厂 ----

_tool_llm: LLM | None = None
_tool_llm_model: str | None = None


def configure_tool_llm(model: str | None = None):
    """用配置文件里的 flash_model 覆盖工具 LLM 的模型。

    优先级：config（flash_model）> 环境变量 TOOL_LLM_MODEL_NAME > 默认 deepseek-v4-flash。
    在 main() 加载配置后调用；传入 None 则回退到环境变量。
    """
    global _tool_llm_model, _tool_llm
    _tool_llm_model = model
    _tool_llm = None  # 已配置则下次重新创建，避免沿用旧模型


def get_tool_llm() -> LLM:
    """获取工具内部使用的 LLM 实例（单例，按需创建）。"""
    global _tool_llm
    if _tool_llm is None:
        _find_and_load_dotenv()
        model = _tool_llm_model or os.getenv("TOOL_LLM_MODEL_NAME", "deepseek-v4-flash")
        _tool_llm = LLM(model=model)
    return _tool_llm
