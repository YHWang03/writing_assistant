"""LLM 调用层 — DeepSeek（Anthropic 兼容接口），chat 纯文本 + chat_with_tools 工具调用"""

import os
from pathlib import Path
from anthropic import Anthropic


def _find_and_load_dotenv():
    """从当前目录向上查找 .env 并加载环境变量（进程内一次）。

    paras: 无
    return: 无
    """
    if getattr(_find_and_load_dotenv, "_loaded", False):
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
    """DeepSeek LLM 客户端（Anthropic 兼容接口）"""

    def __init__(self, model: str | None = None,
                 api_key: str | None = None,
                 base_url: str | None = None):
        """初始化客户端，连接信息取参数或 .env 环境变量。

        paras:
            model: 模型名，默认环境变量 LLM_MODEL_NAME
            api_key: API 密钥，默认环境变量 LLM_API_KEY
            base_url: 接口地址，默认环境变量 LLM_URL
        return: 无
        """
        _find_and_load_dotenv()
        self.model = model or os.getenv("LLM_MODEL_NAME", "deepseek-v4-pro")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_URL")

        if not api_key or not base_url:
            raise ValueError(
                "LLM_API_KEY 和 LLM_URL 必须在 .env 中配置或作为参数传入。"
            )

        self._client = Anthropic(api_key=api_key, base_url=base_url)

    def chat(self, messages: list[dict], system: str = "",
             max_tokens: int = 4096) -> str:
        """纯文本调用。

        paras:
            messages: dict 消息列表
            system: system prompt
            max_tokens: 最大输出 token 数
        return: 模型文本回答
        """
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
        """带工具调用，返回原始 response（content 含 thinking/text/tool_use blocks）。

        paras:
            messages: dict 消息列表
            tools: Anthropic 格式工具定义列表
            system: system prompt
            max_tokens: 最大输出 token 数
        return: Messages API response 对象
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


_tool_llm: LLM | None = None
_tool_llm_model: str | None = None


def configure_tool_llm(model: str | None = None):
    """配置工具 LLM 的模型（main 加载 config 后调用）。

    paras:
        model: flash 模型名；None 回退环境变量 TOOL_LLM_MODEL_NAME
    return: 无
    """
    global _tool_llm_model, _tool_llm
    _tool_llm_model = model
    _tool_llm = None


def get_tool_llm() -> LLM:
    """获取工具内部 LLM 单例（按需创建）。

    paras: 无
    return: LLM 实例
    """
    global _tool_llm
    if _tool_llm is None:
        _find_and_load_dotenv()
        model = _tool_llm_model or os.getenv("TOOL_LLM_MODEL_NAME", "deepseek-v4-flash")
        _tool_llm = LLM(model=model)
    return _tool_llm
