"""LLM 调用层 — DeepSeek（Anthropic 兼容接口），chat 纯文本 + chat_with_tools 工具调用"""

import os
from pathlib import Path
from anthropic import Anthropic
from time import perf_counter
from uuid import uuid4

from ..observability.telemetry import TOKEN_FIELDS, record_call, call_scope
from ..observability.tracing import span, save_detail
from ..observability.http_client import TracedHttpClient


class IncompleteResponseError(RuntimeError):
    """输出预算耗尽；不向调用者暴露可能不完整的文本或工具参数。"""




def _find_and_load_dotenv():
    """从当前目录向上查找 .env 并加载环境变量（进程内一次）。"""
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
                 base_url: str | None = None,
                 thinking: bool | None = None):
        """初始化客户端，连接信息取参数或 .env 环境变量。

        paras:
            model: 模型名，默认环境变量 LLM_MODEL_NAME
            api_key: API 密钥，默认环境变量 LLM_API_KEY
            base_url: 接口地址，默认环境变量 LLM_URL
            thinking: chat 的默认思考开关；True 开启，False 关闭，None 不传开关，使用服务端默认值。
                chat_with_tools 不使用此默认值。
        return: 无
        """
        _find_and_load_dotenv()
        self.model = model or os.getenv("LLM_MODEL_NAME", "deepseek-v4-pro")
        self.default_thinking = thinking
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_URL")

        if not api_key or not base_url:
            raise ValueError(
                "LLM_API_KEY 和 LLM_URL 必须在 .env 中配置或作为参数传入。"
            )

        self._client = Anthropic(api_key=api_key, base_url=base_url, http_client=TracedHttpClient())

    def chat(self, messages: list[dict], system: str = "",
             max_tokens: int = 4096, *, truncation_retries: int = 1,
             thinking: bool | None = None) -> str:
        """调用模型并仅拼接 text 内容块；不将 thinking 内容作为返回文本。

        paras:
            messages: dict 消息列表
            system: system prompt
            max_tokens: 最大输出 token 数
            truncation_retries: 输出因 max_tokens 截断后的额外重试次数，默认 1；0 表示不重试。
                重试不增加输出上限，仍截断则抛出 IncompleteResponseError；与 SDK 网络重试不同。
            thinking: 本次思考开关；None 继承构造时的设置，继承后仍为 None 则使用服务端默认值。
        return: 模型文本回答
        """
        if thinking is None:
            thinking = getattr(self, "default_thinking", None)
        options = {} if thinking is None else {"thinking": {"type": "enabled" if thinking else "disabled"}}
        response = self._create_complete(
            truncation_retries=truncation_retries,
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **options,
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
        return self._create_complete(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )

    def _create_complete(self, *, truncation_retries=1, **kwargs):
        '''获取未因输出上限截断的响应；重试不增加预算，耗尽后抛出 IncompleteResponseError。

        paras:
            truncation_retries: 截断后的额外重试次数，应为非负整数；默认 1。
            **kwargs: 传给 Messages API 的参数，包含 model、messages、max_tokens 等。
        return: 未以 max_tokens 原因结束的原始响应；不返回已截断内容。
        '''
        request = dict(kwargs)
        # 将随机数转成32位十六进制字符串
        # 当输出被`max_tokens` 截断需要重试时，每次`_create`
        # 调用都会通过`call_scope(recovery_id=recovery_id, ...)` 带上同一个 ID，
        # 这样日志/观测层能把同一次"截断恢复"的多次 SDK 调用归为一组（`attempt=1, 2, ...` ），
        # 而不是分散的独立事件
        recovery_id = uuid4().hex # uuid: universally unique identifier
        for attempt in range(truncation_retries + 1):
            with call_scope(recovery_id=recovery_id, attempt=attempt + 1,
                            requested_max_tokens=request["max_tokens"]):
                response = self._create(**request)
            if getattr(response, "stop_reason", None) != "max_tokens":
                return response
            if attempt == truncation_retries:
                break
            request["system"] = (kwargs.get("system", "") +
                "\n上次输出超出预算。请压缩推理和解释，优先完整输出所需结果；"
                "不要省略工具必需参数，不要截断文件正文或 JSON。")
        raise IncompleteResponseError(
            f"模型输出不完整（max_tokens），共尝试 {truncation_retries + 1} 次；"
            "未采用截断内容，请缩小任务或分段生成。")

    def _create(self, **kwargs):
        '''执行一次 SDK 调用，记录请求、响应、耗时和原始 token 用量；异常原样抛出。

        paras:
            **kwargs: 直接传给客户端 messages.create 的请求参数。
        return: SDK 返回的原始响应，可能包含截断内容，由上层判断完整性。
        '''
        call_id = uuid4().hex
        with span("llm", call_id=call_id, model=self.model, timing_scope="sdk_invocation",
                  input_detail=save_detail(kwargs)) as trace:
            started = perf_counter()
            event = dict(call_id=call_id, model=self.model, status="failed",
                         usage=None, usage_source="unknown", request_id=None,
                         operation="chat_with_tools" if "tools" in kwargs else "chat")
            event.update({key: None for key in TOKEN_FIELDS})
            try:
                response = self._client.messages.create(**kwargs)
                # 记录响应状态
                event["status"] = ("incomplete" if getattr(response, "stop_reason", None)
                                   == "max_tokens" else "success")
                event["request_id"] = getattr(response, "_request_id", None)
                event["response_id"] = getattr(response, "id", None)
                event["response_model"] = getattr(response, "model", None)
                event["stop_reason"] = getattr(response, "stop_reason", None)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    raw = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else dict(usage)
                    event.update(usage=raw, usage_source="api")
                    for key in TOKEN_FIELDS:
                        value = raw.get(key)
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            event[key] = value
                event["output_detail"] = save_detail(response.model_dump(mode="json") if hasattr(response, "model_dump") else
                    {'content': [vars(v) for v in getattr(response, 'content', [])],
                     'stop_reason': getattr(response, 'stop_reason', None)})
                return response
            except BaseException as exc:
                event["error_type"] = type(exc).__name__
                event["request_id"] = getattr(exc, "request_id", None)
                event["http_status"] = getattr(exc, "status_code", None)
                raise
            finally:
                event["duration"] = perf_counter() - started
                trace.update(status=event["status"], call_id=call_id)
                record_call(event)

    def __repr__(self) -> str:
        '''生成 LLM 的简短调试表示。

        return: 对象的描述字符串。
        '''
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
    """获取工具内部 LLM 单例（按需创建）。"""
    global _tool_llm
    if _tool_llm is None:
        _find_and_load_dotenv()
        model = _tool_llm_model or os.getenv("TOOL_LLM_MODEL_NAME", "deepseek-v4-flash")
        _tool_llm = LLM(model=model, thinking=False)
    return _tool_llm
