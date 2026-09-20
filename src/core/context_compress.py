"""上下文压缩 — 按 s08 四步管线将过长对话压缩到上下文窗口内。

四步：
  1. 路径指针（落盘占位）：旧段大 tool_result 替换为文件路径指针（确定性、零 LLM）
  2. 安全切点：回退切点保护 tool_use/tool_result 配对
  3. micro_compact：指针化后重新估算，已降到阈值以下则直接返回（免 LLM 摘要）
  4. LLM 摘要兜底：structured（70% 阈值）/ aggressive（90% 阈值）两档，失败走规则摘要

另含 reactive_compact：API 报上下文超限后的紧急压缩（全库指针化 + aggressive 摘要）。
"""

import json
import logging
import re
from collections import Counter

from .message import Message
from .token_counter import estimate_tokens

logger = logging.getLogger(__name__)

# 大内容工具集合：结果可由文件重新获得，压缩时只留路径指针
LARGE_CONTENT_TOOLS = {
    "read_file", "parse_pdf", "get_paper_text", "read_context",
    "compile_latex", "parse_latex_log",
}

# 路径指针替换的最小结果长度：小于此值不值得替换
STUB_MIN_CHARS = 400

# API 上下文超限错误特征串（小写匹配）
CONTEXT_OVERFLOW_PATTERNS = (
    "prompt is too long", "prompt too long",
    "context_length_exceeded", "maximum context length",
    "context length exceeded", "input length exceeds",
    "request too large", "too many tokens",
)


def _get_content(msg) -> tuple:
    """兼容 dict / Message 两种消息形态，取 (role, content)。

    paras:
        msg: dict 或 Message 对象
    return: (role, content) 元组
    """
    if isinstance(msg, dict):
        return msg.get("role", "unknown"), msg.get("content", "")
    return msg.role, msg.content


def serialize_for_compression(messages: list) -> str:
    """把消息列表序列化为结构化元数据文本（供 LLM 摘要用）。

    大内容工具只留路径，写入工具留路径+模式+大小，文本块截 300 字符。

    paras:
        messages: 消息列表（dict 或 Message 混合）
    return: 逐行 "[role/类型]: 内容" 的文本
    """
    lines = []
    for msg in messages:
        role, content = _get_content(msg)
        if isinstance(content, str):
            lines.append(f"[{role}]: {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type", "")
                if t == "text":
                    lines.append(f"[{role}/text]: {(block.get('text', '') or '')[:300]}")
                elif t == "tool_use":
                    lines.append(f"[{role}/tool_use]: {_serialize_tool_use(block)}")
                elif t == "tool_result":
                    raw = str(block.get("content", ""))
                    lines.append(f"[{role}/tool_result]: {summarize_tool_result(raw)}")
                elif t == "thinking":
                    lines.append(f"[{role}/thinking]: {block.get('thinking', '')[:200]}")
    return "\n".join(lines)


def _serialize_tool_use(block: dict) -> str:
    """按工具类型序列化 tool_use block 的关键参数。

    paras:
        block: tool_use block dict
    return: 如 "read_file(path=xxx)" 的短描述
    """
    name = block.get("name", "")
    inp = block.get("input", {}) or {}
    inp_str = json.dumps(inp, ensure_ascii=False)
    if name in LARGE_CONTENT_TOOLS:
        path = inp.get("file_path") or inp.get("pdf_path") or inp.get("tex_path") or ""
        return f"{name}(path={path})"
    if name == "write_file":
        return (f"write_file(path={inp.get('file_path', '')}, "
                f"mode={inp.get('mode', 'write')}, content_size={len(inp.get('content', '') or '')})")
    if name in ("search_papers", "verify_paper"):
        return f"{name}({inp_str[:200]})"
    return f"{name}({inp_str[:300]})"


def summarize_tool_result(raw: str) -> str:
    """压缩单条工具结果文本：保留关键信息，丢弃大段原始内容。

    paras:
        raw: 工具原始输出
    return: 截取后的短摘要（通常 <= 200 字符）
    """
    if not raw:
        return "(empty)"
    if raw.startswith("Error:"):
        return raw[:200]
    if "已写入" in raw or "已追加" in raw or "已替换" in raw or "已删除" in raw:
        return raw[:150]
    if "total" in raw and "chars" in raw and "truncated" in raw:
        first = ""
        total_info = ""
        for line in raw.split("\n"):
            stripped = line.strip()
            if not first and stripped and not line.startswith("提示："):
                first = stripped[:200]
            if "total" in line and "chars" in line:
                total_info = " " + stripped
        return f"{first}{total_info}"
    return raw[:200]


def summarize_for_compression(llm, serialized_text: str,
                              agent_name: str = "",
                              mode: str = "structured") -> str:
    """调用 LLM 把序列化文本压缩为摘要，失败回退规则摘要。

    paras:
        llm: LLM 实例
        serialized_text: serialize_for_compression 的输出
        agent_name: Agent 名（日志用）
        mode: structured 保留文件路径指针 / aggressive 极度浓缩
    return: 摘要文本
    """
    if mode == "aggressive":
        system = "你是一个对话压缩助手。请极度精简地总结对话历史，只保留最重要的信息。"
        prompt = (
            "请将以下对话历史压缩为一段极度精简的摘要。只保留关键决策和文件路径，丢弃所有细节。\n\n"
            "按以下格式输出（纯文本，不要用 markdown 标题）：\n"
            "关键决策: [1-2 句话]\n"
            "涉及文件: [文件路径列表]\n"
            "待处理: [1 句话，没有则写 无]\n\n"
            "对话历史:\n"
            f"{serialized_text}"
        )
    else:
        system = "你是一个对话压缩助手。请简洁准确地总结对话历史，保留文件路径以支持 Agent 按需重新读取。"
        prompt = (
            "请将以下对话历史压缩为一段结构化摘要。保留关键信息，丢弃冗余细节。\n"
            "重要：对 read_file/parse_pdf 等操作，只保留文件路径和内容概要，"
            "Agent 需要详细内容时会重新读取文件。\n\n"
            "按以下格式输出（纯文本，不要用 markdown 标题）：\n"
            "已完成的操作:\n"
            "  - [具体操作]\n"
            "涉及的文件:\n"
            "  - [文件路径]: [内容概要，如'main.tex: 46300字符, 4章节, \\bibliography{references}']\n"
            "关键决策:\n"
            "  - [决策]\n"
            "待处理:\n"
            "  - [未完成事项，没有则写 无]\n\n"
            "对话历史:\n"
            f"{serialized_text}"
        )

    try:
        result = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=512,
        ).strip()
        return result or fallback_summary(serialized_text)
    except Exception as e:
        logger.warning(f"[{agent_name}] 压缩 LLM 调用失败: {e}",
                       extra={"event": "compression_error", "agent": agent_name})
        return fallback_summary(serialized_text)


def fallback_summary(serialized_text: str) -> str:
    """规则型兜底摘要：从序列化文本正则提取工具调用、文件、错误。

    paras:
        serialized_text: serialize_for_compression 的输出
    return: "[规则兜底摘要]" 开头的文本
    """
    tool_names: list[str] = []
    file_paths: list[str] = []
    errors: list[str] = []
    for line in serialized_text.split("\n"):
        m = re.search(r"tool_use\]:\s*(\w+)", line)
        if m:
            tool_names.append(m.group(1))
        m = re.search(r"path=([^\s,)]+)", line)
        if m:
            p = m.group(1).strip("'\"")
            if p not in file_paths:
                file_paths.append(p)
        if "Error:" in line:
            errors.append(line.strip()[:120])

    parts = []
    if tool_names:
        counts = Counter(tool_names)
        parts.append("工具调用: " + ", ".join(f"{n}×{c}" for n, c in counts.most_common(8)))
    if file_paths:
        parts.append("涉及文件: " + ", ".join(file_paths[-8:]))
    if errors:
        parts.append("错误: " + "; ".join(errors[-3:]))
    return "[规则兜底摘要]\n" + "\n".join(parts) if parts else "[规则兜底摘要] 无可用信息"


def _stub_large_tool_results(messages: list[dict], keep_recent: int) -> list:
    """步骤 1 — 路径指针：把旧段大 tool_result 替换为文件路径指针。

    结果已被模型消费过且文件仍在磁盘，指针化即可；只改 content 不增删消息，
    配对完整保留。copy-on-write，无替换返回原列表。

    paras:
        messages: dict 消息列表
        keep_recent: 最近 N 条不动；0 表示全库替换（reactive_compact 用）
    return: 替换后的新列表；无替换时返回原列表
    """
    tool_use_info: dict = {}
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_use_info[b.get("id")] = (b.get("name", ""), b.get("input", {}) or {})

    end = len(messages) - keep_recent if keep_recent > 0 else len(messages)
    result = list(messages)
    changed = False
    for i in range(end):
        msg = result[i]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_blocks = None
        for j, block in enumerate(content):
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            raw = block.get("content", "")
            raw_str = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            if len(raw_str) < STUB_MIN_CHARS:
                continue
            name, inp = tool_use_info.get(block.get("tool_use_id"), ("", {}))
            if name and name not in LARGE_CONTENT_TOOLS:
                continue
            path = inp.get("file_path") or inp.get("pdf_path") or inp.get("tex_path") or ""
            loc = f"文件 {path} " if path else ""
            pointer = (f"[{name or '工具'} 结果已省略 — {loc}共 {len(raw_str)} 字符；"
                       f"需要时重新调用 {name or '相应工具'}]")
            if new_blocks is None:
                new_blocks = list(content)
            new_blocks[j] = {**block, "content": pointer}
        if new_blocks is not None:
            result[i] = {**msg, "content": new_blocks}
            changed = True
    return result if changed else messages


def _safe_cut(messages: list[dict], keep_recent: int) -> int:
    """步骤 2 — 安全切点：向前回退，避免 recent 段以孤立 tool_result 开头。

    若切出的 recent 段开头是 tool_result（其配对 tool_use 被切进摘要区），
    API 会拒绝请求；向前回退把配对的 assistant(tool_use) 划入 recent 段。

    paras:
        messages: dict 消息列表
        keep_recent: 最近保留条数
    return: 安全切点；0 表示无法安全切分，调用方应跳过压缩
    """
    cut = len(messages) - keep_recent
    while 0 < cut < len(messages):
        head = messages[cut]
        content = head.get("content") if isinstance(head, dict) else None
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            cut -= 1
        else:
            break
    return max(cut, 0)


def _pick_mode(estimated: int, context_window: int) -> tuple[str, int, int]:
    """按渐进阈值选压缩档位。

    paras:
        estimated: 当前 token 估算值
        context_window: 上下文窗口大小
    return: (mode, structured_threshold, aggressive_threshold)
    """
    structured = int(context_window * 0.70)
    aggressive = int(context_window * 0.90)
    mode = "aggressive" if estimated >= aggressive else "structured"
    return mode, structured, aggressive


def compress_messages(messages: list[dict], keep_recent: int, context_window: int,
                      llm, agent_name: str,
                      anchor_input_tokens: int | None = None,
                      last_output_tokens: int = 0) -> list:
    """按四步管线压缩 step 循环中的 dict 消息列表，copy-on-write。

    token 估算优先用 anchor（上轮 API 真实 input_tokens + output_tokens），
    不可用时回退本地 estimate_tokens。

    paras:
        messages: dict 消息列表
        keep_recent: 压缩时保留的最近消息数
        context_window: 上下文窗口大小
        llm: LLM 实例（步骤 4 摘要用）
        agent_name: Agent 名（日志用）
        anchor_input_tokens: 上轮 API 真实 input_tokens；None 表示无 anchor
        last_output_tokens: 上轮 API output_tokens
    return: 压缩后的新列表；无需压缩时返回原列表
    """
    if len(messages) <= keep_recent:
        return messages

    if anchor_input_tokens is not None and anchor_input_tokens > 0:
        estimated = anchor_input_tokens + (last_output_tokens or 0)
        estimate_source = "anchor"
    else:
        estimated = estimate_tokens(messages)
        estimate_source = "local"

    mode, structured_threshold, _ = _pick_mode(estimated, context_window)
    if estimated < structured_threshold:
        return messages

    # 步骤 1 + 3：指针化旧段大 tool_result，重新估算，降到阈值以下即免摘要
    stubbed = _stub_large_tool_results(messages, keep_recent)
    if stubbed is not messages:
        estimated = estimate_tokens(stubbed)
        estimate_source = "local+stub"
        messages = stubbed
        if estimated < structured_threshold:
            logger.info(
                f"[{agent_name}] 路径指针替换后免压缩: estimated={estimated} tokens "
                f"< threshold={structured_threshold}",
                extra={"event": "stub_compact", "agent": agent_name, "tokens": estimated},
            )
            return messages

    # 步骤 2：安全切点
    cut = _safe_cut(messages, keep_recent)
    if cut <= 0:
        return messages
    old_messages, recent_messages = messages[:cut], messages[cut:]

    # 步骤 4：LLM 摘要兜底
    mode, structured_threshold, _ = _pick_mode(estimated, context_window)
    old_text = serialize_for_compression(old_messages)
    summary = summarize_for_compression(llm, old_text, agent_name=agent_name, mode=mode)

    logger.info(
        f"[{agent_name}] messages 压缩 ({mode}, source={estimate_source}): "
        f"{len(old_messages)}条 → 摘要 ({len(old_text)} → {len(summary)} 字符, "
        f"estimated={estimated} tokens)",
        extra={"event": "compression", "agent": agent_name, "mode": mode,
               "old_count": len(old_messages), "new_chars": len(summary),
               "tokens": estimated, "threshold": structured_threshold,
               "source": estimate_source},
    )

    compressed = [{"role": "user",
                   "content": f"[上下文摘要 — 之前步骤的关键信息]\n\n{summary}"}]
    compressed.extend(recent_messages)
    return compressed


def compress_history(history: list[Message], keep_recent: int, context_window: int,
                     llm, agent_name: str) -> list:
    """按四步管线压缩 Agent 的 _history（Message 对象列表）。

    paras:
        history: Message 对象列表
        keep_recent: 压缩时保留的最近消息数
        context_window: 上下文窗口大小
        llm: LLM 实例
        agent_name: Agent 名（日志用）
    return: 压缩后的列表；无需压缩时返回原列表
    """
    if not history or len(history) <= keep_recent:
        return history

    estimated = estimate_tokens(history)
    mode, structured_threshold, _ = _pick_mode(estimated, context_window)
    if estimated < structured_threshold:
        return history

    old_messages, recent_messages = history[:-keep_recent], history[-keep_recent:]
    old_text = serialize_for_compression(old_messages)
    summary = summarize_for_compression(llm, old_text, agent_name=agent_name, mode=mode)

    logger.info(
        f"[{agent_name}] _history 压缩 ({mode}): {len(old_messages)}条 → 摘要 "
        f"({len(old_text)} → {len(summary)} 字符, estimated={estimated} tokens)",
        extra={"event": "compression", "agent": agent_name, "mode": mode,
               "old_count": len(old_messages), "new_chars": len(summary),
               "tokens": estimated, "threshold": structured_threshold},
    )

    return [Message(f"[上下文摘要 — 之前步骤的关键信息]\n\n{summary}", "user")] + recent_messages


def is_context_overflow_error(e: Exception) -> bool:
    """判断异常是否为 API 上下文超限。

    paras:
        e: 捕获到的异常
    return: 是上下文超限错误返回 True
    """
    text = f"{type(e).__name__}: {e}".lower()
    return any(p in text for p in CONTEXT_OVERFLOW_PATTERNS)


def reactive_compact(messages: list[dict], llm, agent_name: str,
                     keep_recent: int = 5) -> list:
    """紧急压缩：API 上下文超限后调用（全库指针化 + 安全切分 + aggressive 摘要）。

    paras:
        messages: dict 消息列表
        llm: LLM 实例
        agent_name: Agent 名（日志用）
        keep_recent: 摘要后保留的最近消息数
    return: 压缩后的消息列表（调用方应重试一次）
    """
    logger.warning(
        f"[{agent_name}] reactive_compact: 上下文超限，紧急压缩 {len(messages)} 条消息",
        extra={"event": "reactive_compact", "agent": agent_name, "old_count": len(messages)},
    )
    stubbed = _stub_large_tool_results(messages, keep_recent=0)
    cut = _safe_cut(stubbed, keep_recent)
    if cut <= 0:
        return stubbed
    old_text = serialize_for_compression(stubbed[:cut])
    summary = summarize_for_compression(llm, old_text, agent_name=agent_name, mode="aggressive")
    compressed = [{"role": "user",
                   "content": f"[紧急压缩摘要 — 因上下文超限触发]\n\n{summary}"}]
    compressed.extend(stubbed[cut:])
    logger.warning(
        f"[{agent_name}] reactive_compact 完成: {len(messages)} → {len(compressed)} 条",
        extra={"event": "reactive_compact", "agent": agent_name, "new_count": len(compressed)},
    )
    return compressed
