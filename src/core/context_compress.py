"""
上下文压缩模块 — 将过长的对话历史压缩为结构化摘要。

供 Agent 类在 ReAct 循环中调用，避免上下文窗口溢出。
支持渐进式两档压缩（structured 70% / aggressive 90%），
LLM 压缩失败时自动 fallback 到规则型摘要。
"""

import json
import logging
from collections import Counter
import re
from .message import Message

# 当有其它程序 import context_compress 时，在文件context_compress.py中, __name__的值为context_compress
logger = logging.getLogger(__name__)

# token 计数：使用 DeepSeek V4 官方 BPE 词表精确计数，不可用时自动回退到启发式
# 实现见 src/core/token_counter.py（懒加载单例 + fallback）
from .token_counter import estimate_tokens

# 产生大内容输出的工具集合（压缩时只保留路径，不保留原始内容）
LARGE_CONTENT_TOOLS = {
    "read_file", "parse_pdf", "get_paper_text", "read_context",
    "compile_latex", "parse_latex_log",
}


def serialize_for_compression(messages: list) -> str:
    """将消息列表序列化为结构化元数据，不保留大文件原始内容。

    策略：
    - 大内容工具（read_file, parse_pdf 等）：只保留文件路径 + 大小 + 首行摘要
    - 写入工具：保留文件路径 + 模式 + 大小
    - 搜索工具：保留查询 + 结果数量
    - 其他工具：保留完整结果（通常较短）
    - LLM 文本回复：保留前 300 字符
    """
    lines = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = msg.role
            content = msg.content

        if isinstance(content, str):
            lines.append(f"[{role}]: {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type", "")
                if t == "text":
                    text = block.get("text", "") or ""
                    lines.append(f"[{role}/text]: {text[:300]}")
                elif t == "tool_use":
                    name = block.get("name", "")
                    inp = block.get("input", {})
                    inp_str = json.dumps(inp, ensure_ascii=False)
                    if name in LARGE_CONTENT_TOOLS:
                        file_path = inp.get("file_path") or inp.get("pdf_path") or inp.get("tex_path") or ""
                        lines.append(f"[{role}/tool_use]: {name}(path={file_path})")
                    elif name == "write_file":
                        fp = inp.get("file_path", "")
                        md = inp.get("mode", "write")
                        sz = len(inp.get("content", "") or "")
                        lines.append(f"[{role}/tool_use]: write_file(path={fp}, mode={md}, content_size={sz})")
                    elif name in ("search_papers", "verify_paper", "search_template"):
                        lines.append(f"[{role}/tool_use]: {name}({inp_str[:200]})")
                    else:
                        lines.append(f"[{role}/tool_use]: {name}({inp_str[:300]})")
                elif t == "tool_result":
                    raw = str(block.get("content", ""))
                    lines.append(f"[{role}/tool_result]: {summarize_tool_result(raw)}")
                elif t == "thinking":
                    lines.append(f"[{role}/thinking]: {block.get('thinking', '')[:200]}")
    return "\n".join(lines)


def summarize_tool_result(raw: str) -> str:
    """智能压缩工具结果：保留关键信息，丢弃大段原始内容。

    对 read_file 返回：保留文件路径 + 总大小
    对 parse_pdf 返回：保留标题 + 作者 + 状态
    对 write_file 返回：保留完整结果（通常很短）
    对其他：保留前 200 字符
    """
    if not raw:
        return "(empty)"
    if raw.startswith("Error:"):
        return raw[:200]
    if "文件已写入" in raw or "文件已追加" in raw or "文件已替换" in raw:
        return raw[:150]
    if "文件已删除" in raw or "已删除" in raw:
        return raw[:150]
    if "total" in raw and "chars" in raw and "truncated" in raw:
        lines = raw.split("\n")
        first = ""
        for line in lines:
            if line.strip() and not line.startswith("提示："):
                first = line.strip()[:200]
                break
        total_info = ""
        for line in lines:
            if "total" in line and "chars" in line:
                total_info = " " + line.strip()
                break
        return f"{first}{total_info}"
    if "BibTeX entries" in raw or "参考文献" in raw:
        return raw[:200]
    return raw[:200]


def summarize_for_compression(llm, serialized_text: str,
                              agent_name: str = "",
                              mode: str = "structured") -> str:
    """调用 LLM 将旧消息压缩为摘要。

    mode:
      - "structured" (70% 阈值): 保留文件路径指针，Agent 需要时可重新 read_file
      - "aggressive"  (90% 阈值): 尽可能浓缩，只保留最关键的决策和文件列表
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
        )
        result = result.strip()
        if not result:
            result = fallback_summary(serialized_text)
        return result
    except Exception as e:
        logger.warning(f"[{agent_name}] 压缩 LLM 调用失败: {e}",
                   extra={"event": "compression_error", "agent": agent_name})
        return fallback_summary(serialized_text)


def fallback_summary(serialized_text: str) -> str:
    """规则型 fallback：当 LLM 压缩失败或返回空时，从序列化文本中提取关键信息"""
    # 输出实例：
    # [规则兜底摘要]
    # 工具调用: read_file×2, write_file×1, parse_pdf×1
    # 涉及文件: ./main.tex, ./ref.bib, ./paper.pdf
    # 错误: Error: file not found ./tmp.tex
    lines = serialized_text.split("\n")
    tool_names: list[str] = []
    file_paths: list[str] = []
    errors: list[str] = []

    for line in lines:
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
        tc = Counter(tool_names)
        parts.append("工具调用: " + ", ".join(f"{name}×{cnt}" for name, cnt in tc.most_common(8)))
    if file_paths:
        parts.append("涉及文件: " + ", ".join(file_paths[-8:]))
    if errors:
        parts.append("错误: " + "; ".join(errors[-3:]))

    return "[规则兜底摘要]\n" + "\n".join(parts) if parts else "[规则兜底摘要] 无可用信息"


def compress_history(history: list[Message], keep_recent: int, context_window: int,
                     llm, agent_name: str) -> list:
    """压缩 _history（Agent 内部历史消息列表），返回新列表。

    使用渐进式阈值：70% structured / 90% aggressive。
    """
    if not history or len(history) <= keep_recent:
        return history

    estimated = estimate_tokens(history)
    structured_threshold = int(context_window * 0.70)
    aggressive_threshold = int(context_window * 0.90)

    if estimated < structured_threshold:
        return history

    old_messages = history[:-keep_recent]
    recent_messages = history[-keep_recent:]

    old_text = serialize_for_compression(old_messages)

    if estimated >= aggressive_threshold:
        mode = "aggressive"
    else:
        mode = "structured"

    summary = summarize_for_compression(llm, old_text, agent_name=agent_name, mode=mode)

    logger.info(
        f"[{agent_name}] _history 压缩 ({mode}): {len(old_messages)}条 → 摘要 "
        f"({len(old_text)} → {len(summary)} 字符, estimated={estimated} tokens, "
        f"threshold={structured_threshold})",
        extra={"event": "compression", "agent": agent_name, "mode": mode,
               "old_count": len(old_messages), "new_chars": len(summary),
               "tokens": estimated, "threshold": structured_threshold},
    )

    return [
        Message(f"[上下文摘要 — 之前步骤的关键信息]\n\n{summary}", "user")
    ] + recent_messages

# messages: list[dict] 中的每个 dict 都包含 role 和 content 键
def compress_messages(messages: list[dict], keep_recent: int, context_window: int,
                      llm, agent_name: str,
                      anchor_input_tokens: int | None = None,
                      last_output_tokens: int = 0) -> list:
    """压缩 step 循环中的 messages（dict 列表），返回新列表，不修改原列表。

    Token 估算策略（与 DeepSeek harness 的 anchor 模式对齐）：
      - anchor 可用时：estimated = anchor_input_tokens + last_output_tokens
        （前者是上轮 API 真实 input_tokens，覆盖了上轮历史 + system + tools；
         后者是上轮模型回复的 token 数。两者相加约等于本轮历史 + 上轮回复，
         更接近本轮实际 input 大小，比本地 estimate_tokens 准确得多）
      - anchor 不可用时：fallback 到本地 estimate_tokens(messages)
        （首次调用前，或压缩后 anchor 失效，或 response.usage 缺失时）

    Args:
        anchor_input_tokens: 上轮 API 返回的 response.usage.input_tokens。
            None 表示无可用 anchor（首次调用/压缩后/调用失败）。
        last_output_tokens: 上轮 API 返回的 response.usage.output_tokens。
            anchor 可用时必填，默认 0。
    """
    if len(messages) <= keep_recent:
        return messages

    # ---- Token 估算：anchor 优先，本地 fallback ----
    if anchor_input_tokens is not None and anchor_input_tokens > 0:
        # 用真实 usage 作 baseline，加上轮 output（即将进入本轮历史）
        estimated = anchor_input_tokens + (last_output_tokens or 0)
        estimate_source = "anchor"  # 日志用
    else:
        estimated = estimate_tokens(messages)
        estimate_source = "local"  # 日志用

    structured_threshold = int(context_window * 0.70)
    aggressive_threshold = int(context_window * 0.90)

    if estimated < structured_threshold:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    old_text = serialize_for_compression(old_messages)

    if estimated >= aggressive_threshold:
        mode = "aggressive"
    else:
        mode = "structured"

    summary = summarize_for_compression(llm, old_text, agent_name=agent_name, mode=mode)

    logger.info(
        f"[{agent_name}] messages 压缩 ({mode}, source={estimate_source}): "
        f"{len(old_messages)}条 → 摘要 "
        f"({len(old_text)} → {len(summary)} 字符, estimated={estimated} tokens, "
        f"threshold={structured_threshold})",
        extra={"event": "compression", "agent": agent_name, "mode": mode,
               "old_count": len(old_messages), "new_chars": len(summary),
               "tokens": estimated, "threshold": structured_threshold,
               "source": estimate_source},
    )

    compressed = [
        {"role": "user", "content": f"[上下文摘要 — 之前步骤的关键信息]\n\n{summary}"}
    ]
    compressed.extend(recent_messages)
    return compressed
