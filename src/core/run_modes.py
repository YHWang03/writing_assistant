"""Execution strategies used by :class:`Agent`."""

from __future__ import annotations

import logging

from ..context.context_compress import is_context_overflow_error, reactive_compact
from .message import Message
from ..observability.telemetry import attributed
from ..context.token_counter import estimate_tokens
from .utils import extract_python_list, strip_control_chars

logger = logging.getLogger(__name__)

PLAN_STEP_MAX_STEPS = 6
MAX_PLAN_STEPS = 5


class ReactRunner:
    """Anthropic tool-use loop for one agent task or plan step."""

    def run(self, agent, user_input: str, system_prompt: str,
            verbose: bool = True, record_intermediate: bool = False,
            require_finish: bool = True,
            max_react_steps: int | None = None) -> str:
        agent._sync_context_to_tools()
        tools = agent._get_tools_for_llm()
        agent._token_anchor = None
        agent._last_output_tokens = 0
        agent._compress_history()
        messages = [{"role": "user" if msg.role == "system" else msg.role,
                     "content": msg.content} for msg in agent._history]
        messages.append({"role": "user", "content": user_input})
        context_info = agent._build_context_info()
        if context_info:
            system_prompt = f"{system_prompt}\n\n---\n\n{context_info}"

        tool_calls: list[str] = []
        reactive_compacted = False
        step_limit = max_react_steps or agent.max_steps
        agent.hooks.reset_all()

        for step in range(step_limit):
            try:
                nudge = agent._trigger_hooks("pre_step", step, step_limit, require_finish)
                if nudge:
                    messages.append({"role": "user", "content": nudge})
                messages = agent._compress_messages(messages)
                try:
                    response = agent.llm.chat_with_tools(
                        messages=messages, tools=tools, system=system_prompt,
                        max_tokens=agent.max_tokens,
                    )
                except Exception as exc:
                    if not reactive_compacted and is_context_overflow_error(exc):
                        reactive_compacted = True
                        messages = reactive_compact(messages, agent.llm, agent.name)
                        agent._token_anchor = None
                        agent._last_output_tokens = 0
                        response = agent.llm.chat_with_tools(
                            messages=messages, tools=tools, system=system_prompt,
                            max_tokens=agent.max_tokens,
                        )
                    else:
                        raise

                self._record_tokens(agent, response, messages, step)
                assistant_blocks, tool_results = [], []
                finish_summary = None

                for block in response.content:
                    if block.type in {"thinking", "text"}:
                        assistant_blocks.append(block)
                        if verbose and block.type == "text":
                            logger.info("[%s] step %d text: %s", agent.name, step + 1,
                                        block.text[:200])
                        continue
                    if block.type != "tool_use":
                        continue
                    if verbose:
                        logger.info(
                            "[%s] step %d tool: %s(%s)", agent.name, step + 1,
                            block.name, block.input,
                            extra={"event": "tool_call", "agent": agent.name,
                                   "step": step + 1, "tool": block.name,
                                   "params": block.input},
                        )
                    if block.name == "finish":
                        finish_summary = block.input.get("summary", "任务完成。")
                        break
                    output = agent._execute_tool(block.name, block.input)
                    agent._trigger_hooks("post_tool_use", block, output)
                    if verbose:
                        logger.info(
                            "[%s] step %d result: %s", agent.name, step + 1,
                            strip_control_chars(output),
                            extra={"event": "tool_result", "agent": agent.name,
                                   "step": step + 1, "tool": block.name,
                                   "is_error": output.startswith("Error:"),
                                   "result_size": len(output)},
                        )
                    tool_calls.append(f"{block.name}({str(block.input)[:100]})")
                    assistant_blocks.append(block)
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id, "content": output,
                    })

                if assistant_blocks:
                    messages.append({"role": "assistant", "content": assistant_blocks})
                # Complete every tool-use pair before a finish hook can retry.
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                if finish_summary is not None:
                    nudge = agent._trigger_hooks("pre_finish", require_finish)
                    if nudge:
                        messages.append({"role": "user", "content": nudge})
                        continue
                    if verbose:
                        logger.info(
                            "[%s] step %d finish: %s", agent.name, step + 1,
                            finish_summary[:200],
                            extra={"event": "finish", "agent": agent.name,
                                   "step": step + 1, "total_steps": step + 1},
                        )
                    self._record_result(agent, user_input, finish_summary, tool_calls)
                    return finish_summary

                if tool_results:
                    if record_intermediate:
                        step_tools = [
                            f"{block.name}({str(block.input)[:100]})"
                            for block in response.content if block.type == "tool_use"
                        ]
                        agent.add_message(Message(
                            f"[Step {step + 1} tools] {', '.join(step_tools)}", "assistant"))
                        for result in tool_results:
                            agent.add_message(Message(
                                f"[Step {step + 1} result] {result['content'][:300]}", "user"))
                    continue

                texts = [block.text for block in response.content if block.type == "text"]
                result = "\n".join(texts) if texts else "(no text response)"
                nudge = agent._trigger_hooks("text_only", result, require_finish)
                if nudge:
                    messages.append({"role": "user", "content": nudge})
                    continue
                if verbose and require_finish:
                    logger.warning(
                        "[%s] step %d 无工具调用，强制结束（result: %s）",
                        agent.name, step + 1, result[:120],
                        extra={"event": "force_stop", "agent": agent.name,
                               "step": step + 1},
                    )
                self._record_result(agent, user_input, result, tool_calls)
                return result
            except KeyboardInterrupt:
                print(f"\n[{agent.name}] 已中断。输入提示词继续（输入 'quit' 退出）：")
                hint = input("> ").strip()
                if hint.lower() == "quit":
                    raise
                agent.add_message(Message(f"[用户提示] {hint}", "user"))
                messages.append({"role": "user", "content": f"[用户提示] {hint}"})

        return self._summarize_exhaustion(
            agent, user_input, system_prompt, messages, tool_calls)

    @staticmethod
    def _record_result(agent, user_input: str, result: str,
                       tool_calls: list[str]) -> None:
        agent.add_message(Message(user_input, "user"))
        if tool_calls:
            agent.add_message(Message(
                f"[本轮工具调用] {', '.join(tool_calls)}", "user"))
        agent.add_message(Message(result, "assistant"))

    @staticmethod
    def _record_tokens(agent, response, messages: list, step: int) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None and usage.input_tokens > 0:
            tokens = usage.input_tokens
            agent._token_anchor = tokens
            agent._last_output_tokens = usage.output_tokens
            source = "api"
        else:
            tokens = estimate_tokens(messages)
            source = "local"
        logger.info(
            "[%s] step %d context_tokens: %d (source=%s)",
            agent.name, step + 1, tokens, source,
            extra={"event": "agent_step", "agent": agent.name,
                   "step": step + 1, "tokens": tokens, "source": source},
        )

    @attributed("exhaustion_summary")
    def _summarize_exhaustion(self, agent, user_input: str, system_prompt: str,
                              messages: list, tool_calls: list[str]) -> str:
        tool_summary = "\n".join(f"- {item}" for item in tool_calls) or "(无工具调用)"
        messages.append({
            "role": "user",
            "content": (
                "你已达到最大步数限制。请根据以上对话，简要总结本轮任务的完成情况。\n\n"
                "用以下格式回复（不要调用工具）：\n"
                "已完成的步骤: [简述已完成的工具调用]\n"
                "未完成的任务: [简述尚未完成的部分]\n"
                "产出文件: [列出已生成的文件路径，没有则写'无']\n\n"
                f"本轮工具调用记录:\n{tool_summary}"
            ),
        })
        try:
            summary = agent.llm.chat(
                messages=messages, system=system_prompt, max_tokens=512).strip()
        except Exception as exc:
            summary = f"[WARN] Reached max steps without finishing. (总结失败: {exc})"
        self._record_result(agent, user_input, summary, tool_calls)
        return summary


class PlanExecuteRunner:
    """Create a bounded plan and execute each step with a ReAct runner."""

    def __init__(self, react_runner: ReactRunner | None = None):
        self.react_runner = react_runner or ReactRunner()

    def run(self, agent, user_input: str, system_prompt: str,
            verbose: bool = True, record_intermediate: bool = False) -> str:
        plan = self._plan(agent, user_input, system_prompt, verbose)
        if plan is None:
            logger.warning("[%s] 计划解析失败，回退 ReAct", agent.name)
            return self.react_runner.run(
                agent, user_input, system_prompt, verbose, record_intermediate)

        logger.info("[%s] Plan-Execute: %d 步: %s", agent.name, len(plan), plan)
        history = ""
        for index, step in enumerate(plan, 1):
            prompt = self._step_prompt(user_input, plan, step, history, index)
            result = self.react_runner.run(
                agent, prompt, system_prompt, verbose,
                require_finish=False, max_react_steps=PLAN_STEP_MAX_STEPS,
            )
            history += f"步骤 {index}: {step}\n结果: {result}\n\n"

        if agent.required_output_exts and not agent._output_satisfied(agent._written_paths):
            corrective = (
                "所有计划步骤已执行完，但还没有写出要求的产出文件"
                f"（{', '.join(agent.required_output_exts)}）。请立即用已有材料写出产出文件，"
                "确认已写入且非空后调用 finish。"
            )
            self.react_runner.run(
                agent, corrective, system_prompt, verbose,
                record_intermediate, require_finish=True,
            )

        summary = self._summarize(agent, system_prompt, history)
        agent.add_message(Message(user_input, "user"))
        agent.add_message(Message(history, "user"))
        agent.add_message(Message(summary, "assistant"))
        return summary

    @staticmethod
    @attributed("planning")
    def _plan(agent, user_input: str, system_prompt: str,
              verbose: bool) -> list[str] | None:
        prompt = (
            "你是一个顶级的AI规划专家。将用户的复杂问题分解成 3 到 5 个"
            "粗粒度、可执行、按逻辑顺序排列的步骤。将写产出文件合并到最后一步。\n\n"
            f"问题: {user_input}\n\n"
            "只输出一个 Python 字符串列表，例如：\n"
            '["步骤1", "步骤2", "步骤3"]'
        )
        try:
            raw = agent.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                system=system_prompt, max_tokens=16384,
            ).strip()
        except Exception as exc:
            if verbose:
                logger.warning("[%s] 规划调用失败: %s", agent.name, exc)
            return None
        plan = extract_python_list(raw)
        if not plan:
            return None
        return plan[:MAX_PLAN_STEPS]

    @staticmethod
    def _step_prompt(user_input: str, plan: list[str], step: str,
                     history: str, index: int) -> str:
        return (
            "你正在按计划分步执行任务。请专注完成当前步骤，需要时调用工具，"
            "完成后给出结果说明。\n\n"
            f"# 原始任务:\n{user_input}\n\n# 完整计划:\n{plan}\n\n"
            f"# 已完成步骤:\n{history or '（无）'}\n\n"
            f"# 当前步骤（第 {index}/{len(plan)} 步）:\n{step}"
        )

    @staticmethod
    @attributed("plan_summary")
    def _summarize(agent, system_prompt: str, history: str) -> str:
        try:
            return agent.llm.chat(
                messages=[{"role": "user", "content":
                           f"根据以下执行结果生成任务完成总结（纯文本）：\n\n{history}"}],
                system=system_prompt, max_tokens=1024,
            ).strip()
        except Exception:
            return "计划步骤已执行，但最终总结生成失败或不完整；不能据此确认任务完成，请核对产出。"
