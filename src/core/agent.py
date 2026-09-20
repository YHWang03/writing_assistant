"""Agent 基类 — 提供 react / plan_execute 两种运行范式与共享基础设施。

机制层职责：实例级 hooks 注册表（挂点在循环上，实现见 hooks.py）、
确定性产出闸门、上下文压缩委托、记忆注入点。子类只需实现 run()。
"""

import logging
from pathlib import Path
from abc import ABC, abstractmethod

from .message import Message
from .llm import LLM
from .context_compress import compress_history, compress_messages, reactive_compact, is_context_overflow_error
from ..hooks import HookRegistry
from .token_counter import estimate_tokens
from .utils import extract_python_list, strip_control_chars
from ..memory import AgentMemory

logger = logging.getLogger(__name__)

# plan_execute 单个步骤内部的 ReAct 步数上限
PLAN_STEP_MAX_STEPS = 6

# 计划步骤数量硬上限，超过截断
MAX_PLAN_STEPS = 5


class Agent(ABC):
    """所有 Agent 的抽象基类"""

    def __init__(self, name: str, llm: LLM,
                 system_prompt: str | None = None,
                 max_steps: int = 8, max_tokens: int = 4096,
                 context_window: int = 128000,
                 history_keep_recent: int = 6,
                 run_mode: str = "react"):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self._history: list[Message] = []
        self.tool_registry = None
        self.context = None

        # 运行范式：react | plan_execute（未知值回退 react）
        self.run_mode = run_mode if run_mode in ("react", "plan_execute") else "react"

        # 跨步骤产出路径收集：plan_execute 结束后闸门校验用（任务边界清空）
        self._written_paths: set[str] = set()

        # 上下文压缩
        self.context_window = context_window
        self.history_keep_recent = history_keep_recent

        # 记忆系统：AgentMemory 由外部注入，hooks 在任务边界自动召回/提取
        self.memory: AgentMemory | None = None
        self._recalled_memory: str = ""

        # hooks：HookRegistry 实例，具体 hook 由子类 _setup_hooks() 注册
        self.hooks = HookRegistry()

        # 确定性产出闸门：finish 前必须写出的产出文件后缀，空列表不设闸门
        self.required_output_exts: list[str] = []

        # token anchor：缓存最近一次 API 真实 input_tokens，供下轮压缩决策
        self._token_anchor: int | None = None
        self._last_output_tokens: int = 0

    @abstractmethod
    def run(self, input_text: str) -> str:
        """运行 Agent。

        paras:
            input_text: 任务文本
        return: 最终回答
        """
        ...

    def _sync_context_to_tools(self):
        """把 PaperContext 数据同步到依赖 context 的工具实例。

        paras: 无
        return: 无
        """
        pass

    def add_message(self, message: Message):
        """追加一条消息到 _history。

        paras:
            message: Message 对象
        return: 无
        """
        self._history.append(message)

    def clear_history(self):
        """清空 _history。

        paras: 无
        return: 无
        """
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取 _history 副本。

        paras: 无
        return: Message 对象列表
        """
        return self._history.copy()

    def _execute_tool(self, name: str, params: dict) -> str:
        """通过自己的工具注册表执行工具。

        paras:
            name: 工具名
            params: 工具参数
        return: 工具输出文本
        """
        if self.tool_registry is None:
            return f"Error: Agent '{self.name}' 没有配置工具注册表"
        return self.tool_registry.execute(name, params)

    def _output_satisfied(self, written_paths: set[str]) -> bool:
        """产出闸门：是否已写出要求的产出文件（后缀匹配且存在非空）。

        paras:
            written_paths: 本任务成功写入的路径集合
        return: 未设闸门恒 True；已设闸门时达标返回 True
        """
        if not self.required_output_exts:
            return True
        exts = tuple(self.required_output_exts)
        for p in written_paths:
            if not p.endswith(exts):
                continue
            try:
                if Path(p).exists() and Path(p).stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def _get_tools_for_llm(self) -> list[dict]:
        """获取 Anthropic 格式的工具列表。

        paras: 无
        return: 工具定义列表
        """
        if self.tool_registry is None:
            return []
        return self.tool_registry.to_anthropic_format()

    def _build_context_info(self) -> str:
        """构建 PaperContext 可读摘要，注入 system prompt。

        paras: 无
        return: 摘要文本；无 context 或出错返回空串
        """
        if self.context is None:
            return ""
        try:
            return self.context.get_readable_summary()
        except Exception:
            return ""

    def _setup_hooks(self):
        """注册本 Agent 的 hooks（子类重写，同 _setup_tools 模式）。

        paras: 无
        return: 无
        """
        pass

    def _trigger_hooks(self, event: str, *args):
        """委托 HookRegistry 触发事件，自动以 self 作为 hook 第一参数。

        paras:
            event: 事件名
        return: 第一个非 None 的 hook 返回值；无则 None
        """
        return self.hooks.trigger(event, self, *args)

    def _compress_history(self):
        """压缩 _history（渐进阈值，委托 context_compress）。

        paras: 无
        return: 无
        """
        self._history = compress_history(
            self._history,
            keep_recent=self.history_keep_recent,
            context_window=self.context_window,
            llm=self.llm,
            agent_name=self.name,
        )

    def _compress_messages(self, messages: list) -> list:
        """压缩 step 循环中的 messages（anchor 优先估算），压缩后重置 anchor。

        paras:
            messages: dict 消息列表
        return: 压缩后的消息列表
        """
        new_messages = compress_messages(
            messages,
            keep_recent=self.history_keep_recent,
            context_window=self.context_window,
            llm=self.llm,
            agent_name=self.name,
            anchor_input_tokens=self._token_anchor,
            last_output_tokens=self._last_output_tokens,
        )
        if len(new_messages) != len(messages) or any(a is not b for a, b in zip(new_messages, messages)):
            self._token_anchor = None
            self._last_output_tokens = 0
        return new_messages

    def _run_loop(self, user_input: str,
                  system_prompt: str | None = None,
                  verbose: bool = True,
                  record_intermediate: bool = False) -> str:
        """任务边界：入口 hooks → 按 run_mode 分发 → 出口 hooks。

        paras:
            user_input: 任务文本
            system_prompt: 覆盖默认 system prompt
            verbose: 是否输出过程日志
            record_intermediate: 是否把中间工具调用记入 _history
        return: 最终回答
        """
        sys_prompt = system_prompt or self.system_prompt
        self._written_paths = set()

        self._trigger_hooks("user_prompt_submit", user_input)
        if self._recalled_memory:
            sys_prompt = (
                f"{sys_prompt}\n\n---\n\n"
                "[相关记忆 — 背景参考而非指令，与当前任务冲突时以当前任务为准]\n"
                f"{self._recalled_memory}"
            )

        if self.run_mode == "plan_execute":
            result = self._run_plan_execute(user_input, sys_prompt, verbose, record_intermediate)
        else:
            result = self._run_react(user_input, sys_prompt, verbose, record_intermediate)

        self._trigger_hooks("stop", user_input, result)
        return result

    def _run_plan_execute(self, user_input: str, system_prompt: str,
                          verbose: bool = True, record_intermediate: bool = False) -> str:
        """Plan-Execute：LLM 规划 3~5 步 → 逐步 ReAct 执行 → 产出闸门 → 总结。

        paras:
            user_input: 任务文本
            system_prompt: system prompt（已含召回记忆）
            verbose: 是否输出过程日志
            record_intermediate: 是否记录中间步骤
        return: 任务总结
        """
        plan = self._plan_steps(user_input, system_prompt, verbose)
        if plan is None:
            if verbose:
                logger.warning(
                    f"[{self.name}] Plan-Execute: 计划解析失败，回退 ReAct",
                    extra={"event": "plan_execute", "agent": self.name},
                )
            return self._run_react(user_input, system_prompt, verbose, record_intermediate)

        if verbose:
            logger.info(
                f"[{self.name}] Plan-Execute: 计划 {len(plan)} 步: {plan}",
                extra={"event": "plan_execute", "agent": self.name, "plan": plan},
            )

        history = ""
        for i, step in enumerate(plan, 1):
            if verbose:
                logger.info(
                    f"[{self.name}] Plan-Execute: 执行步骤 {i}/{len(plan)}: {step}",
                    extra={"event": "plan_execute_step", "agent": self.name,
                           "step": i, "total": len(plan)},
                )
            step_prompt = self._step_prompt(user_input, plan, step, history, i, len(plan))
            step_result = self._run_react(
                step_prompt, system_prompt, verbose,
                require_finish=False, max_react_steps=PLAN_STEP_MAX_STEPS,
            )
            history += f"步骤 {i}: {step}\n结果: {step_result}\n\n"

        # 产出闸门：所有步骤完成后统一校验，未达标追加修正轮
        if self.required_output_exts and not self._output_satisfied(self._written_paths):
            if verbose:
                logger.info(
                    f"[{self.name}] Plan-Execute 步骤全部完成但未产出必需文件"
                    f"（需 {self.required_output_exts}），追加修正轮",
                    extra={"event": "output_gate", "agent": self.name},
                )
            corrective = (
                f"所有计划步骤已执行完，但还没有写出要求的产出文件"
                f"（{', '.join(self.required_output_exts)}）。请立即用已有材料写出产出文件，"
                "确认已写入且非空后调用 finish。"
            )
            self._run_react(corrective, system_prompt, verbose,
                            record_intermediate, require_finish=True)

        summary = self._plan_summarize(system_prompt, history)
        self.add_message(Message(user_input, "user"))
        self.add_message(Message(history, "user"))
        self.add_message(Message(summary, "assistant"))
        return summary

    def _plan_steps(self, user_input: str, system_prompt: str, verbose) -> list | None:
        """Planner：纯 LLM 生成 3~5 步计划并解析校验。

        paras:
            user_input: 任务文本
            system_prompt: system prompt
            verbose: 是否输出日志
        return: 步骤字符串列表；规划失败或解析失败返回 None
        """
        plan_prompt = (
            "你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成"
            "3 到 5 个粗粒度的行动计划步骤。\n\n"
            "要求：\n"
            "1. 步骤数量严格控制在 3~5 个，宁可合并同类操作，不要拆得过细。\n"
            "2. 每个步骤是一个独立的、可执行的子任务，按逻辑顺序排列。\n"
            "3. 「写产出文件」等最终动作合并到最后一步，不要为不同文件名重复拆步骤。\n"
            "4. 最后一步通常是汇总/报告结果。\n\n"
            "示例（文献处理场景）：\n"
            '["解析相关 PDF 并入库", '
            '"检查失败条目、补全缺失元数据", '
            '"生成 BibTeX 写入产出文件并汇总结果"]\n\n'
            "问题: {question}\n\n"
            "请严格按照以下格式输出你的计划（一个 Python 列表）:\n"
            "```python\n"
            '["步骤1", "步骤2", "步骤3"]\n'
            "```"
        ).format(question=user_input)

        try:
            plan_result = self.llm.chat(
                messages=[{"role": "user", "content": plan_prompt}],
                system=system_prompt,
                max_tokens=16384,
            ).strip()
        except Exception as e:
            if verbose:
                logger.warning(f"[{self.name}] Plan-Execute: 规划调用失败: {e}")
            return None

        plan = extract_python_list(plan_result)
        if not plan:
            if verbose:
                logger.warning(
                    f"[{self.name}] Plan-Execute: 无法解析计划，回退 ReAct\n"
                    f"  原始输出: {plan_result[:200]}"
                )
            return None
        if len(plan) > MAX_PLAN_STEPS:
            if verbose:
                logger.warning(
                    f"[{self.name}] Plan-Execute: 计划 {len(plan)} 步超过上限"
                    f" {MAX_PLAN_STEPS}，截断为前 {MAX_PLAN_STEPS} 步"
                )
            plan = plan[:MAX_PLAN_STEPS]
        return plan

    def _step_prompt(self, user_input, plan, step, history, i, total) -> str:
        """构建单步执行提示词（含原始任务、完整计划、已完成历史）。

        paras:
            user_input: 原始任务文本
            plan: 完整计划列表
            step: 当前步骤描述
            history: 已完成步骤与结果
            i: 当前步骤序号（1 起）
            total: 总步数
        return: 提示词文本
        """
        return (
            "你正在按计划分步执行任务。请专注完成「当前步骤」，"
            "需要时调用工具（如解析 PDF、写入文件等），"
            "完成当前步骤后给出结果说明。\n\n"
            f"# 原始任务:\n{user_input}\n\n"
            f"# 完整计划:\n{plan}\n\n"
            f"# 已完成的步骤与结果:\n{history if history else '（无，这是第一步）'}\n\n"
            f"# 当前步骤（第 {i}/{total} 步）:\n{step}"
        )

    def _plan_summarize(self, system_prompt: str, history: str) -> str:
        """对所有步骤结果生成任务总结。

        paras:
            system_prompt: system prompt
            history: 各步骤执行记录
        return: 总结文本；LLM 失败返回固定文案
        """
        summary_prompt = (
            "你已完成所有计划步骤。请根据以下执行结果，生成任务完成总结"
            "（纯文本，不要调用工具）：\n\n"
            f"{history}"
        )
        try:
            return self.llm.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                system=system_prompt,
                max_tokens=1024,
            ).strip()
        except Exception:
            return "Plan-Execute 完成。"

    def _run_react(self, user_input: str,
                   system_prompt: str | None = None,
                   verbose: bool = True,
                   record_intermediate: bool = False,
                   require_finish: bool = True,
                   max_react_steps: int | None = None) -> str:
        """ReAct 循环：构建 messages → LLM → 执行工具 → 回填结果，直至结束条件。

        require_finish=False 时作为 plan 子步骤执行器（不强制 finish、不校验闸门）。

        paras:
            user_input: 任务或子步骤文本
            system_prompt: system prompt
            verbose: 是否输出过程日志
            record_intermediate: 是否把中间工具调用记入 _history
            require_finish: 是否要求以 finish 结束并校验产出闸门
            max_react_steps: 本轮步数上限（默认 self.max_steps）
        return: finish 摘要、文本回答或步数耗尽总结
        """
        self._sync_context_to_tools()
        sys_prompt = system_prompt or self.system_prompt
        tools = self._get_tools_for_llm()

        self._token_anchor = None
        self._last_output_tokens = 0
        self._compress_history()

        messages = [{"role": msg.role, "content": msg.content} for msg in self._history]
        messages.append({"role": "user", "content": user_input})

        context_info = self._build_context_info()
        if context_info:
            sys_prompt = f"{sys_prompt}\n\n---\n\n{context_info}"

        tool_calls_made: list[str] = []
        reactive_compacted = False

        step_limit = max_react_steps or self.max_steps
        self.hooks.reset_all()
        for step in range(step_limit):
            try:
                nudge = self._trigger_hooks(
                    "pre_step", step, step_limit, require_finish)
                if nudge:
                    messages.append({"role": "user", "content": nudge})

                messages = self._compress_messages(messages)

                try:
                    response = self.llm.chat_with_tools(
                        messages=messages,
                        tools=tools,
                        system=sys_prompt,
                        max_tokens=self.max_tokens,
                    )
                except Exception as e:
                    # reactive 补救：上下文超限 → 紧急压缩后重试一次（不消耗步数）
                    if not reactive_compacted and is_context_overflow_error(e):
                        reactive_compacted = True
                        messages = reactive_compact(messages, self.llm, self.name)
                        self._token_anchor = None
                        self._last_output_tokens = 0
                        step -= 1
                        continue
                    raise

                self._record_context_tokens(response, messages, step, verbose)

                assistant_blocks = []
                tool_results = []
                finish_summary = None

                for block in response.content:
                    if block.type == "thinking":
                        assistant_blocks.append(block)
                    elif block.type == "text":
                        assistant_blocks.append(block)
                        if verbose:
                            logger.info(f"[{self.name}] step {step+1} text: {block.text[:200]}")
                    elif block.type == "tool_use":
                        if verbose:
                            logger.info(f"[{self.name}] step {step+1} tool: {block.name}({block.input})",
                                extra={"event": "tool_call", "agent": self.name, "step": step+1,
                                       "tool": block.name, "params": block.input})

                        if block.name == "finish":
                            finish_summary = block.input.get("summary", "任务完成。")
                            break

                        output = self._execute_tool(block.name, block.input)
                        self._trigger_hooks("post_tool_use", block, output)

                        if verbose:
                            safe = strip_control_chars(output)
                            logger.info(f"[{self.name}] step {step+1} result: {safe}",
                                extra={"event": "tool_result", "agent": self.name, "step": step+1,
                                       "tool": block.name, "is_error": output.startswith("Error:"),
                                       "result_size": len(output)})

                        tool_calls_made.append(f"{block.name}({str(block.input)[:100]})")
                        assistant_blocks.append(block)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })

                if assistant_blocks:
                    messages.append({"role": "assistant", "content": assistant_blocks})

                # finish 请求：pre_finish hooks 决定放行或阻止
                if finish_summary is not None:
                    nudge = self._trigger_hooks("pre_finish", require_finish)
                    if nudge:
                        messages.append({"role": "user", "content": nudge})
                        continue
                    if verbose:
                        logger.info(f"[{self.name}] step {step+1} finish: {finish_summary[:200]}",
                            extra={"event": "finish", "agent": self.name, "step": step+1,
                                   "total_steps": step+1})
                    self.add_message(Message(user_input, "user"))
                    if tool_calls_made:
                        self.add_message(Message(
                            f"[本轮工具调用] {', '.join(tool_calls_made)}", "user",
                        ))
                    self.add_message(Message(finish_summary, "assistant"))
                    return finish_summary

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                    if record_intermediate:
                        step_tools = [
                            f"{b.name}({str(b.input)[:100]})"
                            for b in response.content if b.type == "tool_use"
                        ]
                        self.add_message(Message(
                            f"[Step {step+1} tools] {', '.join(step_tools)}", "assistant",
                        ))
                        for tr in tool_results:
                            self.add_message(Message(
                                f"[Step {step+1} result] {tr['content'][:300]}", "user",
                            ))
                else:
                    texts = [b.text for b in response.content if b.type == "text"]
                    result = "\n".join(texts) if texts else "(no text response)"

                    # text_only hooks 按注册顺序决定阻止退出或放行
                    nudge = self._trigger_hooks(
                        "text_only", result, require_finish)
                    if nudge:
                        messages.append({"role": "user", "content": nudge})
                        continue

                    if verbose:
                        if require_finish:
                            logger.warning(
                                f"[{self.name}] step {step+1} 无工具调用，强制结束"
                                f"（result: {result[:120] if result != '(no text response)' else '空'}）",
                                extra={"event": "force_stop", "agent": self.name, "step": step+1},
                            )
                        else:
                            logger.info(
                                f"[{self.name}] step {step+1} 子步骤完成（返回文本）",
                                extra={"event": "substep_done", "agent": self.name, "step": step+1},
                            )

                    self.add_message(Message(user_input, "user"))
                    if tool_calls_made:
                        self.add_message(Message(
                            f"[本轮工具调用] {', '.join(tool_calls_made)}", "system",
                        ))
                    self.add_message(Message(result, "assistant"))
                    return result

            except KeyboardInterrupt:
                print(f"\n[{self.name}] 已中断。输入提示词继续（输入 'quit' 退出）：")
                hint = input("> ").strip()
                if hint.lower() == "quit":
                    raise
                self._history.append(Message(f"[用户提示] {hint}", "user"))
                messages.append({"role": "user", "content": f"[用户提示] {hint}"})
                if verbose:
                    logger.info(f"[{self.name}] 用户注入提示: {hint[:200]}",
                                extra={"event": "interrupt", "agent": self.name})
                continue

        # 步数耗尽：让 LLM 总结本轮完成情况
        tool_summary = "\n".join(f"- {t}" for t in tool_calls_made) if tool_calls_made else "(无工具调用)"
        summary_prompt = (
            "你已达到最大步数限制。请根据以上对话，简要总结本轮任务的完成情况。\n\n"
            "用以下格式回复（不要调用工具）：\n"
            "已完成的步骤: [简述已完成的工具调用]\n"
            "未完成的任务: [简述尚未完成的部分]\n"
            "产出文件: [列出已生成的文件路径，没有则写'无']\n\n"
            f"本轮工具调用记录:\n{tool_summary}"
        )
        messages.append({"role": "user", "content": summary_prompt})
        try:
            response = self.llm.chat(
                messages=messages,
                system=sys_prompt,
                max_tokens=512,
            )
            summary = response.strip()
            logger.info(f"[{self.name}] max_steps summary: {summary[:200]}",
                    extra={"event": "max_steps", "agent": self.name, "step": self.max_steps})
        except Exception as e:
            summary = f"[WARN] Reached max steps without finishing. (总结失败: {e})"

        self.add_message(Message(user_input, "user"))
        if tool_calls_made:
            self.add_message(Message(
                f"[本轮工具调用] {', '.join(tool_calls_made)}", "system",
            ))
        self.add_message(Message(summary, "assistant"))
        return summary

    def _has_tool(self, name: str) -> bool:
        """判断工具注册表中是否存在指定工具。

        paras:
            name: 工具名
        return: 存在返回 True
        """
        if self.tool_registry is None:
            return False
        try:
            return self.tool_registry.get_tool(name) is not None
        except Exception:
            return False

    def _record_context_tokens(self, response, messages: list, step: int, verbose: bool):
        """记录本轮上下文 token 占用，并缓存 API 真实 usage 作下轮压缩 anchor。

        paras:
            response: chat_with_tools 的返回对象
            messages: 当前消息列表（无 usage 时本地估算用）
            step: 当前步数（0 起，日志用）
            verbose: 是否输出日志
        return: 无
        """
        if response.usage is not None and response.usage.input_tokens > 0:
            ctx_tokens = response.usage.input_tokens
            self._token_anchor = ctx_tokens
            self._last_output_tokens = response.usage.output_tokens
            source = "api"
        else:
            ctx_tokens = estimate_tokens(messages)
            source = "local"
        logger.info(
            f"[{self.name}] step {step+1} context_tokens: {ctx_tokens} (source={source})",
            extra={"event": "agent_step", "agent": self.name, "step": step+1,
                   "tokens": ctx_tokens, "source": source},
        )

    def __str__(self) -> str:
        return f"Agent(name={self.name}, llm={self.llm})"

    def __repr__(self) -> str:
        return self.__str__()
