"""
Agent 基类 — 所有 Agent 的抽象父类

提供共享的工具调用循环（react / plan_execute 两种运行范式，直接实现在本类），
子类只需实现 run()，决定如何编排 LLM 调用。

特性：
  - 多轮对话：_history 自动纳入 messages 上下文（含工具调用摘要）
  - 记忆系统：MemoryManager 统一管理，_run_loop 自动记录
  - 上下文注入：AgentContextView 的字段自动注入到 system prompt
  - 工具调用：每个 Agent 拥有独立的 ToolRegistry
  - 上下文压缩：历史过长时自动压缩为结构化摘要，降低 LLM schema dropout 风险
"""

import ast as _ast
import json
import logging
import re as _re
from pathlib import Path
from abc import ABC, abstractmethod
from .message import Message
from .llm import LLM, get_tool_llm
from .context_compress import compress_history, compress_messages, reactive_compact, is_context_overflow_error
from .token_counter import estimate_tokens
from ..memory import AgentMemory

logger = logging.getLogger(__name__)

# 注册了 finish 的 agent，若连续「无工具调用」仍不退出的最大轮数，超过则强制结束（防止注入提示后死循环）
MAX_TEXT_ONLY = 2

# 判定「已产出文件」的工具集合：用于在临近步数上限仍未产出时注入收尾提示
_OUTPUT_TOOLS = {"write_file", "write_bib_file", "generate_bib_from_ref_library", "generate_bibtex"}

# 产出文件工具 → 输出路径参数名（用于 finish 前确定性校验「产出文件存在且非空」）
_OUTPUT_PATH_PARAM = {
    "write_file": "file_path",
    "write_bib_file": "output_path",
    "generate_bib_from_ref_library": "path",
}

# 产出闸门最多注入「还没写出产出文件」提示的次数，超过则放行（防止死循环）
MAX_OUTPUT_NUDGES = 3

# plan_execute 单个步骤内部的 ReAct 步数上限：防止某一步反复无效操作
PLAN_STEP_MAX_STEPS = 6

# 计划步骤数量硬上限：超过则截断，防止 LLM 把任务越切越细（实际 3~5 步通常足够）
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
        self.tool_registry = None  # 由外部注入 ToolRegistry 实例
        self.context = None        # AgentContextView，由外部注入

        # ---- 运行范式：react | plan_execute（未知值回退 react） ----
        self.run_mode = run_mode if run_mode in ("react", "plan_execute") else "react"

        # ---- 跨步骤产出路径收集：plan_execute 结束后闸门校验用（_run_loop 任务边界清空） ----
        self._written_paths: set[str] = set()

        # ---- 上下文压缩 ----
        self.context_window = context_window  # LLM 上下文窗口大小（tokens）
        self.history_keep_recent = history_keep_recent  # 压缩时保留的最近消息数

        # ---- 记忆系统：AgentMemory 由外部注入；召回/提取由 hooks 在任务边界自动完成 ----
        self.memory: AgentMemory | None = None
        self._recalled_memory: str = ""  # 入口 hook 召回的记忆文本，_run_loop 注入 system prompt

        # ---- 实例级 hooks：挂在循环上，不写进循环里（每个 Agent 实例独立注册表） ----
        # 事件：user_prompt_submit（任务入口）/ pre_tool_use / post_tool_use / stop（任务出口，整体一次）
        self._hooks: dict[str, list] = {
            event: [] for event in
            ("user_prompt_submit", "pre_tool_use", "post_tool_use", "stop")
        }
        # 默认 hooks：记忆召回（入口）、产出路径收集（工具后）、记忆提取（出口）
        self.register_hook("user_prompt_submit", self._hook_memory_recall)
        self.register_hook("post_tool_use", self._hook_collect_written_paths)
        self.register_hook("stop", self._hook_memory_extract)

        # ---- 确定性产出闸门：完成前必须写出的产出文件后缀（如 [".bib"] / [".tex"]），空列表表示不设闸门 ----
        self.required_output_exts: list[str] = []

        # ---- token anchor：缓存最近一次 API 真实 input_tokens，供下轮压缩决策用 ----
        # None 表示无可用 anchor（首次调用/任务边界/压缩后），fallback 到本地 estimate_tokens
        # 与 DeepSeek harness 的 TokenMeter.measure() anchor 模式对齐：
        #   - 主路径：用 response.usage.input_tokens（100% 准确）
        #   - 兜底：estimate_tokens(messages) 本地 BPE 估算
        self._token_anchor: int | None = None
        self._last_output_tokens: int = 0

    # ---- 子类必须实现 ----

    @abstractmethod
    def run(self, input_text: str) -> str:
        """运行 Agent，返回最终回答"""
        ...

    # ---- context → tools 同步（子类可按需重写） ----
    # synchronize 同步
    def _sync_context_to_tools(self):
        """将 PaperContext 中的相关数据同步到工具实例。

        子类重写此方法，在 context 注入后调用，将 reference_library
        等数据注入到 LookupPaperInfoTool 等依赖 context 的工具中。
        """
        pass

    # ---- 消息历史管理 ----

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    # ---- 工具执行 ----

    def _execute_tool(self, name: str, params: dict) -> str:
        """通过自己的工具注册表执行工具"""
        if self.tool_registry is None:
            return f"Error: Agent '{self.name}' 没有配置工具注册表"
        return self.tool_registry.execute(name, params)

    def _output_satisfied(self, written_paths: set[str]) -> bool:
        """确定性产出闸门：是否已写出要求的产出文件（存在且非空）。

        - 未声明 required_output_exts 的 agent 恒为 True（不设闸门）。
        - 已声明的 agent：任一写入路径后缀匹配且文件存在、大小 > 0 即视为达标。
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
        """获取 Anthropic 格式的工具列表"""
        if self.tool_registry is None:
            return []
        return self.tool_registry.to_anthropic_format()

    # ---- 辅助方法 ----

    def _build_context_info(self) -> str:
        """构建 PaperContext 的可读字段摘要，注入到 system prompt"""
        if self.context is None:
            return ""
        try:
            return self.context.get_readable_summary()
        except Exception:
            return ""

    # ---- Hooks：扩展点不侵入循环 ----

    def register_hook(self, event: str, callback):
        """注册 hook 回调到指定事件（实例级，子类可在 _setup 后追加自己的 hook）"""
        if event not in self._hooks:
            raise ValueError(f"未知 hook 事件: {event}（可用: {list(self._hooks)}）")
        self._hooks[event].append(callback)

    def _trigger_hooks(self, event: str, *args):
        """按注册顺序触发该事件的所有 hook，返回第一个非 None 结果（pre_tool_use 拦截 / stop 强制继续用）"""
        for callback in self._hooks[event]:
            result = callback(*args)
            if result is not None:
                return result
        return None

    # ---- 默认 hooks ----

    def _hook_memory_recall(self, user_input: str):
        """user_prompt_submit：任务入口召回相关记忆，存入 _recalled_memory 供 _run_loop 注入 system prompt"""
        self._recalled_memory = ""
        if self.memory is None:
            return None
        try:
            recalled = self.memory.recall(user_input, llm=get_tool_llm())
            if recalled:
                self._recalled_memory = recalled
                logger.info(f"[{self.name}] 召回相关记忆 {len(recalled)} 字符",
                            extra={"event": "memory_recall", "agent": self.name,
                                   "chars": len(recalled)})
        except Exception as e:
            logger.warning(f"[{self.name}] 记忆召回失败: {e}")
        return None

    def _hook_collect_written_paths(self, block, output: str):
        """post_tool_use：收集成功写出的产出文件路径（供产出闸门校验，plan_execute 跨步骤共享）"""
        path_param = _OUTPUT_PATH_PARAM.get(block.name)
        if path_param:
            p = block.input.get(path_param)
            ok = not output.startswith(("Error:", "警告:"))
            if block.name in ("write_bib_file", "generate_bib_from_ref_library"):
                ok = '"status": "ok"' in output
            if ok and isinstance(p, str) and p:
                try:
                    self._written_paths.add(str(Path(p).resolve()))
                except (OSError, ValueError):
                    pass
        return None

    def _hook_memory_extract(self, user_input: str, result: str):
        """stop：任务出口（每次 run() 整体一次）从对话中提取持久化知识入库，必要时触发整理"""
        if self.memory is None:
            return None
        try:
            lines = [f"{m.role}: {str(m.content)[:500]}" for m in self._history[-12:]]
            lines.append(f"user: {str(user_input)[:500]}")
            lines.append(f"assistant: {str(result)[:500]}")
            dialogue = "\n".join(lines)[:8000]
            stored = self.memory.extract(dialogue, llm=get_tool_llm())
            if stored:
                self.memory.consolidate(llm=get_tool_llm())
        except Exception as e:
            logger.warning(f"[{self.name}] 记忆提取失败: {e}")
        return None

    # ---- 上下文压缩：委托给 context_compress 模块 ----

    def _compress_history(self):
        """压缩 _history，使用渐进式阈值（基于 context_window 的百分比）"""
        self._history = compress_history(
            self._history,
            keep_recent=self.history_keep_recent,
            context_window=self.context_window,
            llm=self.llm,
            agent_name=self.name,
        )

    def _compress_messages(self, messages: list) -> list:
        """压缩 step 循环中的 messages。使用渐进式阈值。

        Token 估算优先用上次 API 真实 input_tokens 作 anchor（与 DeepSeek
        harness 对齐），fallback 到本地 estimate_tokens。
        压缩发生后重置 anchor，因为 messages 大幅缩减，旧 anchor 不再有效。
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
        # 压缩或路径指针替换发生后，旧 anchor 失效（内容已缩减，anchor 是替换前的真实值）
        # compress_messages 是 copy-on-write：无变化返回原对象；有变化必产生新 dict
        if len(new_messages) != len(messages) or any(a is not b for a, b in zip(new_messages, messages)):
            self._token_anchor = None
            self._last_output_tokens = 0
        return new_messages

    def _run_loop(self, user_input: str,
                  system_prompt: str | None = None,
                  verbose: bool = True,
                  record_intermediate: bool = False) -> str:
        """按 run_mode 分发：react / plan_execute。

        任务边界 hooks：入口触发记忆召回（user_prompt_submit），出口触发记忆提取（stop，
        每次 run() 整体触发一次，plan_execute 的子步骤不算出口）。
        """
        sys_prompt = system_prompt or self.system_prompt
        self._written_paths = set()

        # ---- 入口：记忆召回，拼进 system prompt（背景参考，非指令） ----
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

        # ---- 出口：记忆提取（仅一次） ----
        self._trigger_hooks("stop", user_input, result)
        return result

    # ---- Plan-Execute 范式：先规划，再逐步执行（每步带工具） ----

    def _run_plan_execute(self, user_input: str, system_prompt: str,
                          verbose: bool = True, record_intermediate: bool = False) -> str:
        plan = self._plan_steps(user_input, system_prompt, verbose)
        if plan is None:
            # 计划解析失败 → 回退 ReAct
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

        # ---- 产出闸门：所有步骤执行完后校验（_run_react 子步骤不校验，在此统一补上） ----
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

    # ---- Planner：纯 LLM 出计划（无工具） ----

    def _plan_steps(self, user_input: str, system_prompt: str, verbose) -> list | None:
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

        try:
            # 把 LLM 返回的自然语言+结构化数据混合文本，提取成 Python list，再校验长度
            plan_str = plan_result
            code_match = _re.search(r"```(?:python)?\s*(\[[\s\S]*?\])\s*```", plan_str)
            if code_match:
                plan_str = code_match.group(1)
            else:
                list_match = _re.search(r"\[[\s\S]*\]", plan_str)
                if list_match:
                    plan_str = list_match.group(0)
            plan = _ast.literal_eval(plan_str)  # str->list
            if not isinstance(plan, list) or not plan:
                raise ValueError("Empty plan")
            if len(plan) > MAX_PLAN_STEPS:  # 步数超过上限，则做截断
                if verbose:
                    logger.warning(
                        f"[{self.name}] Plan-Execute: 计划 {len(plan)} 步超过上限"
                        f" {MAX_PLAN_STEPS}，截断为前 {MAX_PLAN_STEPS} 步"
                    )
                plan = plan[:MAX_PLAN_STEPS]
            return plan
        except Exception:
            if verbose:
                logger.warning(
                    f"[{self.name}] Plan-Execute: 无法解析计划，回退 ReAct\n"
                    f"  原始输出: {plan_result[:200]}"
                )
            return None

    # ---- Executor：单步提示词 ----

    def _step_prompt(self, user_input, plan, step, history, i, total) -> str:
        return (
            "你正在按计划分步执行任务。请专注完成「当前步骤」，"
            "需要时调用工具（如解析 PDF、写入文件等），"
            "完成当前步骤后给出结果说明。\n\n"
            f"# 原始任务:\n{user_input}\n\n"
            f"# 完整计划:\n{plan}\n\n"
            f"# 已完成的步骤与结果:\n{history if history else '（无，这是第一步）'}\n\n"
            f"# 当前步骤（第 {i}/{total} 步）:\n{step}"
        )

    # ---- 总结 ----

    def _plan_summarize(self, system_prompt: str, history: str) -> str:
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
        """ReAct 范式：Reasoning + Acting 交替循环。

        require_finish=False 时作为「plan 子步骤执行器」使用：不强制 finish、
        不校验产出闸门，模型返回纯文本即视为本子步骤完成。
        max_react_steps 覆盖本子步骤的步数上限（默认沿用 self.max_steps）。

        核心流程：
          1. 构建 messages：历史消息 + 新用户输入
          2. 注入上下文信息到 system prompt
          3. 发送消息 + 工具定义给模型
          4. 如果模型返回 tool_use → 执行工具 → 追加结果 → 回到步骤 3
          5. 如果模型返回 text（无工具调用）→ 保存记忆，返回文本
        """
        # ---- 同步 context 到工具（如注入 reference_library） ----
        self._sync_context_to_tools()

        sys_prompt = system_prompt or self.system_prompt
        tools = self._get_tools_for_llm()

        # ---- 任务边界：上一轮 _token_anchor 已失效（本轮 messages 全新构建） ----
        # 见 __init__ 中 _token_anchor 的注释说明
        self._token_anchor = None
        self._last_output_tokens = 0

        # ---- 压缩 _history（如果过大） ----
        self._compress_history()

        # ---- 多轮对话：从 _history 构建 messages ----
        messages = []
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_input})

        # ---- 注入 Context 信息到 system prompt ----
        context_info = self._build_context_info()
        if context_info:
            sys_prompt = f"{sys_prompt}\n\n---\n\n{context_info}"

        # ---- ReAct 循环 ----
        tool_calls_made: list[str] = []

        # ---- 完成判定：注册了 finish 的 agent，必须以 finish 结束 ----
        has_finish = False
        if self.tool_registry is not None:
            try:
                has_finish = self.tool_registry.get_tool("finish") is not None
            except Exception:
                has_finish = False
        text_only_streak = 0  # 连续「无工具调用」的轮数，防止注入提示后死循环

        # 只有具备「产出文件」工具的 agent 才注入收尾提示（MasterAgent 只分派，不适用）
        has_output_tools = False
        if self.tool_registry is not None:
            try:
                has_output_tools = any(
                    self.tool_registry.get_tool(t) is not None for t in _OUTPUT_TOOLS
                )
            except Exception:
                has_output_tools = False
        wrote_output = False
        deadline_nudged = False
        output_nudges = 0                # 产出闸门已注入「还没写产出」提示的次数（防死循环）

        step_limit = max_react_steps or self.max_steps
        reactive_compacted = False  # 上下文超限补救只试一次，防止反复重试
        for step in range(step_limit):
            try:
                # ---- 收尾提示：临近步数上限仍未产出文件时，强制转向「写产出」 ----
                if (require_finish and has_output_tools and not wrote_output and not deadline_nudged
                        and step >= step_limit - 3):
                    deadline_nudged = True
                    messages.append({
                        "role": "user",
                        "content": (
                            "你已经执行了多步，但还没有写出任何产出文件。"
                            "不要继续收集信息或反复调用工具了，立即用现有材料写出任务要求的产出文件，"
                            "然后调用 finish。"
                        ),
                    })
                    if verbose:
                        logger.info(
                            f"[{self.name}] step {step+1} 临近步数上限且未产出文件，注入收尾提示",
                            extra={"event": "deadline_nudge", "agent": self.name,
                                   "step": step+1},
                        )

                # ---- 压缩 messages（如果过大） ----
                messages = self._compress_messages(messages)

                try:
                    response = self.llm.chat_with_tools(
                        messages=messages,
                        tools=tools,
                        system=sys_prompt,
                        max_tokens=self.max_tokens,
                    )
                except Exception as e:
                    # ---- reactive 补救：API 报上下文超限 → 紧急压缩后重试一次 ----
                    if not reactive_compacted and is_context_overflow_error(e):
                        reactive_compacted = True
                        messages = reactive_compact(messages, self.llm, self.name)
                        self._token_anchor = None
                        self._last_output_tokens = 0
                        step -= 1  # 重试本轮，不消耗步数
                        continue
                    raise

                # ---- 记录当前上下文的 token 占用 ----
                # 优先用 API 返回的真实 usage（100% 准确，与 DeepSeek harness 对齐），
                # fallback 到本地 estimate_tokens（BPE 估算，99.9% 准确）
                if response.usage is not None and response.usage.input_tokens > 0:
                    ctx_tokens = response.usage.input_tokens
                    # 缓存供下一轮压缩决策用（_compress_messages 会读这两个值）
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

                assistant_blocks = []
                tool_results = []
                finish_summary = None  # 本步是否请求 finish（None=未请求）

                for block in response.content:
                    if block.type == "thinking":
                        assistant_blocks.append(block)

                    elif block.type == "text":
                        assistant_blocks.append(block)
                        if verbose:
                            logger.info(f"[{self.name}] step {step+1} text: {block.text[:200]}")

                    elif block.type == "tool_use":
                        if block.name in _OUTPUT_TOOLS:
                            wrote_output = True
                        if verbose:
                            logger.info(f"[{self.name}] step {step+1} tool: {block.name}({block.input})",
                                extra={"event": "tool_call", "agent": self.name, "step": step+1,
                                       "tool": block.name, "params": block.input})

                        # ---- 检测 finish 工具：记下摘要，闸门校验通过后再真正结束 ----
                        if block.name == "finish":
                            finish_summary = block.input.get("summary", "任务完成。")
                            break  # 不再执行后续工具，统一在下方做闸门校验

                        output = self._execute_tool(block.name, block.input)

                        # PostToolUse hook：产出路径收集等副作用（注册表决定跑什么）
                        self._trigger_hooks("post_tool_use", block, output)

                        if verbose:
                            safe = (
                                output[:150]
                                .replace("\x07", "")
                            )
                            safe = "".join(
                                c for c in safe if c == "\n" or c == "\t" or c >= " "
                            )
                            safe = safe.strip() or "(empty/whitespace)"
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

                # ---- 处理 finish 请求：确定性产出闸门校验 ----
                if finish_summary is not None:
                    if (require_finish
                            and not self._output_satisfied(self._written_paths)
                            and output_nudges < MAX_OUTPUT_NUDGES):
                        output_nudges += 1
                        if verbose:
                            logger.info(
                                f"[{self.name}] step {step+1} 请求 finish 但未产出必需文件"
                                f"（需 {self.required_output_exts}），阻止结束并提示继续"
                                f"（第 {output_nudges}/{MAX_OUTPUT_NUDGES} 次）",
                                extra={"event": "output_gate", "agent": self.name, "step": step+1},
                            )
                        messages.append({
                            "role": "user",
                            "content": (
                                f"任务尚未完成：你还没有写出要求的产出文件"
                                f"（{', '.join(self.required_output_exts)}）。不要就此 finish，"
                                "请立即用已有材料写出产出文件，确认已写入且非空后再调用 finish。"
                            ),
                        })
                        continue
                    # 闸门通过（或达到最大放行次数）：正常结束
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
                    text_only_streak = 0  # 有工具调用，重置连续空转计数
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

                    # 注册了 finish 的 agent，若未调用 finish 就返回文本，视为「未完成」，
                    # 注入提示继续，而不是静默当成完成（修复：读够信息后提前结束、不写产出文件）
                    if require_finish and has_finish and text_only_streak < MAX_TEXT_ONLY:
                        text_only_streak += 1
                        nudge = (
                            "你还没有调用 finish，任务尚未完成。不要就此停止。"
                            "请继续执行剩余工作（尤其是写出任务要求的产出文件），"
                            "确认全部完成后调用 finish。"
                        )
                        messages.append({"role": "user", "content": nudge})
                        if verbose:
                            logger.info(
                                f"[{self.name}] step {step+1} 未调用 finish 即返回文本，"
                                f"注入提示继续（第 {text_only_streak}/{MAX_TEXT_ONLY} 次）",
                                extra={"event": "nudge", "agent": self.name,
                                       "step": step+1},
                            )
                        continue

                    # 连续无工具调用也要结束时，仍校验产出闸门：未产出必需文件再给一次机会
                    if (require_finish
                            and self.required_output_exts
                            and not self._output_satisfied(self._written_paths)
                            and output_nudges < MAX_OUTPUT_NUDGES):
                        output_nudges += 1
                        nudge = (
                            f"你还没有写出要求的产出文件（{', '.join(self.required_output_exts)}），"
                            "任务尚未完成。不要就此结束，请立即写出产出文件，"
                            "确认已写入且非空后再调用 finish。"
                        )
                        messages.append({"role": "user", "content": nudge})
                        if verbose:
                            logger.info(
                                f"[{self.name}] step {step+1} 未产出必需文件即返回文本，"
                                f"注入产出提示（第 {output_nudges}/{MAX_OUTPUT_NUDGES} 次）",
                                extra={"event": "output_gate", "agent": self.name, "step": step+1},
                            )
                        continue

                    if verbose:
                        if require_finish:
                            logger.warning(
                                f"[{self.name}] step {step+1} 连续 {text_only_streak} 轮无工具调用，"
                                f"强制结束（result: {result[:120] if result != '(no text response)' else '空'}）",
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

        # ---- 达到 max_steps：让 LLM 总结本轮已完成的工作 ----
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

    def __str__(self) -> str:
        return f"Agent(name={self.name}, llm={self.llm})"

    def __repr__(self) -> str:
        return self.__str__()