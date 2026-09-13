"""
运行范式（Run Mode）策略模块

将 react / plan_execute / reflection 三种运行范式从 Agent 基类中拆出，
每种范式一个类（策略模式）。Agent 持有一个 RunMode 实例，run() 时委托给它。

业务子类（Master / Literature / Writing / Citation / Review / Build）继续用
「继承」表达角色差异；运行范式用「组合」表达，二者正交，避免 6×3 的组合爆炸。

- ReactMode        ：Reasoning + Acting 交替循环（复用 Agent._run_react）
- PlanExecuteMode  ：Planner 先出计划，再逐步执行；每一步交给一个「能调工具」
                     的 ReAct 子会话（_run_react，require_finish=False）
- ReflectionMode   ：执行 → 反思 → 修正
"""

import ast as _ast
import logging
import re as _re
from abc import ABC, abstractmethod

from .message import Message

logger = logging.getLogger(__name__)

# 单个 plan 步骤内部的 ReAct 步数上限：防止某一步反复无效操作（见 PROJECT_LOG 主题 24）
PLAN_STEP_MAX_STEPS = 6

# 计划步骤数量硬上限：超过则截断，防止 LLM 把任务越切越细（实际 3~5 步通常足够）
MAX_PLAN_STEPS = 5


class RunMode(ABC):
    """运行范式基类。子类实现 run()，编排 agent 的 LLM 调用。"""

    @abstractmethod
    def run(self, agent, user_input: str, system_prompt: str,
            verbose: bool = True, record_intermediate: bool = False) -> str:
        ...


class ReactMode(RunMode):
    """ReAct：推理与行动交替，直到 finish 或步数耗尽。"""

    def run(self, agent, user_input, system_prompt, verbose=True, record_intermediate=False):
        return agent._run_react(user_input, system_prompt, verbose, record_intermediate)


class PlanExecuteMode(RunMode):
    """Plan-Execute：先规划，再逐步执行（每步带工具）。"""

    def run(self, agent, user_input, system_prompt, verbose=True, record_intermediate=False):
        plan = self._plan(agent, user_input, system_prompt, verbose)
        if plan is None:
            # 计划解析失败 → 回退 ReAct
            if verbose:
                logger.warning(
                    f"[{agent.name}] Plan-Execute: 计划解析失败，回退 ReAct",
                    extra={"event": "plan_execute", "agent": agent.name},
                )
            return agent._run_react(user_input, system_prompt, verbose, record_intermediate)

        if verbose:
            logger.info(
                f"[{agent.name}] Plan-Execute: 计划 {len(plan)} 步: {plan}",
                extra={"event": "plan_execute", "agent": agent.name, "plan": plan},
            )

        history = ""
        for i, step in enumerate(plan, 1):
            if verbose:
                logger.info(
                    f"[{agent.name}] Plan-Execute: 执行步骤 {i}/{len(plan)}: {step}",
                    extra={"event": "plan_execute_step", "agent": agent.name,
                           "step": i, "total": len(plan)},
                )

            step_prompt = self._step_prompt(user_input, plan, step, history, i, len(plan))
            step_result = agent._run_react(
                step_prompt, system_prompt, verbose,
                require_finish=False, max_react_steps=PLAN_STEP_MAX_STEPS,
            )
            history += f"步骤 {i}: {step}\n结果: {step_result}\n\n"

        summary = self._summarize(agent, system_prompt, history)
        agent.add_message(Message(user_input, "user"))
        agent.add_message(Message(history, "user"))
        agent.add_message(Message(summary, "assistant"))
        agent._save_to_memory(user_input, summary)
        return summary

    # ---- Planner：纯 LLM 出计划（无工具） ----

    def _plan(self, agent, user_input, system_prompt, verbose) -> list | None:
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
            plan_result = agent.llm.chat(
                messages=[{"role": "user", "content": plan_prompt}],
                system=system_prompt,
                max_tokens=16384,
            ).strip()
        except Exception as e:
            if verbose:
                logger.warning(f"[{agent.name}] Plan-Execute: 规划调用失败: {e}")
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
            plan = _ast.literal_eval(plan_str) # str->list
            if not isinstance(plan, list) or not plan:
                raise ValueError("Empty plan")
            if len(plan) > MAX_PLAN_STEPS: # 步数超过上限，则做截断
                if verbose:
                    logger.warning(
                        f"[{agent.name}] Plan-Execute: 计划 {len(plan)} 步超过上限"
                        f" {MAX_PLAN_STEPS}，截断为前 {MAX_PLAN_STEPS} 步"
                    )
                plan = plan[:MAX_PLAN_STEPS]
            return plan
        except Exception:
            if verbose:
                logger.warning(
                    f"[{agent.name}] Plan-Execute: 无法解析计划，回退 ReAct\n"
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

    def _summarize(self, agent, system_prompt, history) -> str:
        summary_prompt = (
            "你已完成所有计划步骤。请根据以下执行结果，生成任务完成总结"
            "（纯文本，不要调用工具）：\n\n"
            f"{history}"
        )
        try:
            return agent.llm.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                system=system_prompt,
                max_tokens=1024,
            ).strip()
        except Exception:
            return "Plan-Execute 完成。"


class ReflectionMode(RunMode):
    """Reflection：执行 → 反思 → 修正（最多 2 轮反思）。"""

    MAX_ROUNDS = 2

    def run(self, agent, user_input, system_prompt, verbose=True, record_intermediate=False):
        result = agent._run_react(user_input, system_prompt, verbose, record_intermediate)
        if verbose:
            logger.info(f"[{agent.name}] Reflection 第 1 轮执行完成",
                        extra={"event": "reflection", "agent": agent.name})

        for r in range(self.MAX_ROUNDS):
            reflection_prompt = (
                "你刚完成了一个任务。请回顾你的工作，进行自我评估。\n\n"
                "## 原始任务\n"
                f"{user_input}\n\n"
                "## 你的执行结果\n"
                f"{result}\n\n"
                "## 请回答以下问题\n"
                "1. 任务是否完全完成？有没有遗漏？\n"
                "2. 结果中是否有错误或不准确的地方？\n"
                "3. 如果有问题，需要如何修正？\n\n"
                "如果一切正确且完整，回复 'OK: 无需修正'。\n"
                "如果需要修正，回复 'FIX: <具体修正方案>'。"
            )
            try:
                reflection = agent.llm.chat(
                    messages=[{"role": "user", "content": reflection_prompt}],
                    system=system_prompt,
                    max_tokens=1024,
                ).strip()
            except Exception as e:
                logger.warning(f"[{agent.name}] Reflection 调用失败: {e}")
                break

            if verbose:
                logger.info(
                    f"[{agent.name}] Reflection 第 {r+1} 轮评估: {reflection[:200]}"
                )

            if reflection.startswith("OK:") or reflection.startswith("OK："):
                if verbose:
                    logger.info(f"[{agent.name}] Reflection: 无需修正，任务完成")
                break

            fix_prompt = (
                "根据以下反思结果，修正你之前的工作：\n\n"
                f"{reflection}\n\n"
                "请执行必要的修正操作。"
            )
            if verbose:
                logger.info(f"[{agent.name}] Reflection: 开始修正轮 {r+1}")
            result = agent._run_react(fix_prompt, system_prompt, verbose, record_intermediate)

        return result


_RUN_MODES: dict[str, type[RunMode]] = {
    "react": ReactMode,
    "plan_execute": PlanExecuteMode,
    "reflection": ReflectionMode,
}


def resolve_run_mode(run_mode) -> RunMode:
    """将 run_mode 配置解析为 RunMode 实例。

    - 已是 RunMode 实例 → 原样返回
    - 字符串（"react" / "plan_execute" / "reflection"）→ 查表实例化
    - 未知值 → 回退 ReactMode
    """
    if isinstance(run_mode, RunMode):
        return run_mode
    return _RUN_MODES.get(run_mode, ReactMode)()
