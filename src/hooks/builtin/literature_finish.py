"""文献专用收尾阶段：预留交付时间，导出同步成功后立即结束。"""
import json
import logging

from ..base import Hook, HookFinish

logger = logging.getLogger(__name__)


class LiteratureFinishHook(Hook):
    """临近预算上限限制搜集工作，导出后核对交付状态。"""

    ALLOWED = frozenset({'read_context', 'update_reference', 'remove_reference',
                         'write_library', 'generate_bib_from_ref_library', 'finish'})

    def __init__(self):
        '''初始化当前 ReAct 轮次的收尾状态。'''
        self.reset()

    def reset(self):
        '''清空当前 ReAct 轮次的状态，不改变文献库或导出记录。'''
        self.closing = False

    def on_pre_step(self, agent, step, step_limit, require_finish):
        '''仅在临近步数上限时切换到收尾阶段。

        paras:
            agent: 文献 Agent。
            step: 当前轮索引，从 0 开始。
            step_limit: 本轮执行步数上限。
            require_finish: 计划内部子步骤为 False，不在此强制收尾。
        return: 收尾提示；未触发时返回 None。
        '''
        agent._finish_confirmed = False
        if not require_finish or agent.context is None or self.closing:
            return None
        reason = None
        if step >= max(0, step_limit - agent.finish_reserve_steps):
            reason = '已进入预留收尾轮次'
        if reason is None:
            return None
        self.closing = True
        logger.info('[%s] 文献收尾：%s', agent.name, reason,
                    extra={'event': 'literature_closing', 'agent': agent.name,
                           'step': step + 1, 'reason': reason})
        return (f'{reason}。停止调查，使用已有材料完成必要修订、write_library 和 '
                'generate_bib_from_ref_library，然后汇报缺失字段、未处理材料或检索缺口。'
                '收尾阶段禁止搜索、解析和泛读文件；预算耗尽不代表检索任务已充分完成。')

    def on_pre_tool_use(self, agent, block):
        '''在收尾阶段拦截继续搜集信息的工具调用。

        paras:
            agent: 文献 Agent。
            block: 模型请求的工具调用块。
        return: 被禁止时返回 Error 文本，否则返回 None。
        '''
        if self.closing and block.name not in self.ALLOWED:
            return ('Error: 已进入文献收尾阶段，禁止继续搜索、解析或读取文件。'
                    '请保存当前库、导出 .bib，并汇报未完成项。')
        return None

    def on_post_tool_use(self, agent, block, output):
        '''导出成功且磁盘产物同步后返回结束信号，失败时保留修复机会。

        paras:
            agent: 文献 Agent。
            block: 已执行的工具调用。
            output: 工具返回的文本。
        return: 程序生成的 HookFinish 交付摘要，或不结束时的 None。
        '''
        if block.name != 'generate_bib_from_ref_library':
            return None
        try:
            result = json.loads(output)
        except (ValueError, TypeError):
            return None
        if not isinstance(result, dict) or result.get('status') != 'ok' or agent.completion_issues():
            return None
        agent._finish_confirmed = True
        return HookFinish(f"文献交付已完成：已保存文献库并同步导出 {result['path']}，"
                          f"共 {result['total']} 条。程序已结束本次派发；产物同步不代表任务覆盖或语义质量通过。")

    def on_pre_finish(self, agent, require_finish):
        '''按真实磁盘状态阻止尚未保存或同步导出的完成声明。

        paras:
            agent: 文献 Agent。
            require_finish: 是否要求当前调用完成完整任务。
        return: 未完成项提示；条件满足或计划子步骤时返回 None。
        '''
        if not require_finish:
            return None
        issues = agent.completion_issues()
        if issues:
            self.closing = True
            return '任务尚未完成：' + '；'.join(issues) + '。请保存并重新导出，失败时如实汇报。'
        agent._finish_confirmed = True
        return None

    def on_text_only(self, agent, result_text, require_finish):
        '''对纯文本出口执行与 finish 相同的真实产物检查。

        paras:
            agent: 文献 Agent。
            result_text: 模型文本答复。
            require_finish: 是否要求完成完整任务。
        return: 未完成项提示或 None。
        '''
        return self.on_pre_finish(agent, require_finish)
