"""文献库更新/删除命令；不允许通过文件工具绕过结构化状态。"""

import json
from ...base import Tool


class UpdateReferenceTool(Tool):
    def __init__(self):
        '''初始化结构化文献更新工具。
        '''
        super().__init__(name="update_reference", description="更新已有文献的指定字段，保留其他字段。随后导出 bib。")
        self.context = None

    def get_parameters(self):
        '''定义工具接受的参数及必填字段。

        return: 工具参数的 JSON Schema 字典。
        '''
        return {"type": "object", "properties": {
            "cite_key": {"type": "string"},
            "changes": {"type": "object", "description": "Paper字段，如 authors/title/year/entry_type/publisher/bib_fields；不得改cite_key/source。"}},
            "required": ["cite_key", "changes"]}

    def execute(self, cite_key, changes):
        '''更新已有文献字段，保留原始来源信息，不执行联网存在性验证。

        paras:
            cite_key: 目标文献的引用键。
            changes: 需要更新的字段及新值字典。
        return: 成功时返回引用键 JSON，校验失败时返回 Error 文本。
        '''
        try:
            self.context.update_reference(cite_key, changes)
            return json.dumps({"status": "ok", "cite_key": cite_key})
        except (AttributeError, ValueError, TypeError) as exc:
            return f"Error: {exc}"


class RemoveReferenceTool(Tool):
    def __init__(self):
        '''初始化带引用保护的文献删除工具。
        '''
        super().__init__(name="remove_reference", description="删除无用或重复文献。代码检查正文及共享章节，仍被引用或无法完成检查时拒绝删除并返回位置；先修改引用再删除，最后导出bib。")
        self.context = None

    def get_parameters(self):
        '''定义工具接受的参数及必填字段。

        return: 工具参数的 JSON Schema 字典。
        '''
        return {"type": "object", "properties": {"cite_key": {"type": "string"}}, "required": ["cite_key"]}

    def execute(self, cite_key):
        '''通过上下文权限和引用保护检查删除指定文献。

        paras:
            cite_key: 目标文献的引用键。
        return: 成功时返回已删除引用键 JSON，失败时返回 Error 文本。
        '''
        try:
            self.context.remove_reference(cite_key)
            return json.dumps({"status": "ok", "removed": cite_key})
        except (AttributeError, ValueError) as exc:
            return f"Error: {exc}"
