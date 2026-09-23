"""文献库更新/删除命令；不允许通过文件工具绕过结构化状态。"""

import json
from ...base import Tool


class UpdateReferenceTool(Tool):
    def __init__(self):
        super().__init__(name="update_reference", description="更新已有文献的指定字段，保留其他字段。随后导出 bib。")
        self.context = None

    def get_parameters(self):
        return {"type": "object", "properties": {
            "cite_key": {"type": "string"},
            "changes": {"type": "object", "description": "Paper字段，如 authors/title/year/entry_type/publisher/bib_fields；不得改cite_key/source。"}},
            "required": ["cite_key", "changes"]}

    def execute(self, cite_key, changes):
        try:
            if "title" in changes:
                from ....domain.reference_provenance import title_key
                from copy import deepcopy
                current = next((p for p in self.context.get_references() if p.cite_key == cite_key), None)
                if current is None:
                    raise ValueError("文献不存在")
                if title_key(changes['title']) != title_key(current.title):
                    gate = getattr(self, 'provenance', None)
                    if gate is None:
                        raise ValueError("来源门禁未初始化，禁止更改标题")
                    candidate = deepcopy(current)
                    candidate.title = changes['title']
                    gate.admit(candidate, self.verifier)
            self.context.update_reference(cite_key, changes)
            return json.dumps({"status": "ok", "cite_key": cite_key})
        except (AttributeError, ValueError, TypeError) as exc:
            return f"Error: {exc}"


class RemoveReferenceTool(Tool):
    def __init__(self):
        super().__init__(name="remove_reference", description="删除确认无用或重复的文献。先确认正文不再引用该键，然后重新导出 bib。")
        self.context = None

    def get_parameters(self):
        return {"type": "object", "properties": {"cite_key": {"type": "string"}}, "required": ["cite_key"]}

    def execute(self, cite_key):
        try:
            self.context.remove_reference(cite_key)
            return json.dumps({"status": "ok", "removed": cite_key})
        except (AttributeError, ValueError) as exc:
            return f"Error: {exc}"
