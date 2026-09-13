"""LaTeX 模板工具
- SearchTemplateTool: 搜索模板
- ValidateTemplateTool: 验证模板
- DownloadTemplateTool: 下载模板
"""

import json as json_mod
from ..base import Tool


class SearchTemplateTool(Tool):
    """搜索 LaTeX 模板"""

    def __init__(self):
        super().__init__(
            name="search_template",
            description="搜索目标期刊的 LaTeX 模板。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "期刊名称"},
            },
            "required": ["journal"],
        }

    def execute(self, journal: str) -> str:
        return json_mod.dumps({
            "journal": journal,
            "status": "search_initiated",
            "message": f"请手动下载 {journal} 的 LaTeX 模板并放入 template_dir",
        })


class ValidateTemplateTool(Tool):
    """验证 LaTeX 模板"""

    def __init__(self):
        super().__init__(
            name="validate_template",
            description="验证 LaTeX 模板是否可用。输入模板目录路径。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "template_dir": {"type": "string", "description": "模板目录路径"},
            },
            "required": ["template_dir"],
        }

    def execute(self, template_dir: str) -> str:
        from pathlib import Path
        path = Path(template_dir)
        if not path.exists():
            return json_mod.dumps({"valid": False, "reason": f"目录不存在: {template_dir}"})
        cls_files = list(path.glob("*.cls"))
        sty_files = list(path.glob("*.sty"))
        tex_files = list(path.glob("*.tex"))
        if not cls_files and not sty_files:
            return json_mod.dumps({
                "valid": False, "reason": "模板目录中未找到 .cls 或 .sty 文件",
            })
        return json_mod.dumps({
            "valid": True,
            "cls_files": [f.name for f in cls_files],
            "sty_files": [f.name for f in sty_files],
            "tex_files": [f.name for f in tex_files],
        })


class DownloadTemplateTool(Tool):
    """下载 LaTeX 模板"""

    def __init__(self):
        super().__init__(
            name="download_template",
            description="下载目标期刊的 LaTeX 模板。输入 URL 和目标目录。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "模板下载 URL"},
                "target_dir": {"type": "string", "description": "目标目录"},
            },
            "required": ["url", "target_dir"],
        }

    def execute(self, url: str, target_dir: str) -> str:
        return json_mod.dumps({
            "status": "download_suggested",
            "message": f"请手动下载模板: {url} → {target_dir}",
        })