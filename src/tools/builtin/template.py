"""LaTeX 模板工具
- ValidateTemplateTool: 验证模板目录可用性（存在可编译的主 .tex）

模板来源：用户提供的 template_dir；未提供时使用内置默认模板
（resources/templates/default/，由 main.py 在构建 PaperContext 时 fallback）。
搜索/下载模板功能已移除（假实现，无实际功能）。
"""

import json as json_mod
from ..base import Tool


class ValidateTemplateTool(Tool):
    """验证 LaTeX 模板目录"""

    def __init__(self):
        super().__init__(
            name="validate_template",
            description="验证 LaTeX 模板目录是否可用（存在 .tex 文件即可编译）。输入模板目录路径。"
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
        if not path.is_dir():
            return json_mod.dumps({"valid": False, "reason": f"不是目录: {template_dir}"})

        tex_files = list(path.glob("*.tex"))
        if not tex_files:
            return json_mod.dumps({
                "valid": False,
                "reason": "模板目录中未找到任何 .tex 文件（校验标准：存在可编译的主 .tex）",
            })
        cls_files = list(path.glob("*.cls"))
        sty_files = list(path.glob("*.sty"))
        return json_mod.dumps({
            "valid": True,
            "main_tex_candidates": [f.name for f in tex_files],
            "cls_files": [f.name for f in cls_files],
            "sty_files": [f.name for f in sty_files],
        })
