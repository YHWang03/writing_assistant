"""产出路径收集 hook — 收集成功写出的产出文件路径。"""

from pathlib import Path

from ..base import Hook
from ...tools.builtin._safe_path import safe_resolve

# 产出文件工具 → 输出路径参数名
OUTPUT_PATH_PARAM = {
    "write_file": "file_path",
    "generate_bib_from_ref_library": "path",
}


class CollectWrittenPathsHook(Hook):
    """post_tool_use：收集成功写出的产出文件路径。"""

    def on_post_tool_use(self, agent, block, output):
        """解析产出工具路径参数，成功写入则记入 agent._written_paths。

        paras:
            agent: 宿主 Agent
            block: tool_use block（含 name / input）
            output: 工具执行输出文本
        return: None
        """
        path_param = OUTPUT_PATH_PARAM.get(block.name)
        if not path_param:
            return None
        p = block.input.get(path_param)
        ok = not output.startswith(("Error:", "警告:"))
        if block.name == "generate_bib_from_ref_library":
            ok = '"status": "ok"' in output
        if ok and isinstance(p, str) and p:
            try:
                resolved = str(safe_resolve(p))
                agent._written_paths.add(resolved)
                if (agent.name == "WritingAgent" and agent.context is not None
                        and Path(resolved) == (Path(agent.context.output_dir) / "main.tex").resolve()):
                    agent.context.set_main_tex_path(resolved)
            except (AttributeError, OSError, ValueError):
                pass
        return None
