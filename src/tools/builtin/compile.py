"""LaTeX 编译工具 V3
- CompileLatexTool: 编译 .tex 为 PDF
- ParseLatexLogTool: 解析编译日志
"""

import subprocess
import re
import json as json_mod
from pathlib import Path
from ..base import Tool
from ._safe_path import safe_resolve


class CompileLatexTool(Tool):
    """
    编译 LaTeX → PDF，自动执行 latex + bibtex 多轮编译
    通过subprocess.run 执行终端命令
    """

    def __init__(self):
        super().__init__(
            name="compile_latex",
            description="编译 LaTeX 文件生成 PDF。自动检测是否需要 bibtex，"
                        "并按需执行 latex → bibtex → latex → latex 多轮编译。"
                        "自动尝试 pdflatex → xelatex → lualatex。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
                "output_dir": {"type": "string", "description": "输出目录（默认为 .tex 所在目录）"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str, output_dir: str = "") -> str:
        try:
            tex = safe_resolve(tex_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e), "_version": "V3"})
        if not tex.exists():
            return json_mod.dumps({"error": f"文件不存在: {tex_path}", "_version": "V3"})

        if output_dir:
            try:
                out_dir = safe_resolve(output_dir)
            except ValueError as e:
                return json_mod.dumps({"error": str(e), "_version": "V3"})
        else:
            out_dir = tex.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = tex.parent

        # 编译前清理旧的辅助文件
        _AUX_EXTS = (".aux", ".log", ".out", ".toc", ".lof", ".lot",
                     ".bbl", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz",
                     ".nav", ".snm", ".vrb")
        for ext in _AUX_EXTS:
            f = out_dir / f"{tex.stem}{ext}"
            if f.exists():
                f.unlink()

        needs_bibtex = self._detect_bibtex(tex)

        # 尝试所有编译器：pdflatex, xelatex, lualatex
        compilers = ["pdflatex", "xelatex", "lualatex"]
        final_pdf = out_dir / f"{tex.stem}.pdf"
        all_tried = []

        for compiler in compilers:
            # 清理
            for ext in _AUX_EXTS:
                f = out_dir / f"{tex.stem}{ext}"
                if f.exists():
                    f.unlink()
            if final_pdf.exists():
                final_pdf.unlink()

            steps = []
            all_stdout = []

            # 第1轮
            r1 = self._run_latex(compiler, tex, out_dir, work_dir)
            steps.append({"step": f"{compiler} (1)", "return_code": r1["rc"]})
            all_stdout.append(r1["stdout"])
            crash1 = self._is_crash(r1["rc"])

            if crash1:
                all_tried.append({"compiler": compiler, "result": "crashed on pass 1", "rc": r1["rc"]})
                continue

            # bibtex
            if needs_bibtex:
                r_bib = self._run_bibtex(tex, out_dir, work_dir)
                steps.append({"step": "bibtex", "return_code": r_bib["rc"]})
                all_stdout.append(r_bib["stdout"])

                # 第2轮
                r2 = self._run_latex(compiler, tex, out_dir, work_dir)
                steps.append({"step": f"{compiler} (2)", "return_code": r2["rc"]})
                all_stdout.append(r2["stdout"])
                if self._is_crash(r2["rc"]):
                    all_tried.append({"compiler": compiler, "result": "crashed on pass 2", "rc": r2["rc"]})
                    continue

                # 第3轮
                r3 = self._run_latex(compiler, tex, out_dir, work_dir)
                steps.append({"step": f"{compiler} (3)", "return_code": r3["rc"]})
                all_stdout.append(r3["stdout"])
                if self._is_crash(r3["rc"]):
                    all_tried.append({"compiler": compiler, "result": "crashed on pass 3", "rc": r3["rc"]})
                    continue
            else:
                r2 = self._run_latex(compiler, tex, out_dir, work_dir)
                steps.append({"step": f"{compiler} (2)", "return_code": r2["rc"]})
                all_stdout.append(r2["stdout"])
                if self._is_crash(r2["rc"]):
                    all_tried.append({"compiler": compiler, "result": "crashed on pass 2", "rc": r2["rc"]})
                    continue

            combined = "\n".join(all_stdout)
            success = final_pdf.exists()
            errs, warns = self._extract_diagnostics(combined)

            if success:
                return json_mod.dumps({
                    "success": True,
                    "pdf_path": str(final_pdf),
                    "compiler": compiler,
                    "bibtex_used": needs_bibtex,
                    "steps": steps,
                    "error_count": len(errs),
                    "warning_count": len(warns),
                    "errors": errs[-10:],
                    "warnings": warns[-10:],
                    "stdout_tail": combined[-2000:],
                    "_version": "V3",
                })
            else:
                # 编译了但没有PDF — LaTeX错误（不是崩溃）
                all_tried.append({"compiler": compiler, "result": "compiled but no PDF", "rc": r3.get("rc", r2["rc"]) if needs_bibtex else r2["rc"]})
                # 继续尝试下一个编译器
                continue

        # 所有编译器都失败了
        return json_mod.dumps({
            "success": False,
            "pdf_path": "",
            "error": "All compilers failed",
            "all_tried": all_tried,
            "_version": "V3",
        })

    def _is_crash(self, rc: int) -> bool:
        """检测是否为崩溃（C++运行时异常等）"""
        if rc >= 0:
            return False
        u = rc & 0xFFFFFFFF
        return u in (0xE06D7363, 0xC0000005, 0xC0000135, 0xC0000409) or u > 0xC0000000

    def _detect_bibtex(self, tex_file: Path) -> bool:
        try:
            content = tex_file.read_text(encoding="utf-8", errors="replace")
            return bool(re.search(r'\\(bibliography|bibliographystyle|addbibresource)\b', content))
        except Exception:
            return False

    def _run_latex(self, compiler: str, tex: Path, out_dir: Path, work_dir: Path) -> dict:
        # 使用 shell=True，使 Windows 能找到 .bat 包装脚本
        cmd = f'"{compiler}" -interaction=nonstopmode -output-directory="{out_dir}" "{tex}"'
        try:
            # subprocess.run 相当于 在python程序中开一个子进程执行终端命令
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120,
                cwd=str(work_dir),
                shell=True,
            )
            return {
                "rc": result.returncode,
                "stdout": result.stdout[-3000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"rc": -1, "stdout": "", "stderr": "编译超时"}
        except Exception as e:
            return {"rc": -1, "stdout": "", "stderr": str(e)}

    def _run_bibtex(self, tex: Path, out_dir: Path, work_dir: Path) -> dict:
        aux = out_dir / f"{tex.stem}.aux"
        if not aux.exists():
            return {"rc": -1, "stdout": "", "stderr": f".aux 文件不存在: {aux}"}
        try:
            result = subprocess.run(
                f'bibtex "{aux.stem}"',
                capture_output=True, text=True, timeout=60,
                cwd=str(out_dir),
                shell=True,
            )
            return {
                "rc": result.returncode,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"rc": -1, "stdout": "", "stderr": "bibtex 超时"}
        except Exception as e:
            return {"rc": -1, "stdout": "", "stderr": str(e)}

    def _extract_diagnostics(self, stdout: str) -> tuple:
        errors = []
        warnings = []
        for line in stdout.split("\n"):
            stripped = line.strip()
            if stripped.startswith("!") or "Error" in stripped:
                errors.append(stripped[:200])
            elif "Warning" in stripped:
                warnings.append(stripped[:200])
        return errors, warnings


class ParseLatexLogTool(Tool):
    """
    解析 LaTeX 编译日志
    读取latex编译日志文件，通过解析文本内容中'!'和'Warning'开头的line，提取error和warning信息
    """

    def __init__(self):
        super().__init__(
            name="parse_latex_log",
            description="解析 LaTeX 编译日志，提取错误和警告信息。"
        )

    def get_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "log_path": {"type": "string", "description": ".log 文件路径"},
            },
            "required": ["log_path"],
        }

    def execute(self, log_path: str) -> str:
        try:
            path = safe_resolve(log_path)
        except ValueError as e:
            return json_mod.dumps({"error": str(e)})
        if not path.exists():
            return json_mod.dumps({"error": f"日志文件不存在: {log_path}"})
        content = path.read_text(encoding="utf-8", errors="replace")
        errors = []
        warnings = []
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("!"):
                block_lines = [line.strip()]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if (next_line.startswith("l.") or
                        next_line.startswith("<") or
                        next_line.startswith("?") or
                        (next_line and next_line[0] == " ")):
                        block_lines.append(next_line.strip())
                        i += 1
                    else:
                        break
                errors.append("\n".join(block_lines))
            elif "Warning" in line:
                warnings.append(line.strip())
                i += 1
            else:
                i += 1
        return json_mod.dumps({
            "errors": errors, "warnings": warnings,
            "error_count": len(errors), "warning_count": len(warnings),
        })