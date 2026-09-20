"""LaTeX 编译与日志解析工具。"""

import subprocess
import re
import json as json_mod
from pathlib import Path
from ..base import Tool
from ._safe_path import safe_resolve


class CompileLatexTool(Tool):
    """编译 .tex 为 PDF，按需执行 latex + bibtex 多轮编译（subprocess 调终端命令）"""

    def __init__(self):
        super().__init__(
            name="compile_latex",
            description="编译 LaTeX 文件生成 PDF。自动检测是否需要 bibtex，"
                        "并按需执行 latex → bibtex → latex → latex 多轮编译。"
                        "自动尝试 pdflatex → xelatex → lualatex。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "tex_path": {"type": "string", "description": ".tex 文件路径"},
                "output_dir": {"type": "string", "description": "输出目录（默认为 .tex 所在目录）"},
            },
            "required": ["tex_path"],
        }

    def execute(self, tex_path: str, output_dir: str = "") -> str:
        """编译 .tex 为 PDF，按需执行 latex → bibtex → latex 多轮编译。

        paras:
            tex_path: .tex 文件路径
            output_dir: 输出目录，缺省为 .tex 所在目录
        return: JSON 字符串；成功含 pdf_path/compiler/steps/errors 等字段，
                全部编译器失败时含 all_tried 明细
        """
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

        _AUX_EXTS = (".aux", ".log", ".out", ".toc", ".lof", ".lot",
                     ".bbl", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz",
                     ".nav", ".snm", ".vrb")
        for ext in _AUX_EXTS:
            f = out_dir / f"{tex.stem}{ext}"
            if f.exists():
                f.unlink()

        needs_bibtex = self._detect_bibtex(tex)

        compilers = ["pdflatex", "xelatex", "lualatex"]
        final_pdf = out_dir / f"{tex.stem}.pdf"
        all_tried = []

        for compiler in compilers:
            for ext in _AUX_EXTS:
                f = out_dir / f"{tex.stem}{ext}"
                if f.exists():
                    f.unlink()
            if final_pdf.exists():
                final_pdf.unlink()

            steps = []
            all_stdout = []

            r1 = self._run_latex(compiler, tex, out_dir, work_dir)
            steps.append({"step": f"{compiler} (1)", "return_code": r1["rc"]})
            all_stdout.append(r1["stdout"])
            crash1 = self._is_crash(r1["rc"])

            if crash1:
                all_tried.append({"compiler": compiler, "result": "crashed on pass 1", "rc": r1["rc"]})
                continue

            if needs_bibtex:
                r_bib = self._run_bibtex(tex, out_dir, work_dir)
                steps.append({"step": "bibtex", "return_code": r_bib["rc"]})
                all_stdout.append(r_bib["stdout"])

                r2 = self._run_latex(compiler, tex, out_dir, work_dir)
                steps.append({"step": f"{compiler} (2)", "return_code": r2["rc"]})
                all_stdout.append(r2["stdout"])
                if self._is_crash(r2["rc"]):
                    all_tried.append({"compiler": compiler, "result": "crashed on pass 2", "rc": r2["rc"]})
                    continue

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
                all_tried.append({"compiler": compiler, "result": "compiled but no PDF", "rc": r3.get("rc", r2["rc"]) if needs_bibtex else r2["rc"]})
                continue

        return json_mod.dumps({
            "success": False,
            "pdf_path": "",
            "error": "All compilers failed",
            "all_tried": all_tried,
            "_version": "V3",
        })

    def _is_crash(self, rc: int) -> bool:
        """检测返回码是否为进程崩溃。

        paras:
            rc: 进程返回码
        return: 是否崩溃（负返回码映射 Windows 异常码）
        """
        if rc >= 0:
            return False
        # Windows 异常码：0xE06D7363=C++ 异常，0xC0000005=访问冲突，0xC0000135=缺 DLL，0xC0000409=栈保护
        u = rc & 0xFFFFFFFF
        return u in (0xE06D7363, 0xC0000005, 0xC0000135, 0xC0000409) or u > 0xC0000000

    def _detect_bibtex(self, tex_file: Path) -> bool:
        """检测 .tex 是否需要 bibtex 处理参考文献。

        paras:
            tex_file: .tex 文件路径
        return: 是否包含参考文献相关命令
        """
        try:
            content = tex_file.read_text(encoding="utf-8", errors="replace")
            return bool(re.search(r'\\(bibliography|bibliographystyle|addbibresource)\b', content))
        except Exception:
            return False

    def _run_latex(self, compiler: str, tex: Path, out_dir: Path, work_dir: Path) -> dict:
        """运行一轮 LaTeX 编译命令。

        paras:
            compiler: 编译器名（pdflatex/xelatex/lualatex）
            tex: .tex 文件路径
            out_dir: 输出目录
            work_dir: 子进程工作目录
        return: {"rc", "stdout", "stderr"} 字典；超时或异常时 rc 为 -1
        """
        # shell=True 使 Windows 能找到 .bat 包装脚本
        cmd = f'"{compiler}" -interaction=nonstopmode -output-directory="{out_dir}" "{tex}"'
        try:
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
        """对 .aux 运行 bibtex。

        paras:
            tex: .tex 文件路径
            out_dir: 输出目录（同时作为 bibtex 工作目录）
            work_dir: 未使用，保持与 _run_latex 签名一致
        return: {"rc", "stdout", "stderr"} 字典；超时或异常时 rc 为 -1
        """
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
        """从编译输出中提取错误与警告行。

        paras:
            stdout: 编译器输出文本
        return: (errors, warnings) 元组，每项为截断到 200 字符的行
        """
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
    """解析 LaTeX 编译日志，提取 error 与 warning"""

    def __init__(self):
        super().__init__(
            name="parse_latex_log",
            description="解析 LaTeX 编译日志，提取错误和警告信息。"
        )

    def get_parameters(self) -> dict:
        """返回工具参数的 JSON Schema 定义。

        return: input_schema 字典
        """
        return {
            "type": "object",
            "properties": {
                "log_path": {"type": "string", "description": ".log 文件路径"},
            },
            "required": ["log_path"],
        }

    def execute(self, log_path: str) -> str:
        """解析 LaTeX 编译日志，提取错误块（! 开头及其后续行）与警告行。

        paras:
            log_path: .log 文件路径
        return: JSON 字符串，含 errors/warnings/error_count/warning_count
        """
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
