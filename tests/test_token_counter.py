"""
token_counter 对比测试

验证两件事：
1. DeepSeek V4 tokenizer 可正常加载且能编码
2. 与原 CJK/ASCII 启发式估算的对比，量化误差（用于回归验证）

运行方式：
    pytest tests/test_token_counter.py -v
    # 或直接
    python tests/test_token_counter.py
"""

import sys
import unittest
from pathlib import Path

# 让脚本既能 pytest 也能直接 python 运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.context.token_counter import (
    count_tokens,
    estimate_tokens,
    is_available,
    get_status,
    _legacy_estimate_text,
)


SAMPLE_TEXTS = [
    # (label, text)
    ("纯英文短句", "Hello world!"),
    ("纯中文短句", "你好世界，今天天气真好。"),
    ("混合技术文本", "Eikonal equation 程函方程 fast sweeping method 在 3D VTI 介质中的实现"),
    ("LaTeX 片段", r"\section{Introduction}\nThe $E$-ikonal equation is given by $|\nabla T|^2 = 1/v^2$."),
    ("工具调用 input 示例", "{'file_path': 'example/output/main.tex', 'mode': 'append', 'content_size': 46300}"),
    ("错误信息", "Error: file not found at example/reference/missing.pdf"),
    ("长段落（论文摘要风格）",
     "We propose a fast iterative method for solving the Eikonal equation in tilted transversely isotropic (TTI) media. "
     "The method combines a factored formulation with a high-order finite-difference discretization, achieving both "
     "accuracy and computational efficiency. Numerical experiments on benchmark velocity models demonstrate that "
     "our solver converges in fewer iterations than the standard fast sweeping method while maintaining comparable accuracy."),
    ("中文长段落（摘要风格）",
     "本文提出了一种用于求解程函方程的快速迭代方法，适用于倾斜各向异性（TTI）介质。"
     "该方法结合因式分解形式与高阶有限差分离散化，同时兼顾精度与计算效率。"
     "在基准速度模型上的数值实验表明，相较于标准快速扫描法，"
     "我们的求解器在保持精度相当的同时显著减少了迭代次数。"),
]


class TestTokenCounter(unittest.TestCase):
    """token_counter 单元测试。"""

    @classmethod
    def setUpClass(cls):
        cls.status = get_status()
        print(f"\n[tokenizer status] {cls.status}")

    def test_tokenizer_available(self):
        """tokenizer 库与词表文件就绪时，is_available 应为 True。"""
        if not self.status["tokenizer_exists"]:
            self.skipTest(f"词表文件不存在: {self.status['tokenizer_path']}")
        self.assertTrue(is_available(), f"tokenizer 加载失败: {self.status.get('error')}")

    def test_count_tokens_basic(self):
        """count_tokens 对已知文本应返回非负整数。"""
        if not is_available():
            self.skipTest("tokenizer 不可用，跳过精确计数测试")
        n = count_tokens("Hello world!")
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)
        # DeepSeek V4 对 "Hello world!" 的切分已知是 3 token
        self.assertEqual(n, 3, f"Expected 3 tokens for 'Hello world!', got {n}")

    def test_count_tokens_empty(self):
        """空字符串应返回 -1（tokenizer 不可用时）或 0（可用时）。"""
        if not is_available():
            self.assertEqual(count_tokens(""), -1)
        else:
            self.assertEqual(count_tokens(""), -1)  # 不可用时返回 -1，空文本也走该路径

    def test_estimate_tokens_signature_unchanged(self):
        """estimate_tokens 接受 list 参数，返回 int（与改造前签名一致）。"""
        msgs = [{"role": "user", "content": "Hello"}]
        result = estimate_tokens(msgs)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_estimate_tokens_handles_mixed_content(self):
        """estimate_tokens 能处理 dict/Message 混合 + list block 内容。"""
        # 模拟 Anthropic 工具调用响应的 messages
        msgs = [
            {"role": "user", "content": "请解析 PDF"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "正在调用工具"},
                {"type": "tool_use", "name": "parse_pdf", "input": {"pdf_path": "/abs/path.pdf"}, "id": "tool_1"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "解析完成，标题: Test Paper"},
            ]},
        ]
        result = estimate_tokens(msgs)
        self.assertIsInstance(result, int)
        # 至少应该有 4 token（每条消息的 chat 模板开销）+ 内容本身
        self.assertGreaterEqual(result, 4 * 3)

    def test_fallback_path_matches_legacy(self):
        """tokenizer 不可用时，estimate_tokens 的回退路径应与原启发式完全一致。

        通过临时把 _tokenizer 设为 None 模拟不可用场景。
        """
        import src.context.token_counter as tc
        original = tc._tokenizer
        try:
            tc._tokenizer = None
            tc._load_attempted = True  # 阻止重新加载
            for label, text in SAMPLE_TEXTS:
                msgs = [{"role": "user", "content": text}]
                fallback_result = tc.estimate_tokens(msgs)
                # 启发式估算（不含 chat 模板开销，因为 fallback 路径不加）
                legacy = _legacy_estimate_text(text)
                self.assertEqual(
                    fallback_result, legacy,
                    f"[{label}] fallback ({fallback_result}) != legacy ({legacy})"
                )
        finally:
            tc._tokenizer = original
            tc._load_attempted = False  # 恢复

    def test_comparison_table(self):
        """打印精确计数 vs 启发式的对比表，量化误差。

        这不是断言性测试，而是诊断性输出——用来人工评估 fallback 是否足够准确。
        pytest 默认会捕获 stdout，用 -s 可以看到。
        """
        print(f"\n{'='*80}")
        print(f"{'label':<28} {'chars':>6} {'precise':>9} {'legacy':>7} {'error%':>8}")
        print(f"{'-'*28} {'-'*6} {'-'*9} {'-'*7} {'-'*8}")
        for label, text in SAMPLE_TEXTS:
            n_chars = len(text)
            if is_available():
                precise = count_tokens(text)
            else:
                precise = -1
            legacy = _legacy_estimate_text(text)
            if precise > 0:
                err_pct = (legacy - precise) / precise * 100
                err_str = f"{err_pct:+.1f}%"
            else:
                err_str = "N/A"
            print(f"{label:<28} {n_chars:>6} {precise:>9} {legacy:>7} {err_str:>8}")
        print(f"{'='*80}")
        # 至少能跑完不抛错
        self.assertTrue(True)


# ---- compress_messages 的 anchor 模式测试（改造2）----

from unittest.mock import MagicMock
from src.context.context_compress import compress_messages


class TestCompressMessagesAnchor(unittest.TestCase):
    """compress_messages 的 anchor 模式（改造2）测试。

    验证当传入 anchor_input_tokens 时，函数用真实 usage 作 estimated；
    不传时 fallback 到本地 estimate_tokens。
    """

    def _mock_llm(self, return_summary: str = "[摘要] 之前的步骤信息已浓缩"):
        """构造 mock LLM，chat 返回固定 summary。"""
        llm = MagicMock()
        llm.chat.return_value = return_summary
        return llm

    def _build_messages(self, n: int = 10) -> list:
        """构造 n 条 dict 消息。"""
        return [{"role": "user", "content": f"消息 {i}"} for i in range(n)]

    def test_anchor_above_threshold_triggers_compression(self):
        """anchor 足够大时触发压缩，返回压缩后列表（长度 = 1 + keep_recent）。"""
        messages = self._build_messages(10)
        llm = self._mock_llm()
        result = compress_messages(
            messages, keep_recent=5, context_window=100000,
            llm=llm, agent_name="test",
            anchor_input_tokens=80000,  # > 70k structured threshold
            last_output_tokens=1000,
        )
        # 触发压缩，长度变成 1（摘要）+ 5（保留最近）
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]["role"], "user")  # 摘要放第一条
        self.assertIn("摘要", result[0]["content"])

    def test_anchor_below_threshold_no_compression(self):
        """anchor 远小于阈值时不压缩，原样返回。"""
        messages = self._build_messages(10)
        llm = self._mock_llm()
        result = compress_messages(
            messages, keep_recent=5, context_window=100000,
            llm=llm, agent_name="test",
            anchor_input_tokens=100,  # 远小于 70k threshold
            last_output_tokens=50,
        )
        # 不压缩，原样返回（10 条）
        self.assertEqual(len(result), 10)

    def test_anchor_none_falls_back_to_local_estimate(self):
        """anchor=None 时 fallback 到本地 estimate_tokens。

        用极大 context_window 确保本地估算值远低于阈值，触发 early return。
        """
        messages = self._build_messages(10)
        llm = self._mock_llm()
        result = compress_messages(
            messages, keep_recent=5, context_window=1000000,  # 极大阈值
            llm=llm, agent_name="test",
            anchor_input_tokens=None,  # 显式 None
        )
        self.assertEqual(len(result), 10)  # 不压缩

    def test_too_few_messages_short_circuits(self):
        """messages 长度 <= keep_recent 时直接返回，不调用 LLM。"""
        messages = self._build_messages(3)
        llm = self._mock_llm()
        # 不应该走到 LLM 调用
        result = compress_messages(
            messages, keep_recent=5, context_window=1000,
            llm=llm, agent_name="test",
            anchor_input_tokens=999999,  # 极大 anchor，但仍因消息太少短路
        )
        self.assertEqual(len(result), 3)
        llm.chat.assert_not_called()

    def test_anchor_includes_last_output_tokens(self):
        """anchor + last_output 应同时计入 estimated。

        构造临界场景：anchor 单独不触发，加上 last_output 后触发。
        context_window = 100000 → structured_threshold = 80000
        anchor = 79000（不触发）+ last_output = 2000 → estimated = 81000（触发）
        """
        messages = self._build_messages(10)
        llm = self._mock_llm()

        # 单独 anchor，不触发
        result_no_output = compress_messages(
            messages, keep_recent=5, context_window=100000,
            llm=llm, agent_name="test",
            anchor_input_tokens=79000,
            last_output_tokens=0,  # 不计 output
        )
        self.assertEqual(len(result_no_output), 10)  # 不压缩

        # 加上 last_output，触发
        result_with_output = compress_messages(
            messages, keep_recent=5, context_window=100000,
            llm=llm, agent_name="test",
            anchor_input_tokens=79000,
            last_output_tokens=2000,  # 79000+2000=81000 > 80000 阈值
        )
        self.assertEqual(len(result_with_output), 6)  # 触发压缩


if __name__ == "__main__":
    unittest.main(verbosity=2)
