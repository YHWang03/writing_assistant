# 运行环境与依赖

- **Python**：3.12（实测 3.12.13）
- **操作系统**：Windows / Linux / macOS 均可（编译 LaTeX 需要系统安装 TeX 发行版）

## Python 依赖

| 包名 | 实测版本 | 用途 | 必需性 |
| --- | --- | --- | --- |
| anthropic | 0.122.0 | LLM 调用客户端（DeepSeek 的 Anthropic 兼容接口） | 必需 |
| PyYAML | 6.0.3 | 读取 `config.yaml` 全局配置 | 必需 |
| PyMuPDF | 1.28.0 | PDF 文本提取（代码中 `import fitz`） | 必需 |
| python-dotenv | 1.2.2 | 加载 `.env` 环境变量；缺失时代码自动回退到内置解析 | 可选 |
| tokenizers | 0.22.2 | 加载 DeepSeek BPE 词表做精确 token 计数；缺失时自动回退到启发式估算 | 可选 |

一键安装：

```bash
pip install anthropic==0.122.0 PyYAML==6.0.3 PyMuPDF==1.28.0 python-dotenv==1.2.2 tokenizers==0.22.2
```

> 版本号基于开发环境（conda 环境 `agent`）实测记录，可按需要升级，后续如生成 `requirements.txt` 可从此文件转换。

## 系统依赖

| 依赖 | 用途 | 安装方式（Windows） |
| --- | --- | --- |
| pdflatex + bibtex | 编译论文产出 PDF | 安装 [MiKTeX](https://miktex.org/) 或 TeX Live，并确保在 PATH 中 |

## 外部服务

- **DeepSeek API Key**：在 `.env` 中配置 `LLM_API_KEY`（参考 `.env.example`）
- **Semantic Scholar API Key**（可选）：配置 `S2_API_KEY` 后文献检索限流从 1 req/s 提升到约 100 req/s
