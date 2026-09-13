# writing_assistant

多 Agent ReAct 学术论文写作助手。输入种子论文 PDF、参考文献 PDF、创新点/实验描述等手稿，自动产出可编译的 LaTeX 论文（`main.tex` + `references.bib` + `main.pdf`）。基于 DeepSeek 的 Anthropic 兼容 API。

## 功能特性

- **多 Agent 协作**：6 个角色分工，由 MasterAgent 统一协调调度
  - **LiteratureAgent**：PDF 解析、Semantic Scholar 检索验证、文献入库、生成 `references.bib`（Plan-Execute 模式）
  - **WritingAgent**：读取期刊模板，按创新点/实验描述分节撰写 `main.tex`
  - **CitationAgent**：扫描 `\cite`，逐条比对引用与原文，产出引用核查报告
  - **ReviewAgent**：盲审视角评审（只读论文成品，无法访问创新点等作者内部材料）
  - **BuildAgent**：模板搜索/验证/下载，执行 `pdflatex → bibtex → pdflatex × 2` 编译并解析日志
- **三种运行范式可配**：ReAct / Plan-Execute / Reflection，按 Agent 单独配置
- **权限隔离的共享上下文**：`AgentContextView` 按字段控制每个 Agent 的读写范围
- **记忆系统**：情景记忆 + TF-IDF 长期记忆，按 Agent 独立持久化
- **精确 token 计数**：DeepSeek 官方 BPE 词表，带渐进式上下文压缩（缺失词表时自动回退启发式）
- **产出闸门**：Agent 结束前校验必需产出文件已写出且非空
- **并行工具**：PDF 批量解析、引用批量核查使用线程池；Semantic Scholar 429 自动指数退避重试

## Quickstart

### 1. 准备环境

要求 Python 3.12，系统已安装 LaTeX（MiKTeX 或 TeX Live，需含 `pdflatex`、`bibtex`）。依赖说明详见 [requirements.md](requirements.md)。

```bash
conda create -n agent python=3.12 -y
conda activate agent
pip install anthropic==0.122.0 PyYAML==6.0.3 PyMuPDF==1.28.0 python-dotenv==1.2.2 tokenizers==0.22.2
```

### 2. 配置环境变量

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# Linux / macOS
cp .env.example .env
```

编辑 `.env`，填入 DeepSeek API Key：

```
LLM_MODEL_NAME=deepseek-v4-flash
LLM_API_KEY=sk-你的key
LLM_URL=https://api.deepseek.com/anthropic
```

### 3. 准备输入材料

仓库中的 `example/` 各子目录仅保留空目录结构，运行前自行放入材料（路径可在 `config.yaml` 中修改）：

| 目录 / 文件 | 放置内容 |
| --- | --- |
| `example/refs/` | 种子论文 PDF |
| `example/reference/` | 补充参考文献 PDF |
| `example/input/innovations.txt` | 创新点描述 |
| `example/input/prompt.txt` | 写作需求提示词 |
| `example/input/formula.tex` | 公式手稿（可选） |
| `example/experiments/description.txt` | 实验描述（可选） |
| `example/experiments/` | 实验图片（png/jpg/pdf/eps/svg，可选） |
| `example/journal_tex/` | 期刊 LaTeX 模板（没有可让 BuildAgent 联网搜索下载） |

### 4. 运行

```bash
python main.py                    # 交互模式，默认读取 config.yaml
python main.py --config config.yaml
```

启动后在 `[You] >` 提示符输入写作需求即可；输入 `quit` 退出。产出在 `example/output/`，日志在 `logs/`。

## 目录结构

```
writing_assistant/
├── main.py                     # 入口：加载 config.yaml → 构建 PaperContext → 装配 6 个 Agent → 交互运行
├── config.yaml                 # 全局配置：路径、LLM 模型、各 Agent max_steps/run_mode、记忆参数
├── .env.example                # 环境变量模板（复制为 .env 并填入 API Key）
├── requirements.md             # Python 版本与依赖说明
├── resources/
│   └── ds-v4/tokenizer.json    # DeepSeek BPE 词表，token 精确计数用（缺失自动回退）
├── example/                    # 工作区：仅上传空目录结构，材料/产出均不入 git
│   ├── refs/                   #   种子论文 PDF
│   ├── reference/              #   参考文献 PDF
│   ├── input/                  #   创新点/提示词/公式手稿
│   ├── experiments/            #   实验描述与图片
│   ├── journal_tex/            #   期刊 LaTeX 模板
│   ├── library/                #   持久文献库 reference_library.json
│   └── output/                 #   main.tex / references.bib / main.pdf 等产出
├── memory/                     # 各 Agent 长期记忆 JSON（运行时生成，不入 git）
├── logs/                       # 运行日志 + 结构化 JSONL 日志（运行时生成，不入 git）
└── src/
    ├── agents/                 # 6 个业务 Agent（master/literature/writing/citation/review/build）
    ├── core/                   # 框架核心
    │   ├── agent.py            #   Agent 抽象基类：ReAct 循环、工具调用、记忆集成、产出闸门
    │   ├── llm.py              #   LLM 调用层（DeepSeek + Anthropic 兼容接口）
    │   ├── run_modes.py        #   运行范式：ReactMode / PlanExecuteMode / ReflectionMode
    │   ├── paper_context.py    #   PaperContext 共享上下文 + AgentContextView 权限隔离
    │   ├── context_compress.py #   渐进式上下文压缩（70%/90% 双阈值）
    │   ├── token_counter.py    #   BPE 精确计数 + 启发式 fallback
    │   ├── library.py          #   持久文献库读写与按 cite_key 合并去重
    │   ├── paper.py            #   Paper 数据类 + Source 枚举
    │   ├── message.py          #   Message 封装
    │   └── logging_setup.py    #   结构化 JSONL 日志
    ├── memory/                 # 记忆系统：base / working / episodic / long_term / manager
    ├── prompts/                # 各 Agent 系统提示词（.md，流程驱动）
    └── tools/                  # 工具系统
        ├── base.py             #   Tool 抽象基类
        ├── registry.py         #   ToolRegistry（每个 Agent 独立实例，权限隔离）
        └── builtin/            #   内置工具：pdf / search / bib / tex / template / compile
                                #   / dispatch / context / memory_tool / library / finish
```

## 典型工作流

```
Literature（解析 PDF → 检索验证 → 建文献库 → 生成 references.bib）
        ↓
Writing（读模板 → 分节撰写 main.tex，用 cite_key 占位引用）
        ↓
Citation（扫描 \cite → 比对原文 → 补全 .bib → 核查报告）
        ↓
Review（盲审视角评审论文成品）  ←→ Writing 按评审意见修改
        ↓
Build（pdflatex → bibtex → pdflatex × 2，产出 main.pdf）
```
