# writing_assistant

多 Agent 协作的学术论文写作助手。输入创新点和实验材料等，可由agent自行通过LaTeX模板生成论文。

## Quickstart

### 1. 准备环境

需要 **Python 3.12+**，以及安装了 `pdflatex` 和 `bibtex` 的 LaTeX 环境（MiKTeX 或 TeX Live）。确认这两个命令可在终端直接运行。

在项目根目录安装依赖：

```bash
pip install -r requirements.txt
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

编辑 `.env`，填入 API Key 和 Anthropic 兼容接口地址：

```dotenv
LLM_API_KEY=sk-你的key
LLM_URL=https://api.deepseek.com/anthropic
```

模型配置, 文件路径等相关配置在 `config.yaml` 中设置。



### 3. 运行

```bash
python main.py
```

### 查看运行日志

运行后打开 `logs/index.html`，选择一次运行即可查看调用明细、耗时与 token 用量。报告在任务结束时自动更新，已打开的页面需刷新。

默认保存完整模型请求、响应和工具详情；可通过环境变量 `LOG_DETAILS=0` 关闭。日志可能包含论文材料和提示词，分享前请检查内容。

## 目录结构

```text
writing_assistant/
├── main.py                     # 入口：启动命令行交互
├── config.yaml                 # 路径、模型、Agent 运行模式与记忆配置
├── .env.example                # 环境变量模板（复制为 .env）
├── requirements.txt            # 运行依赖
├── pyproject.toml              # 项目元数据与开发工具配置
├── resources/                  # 本地 tokenizer 资源
├── example/                    # 输入材料与工作产出
│   ├── refs/                   #   种子论文 PDF
│   ├── reference/              #   补充参考文献 PDF
│   ├── input/                  #   创新点、写作要求与公式手稿
│   ├── experiments/            #   实验描述与图片
│   ├── journal_tex/            #   期刊 LaTeX 模板
│   ├── library/                #   持久文献库 reference_library.json
│   └── output/                 #   LaTeX、BibTeX、PDF 与引用核查报告
├── src/                        # 全部源码
│   ├── cli.py                  #   参数解析与交互循环
│   ├── config.py               #   配置加载与校验
│   ├── bootstrap.py            #   Agent、工具与共享上下文装配
│   ├── lifecycle.py            #   任务执行与退出时的数据保存
│   ├── agents/                 #   Master / Literature / Writing / Citation / Review / Build
│   ├── core/                   #   LLM 调用与 ReAct / Plan-Execute 执行机制
│   ├── domain/                 #   论文数据、文献库、引用证据与访问权限
│   ├── tools/                  #   PDF 解析、文献检索、文件操作与 LaTeX 编译
│   ├── hooks/                  #   记忆召回、产出校验与任务收尾
│   ├── context/                #   token 计数、上下文压缩与 PDF 解析缓存
│   ├── memory/                 #   长期记忆存储与语义召回
│   ├── prompts/                #   各 Agent 的系统提示词
│   └── observability/          #   日志、调用追踪与离线 HTML 报告
├── tests/                      # 自动化测试与手动验证脚本
├── memory/                     # 运行时生成的长期记忆与向量缓存
└── logs/                       # 运行日志与离线报告
```

## 处理流程

```text
论文 PDF + 创新点 + 实验材料 + LaTeX 模板
                     ↓
                MasterAgent 协调
                     ↓
Literature：解析 PDF / 检索文献 → 建文献库 → 导出 references.bib
                     ↓
Writing：读取材料与模板 → 分节撰写 main.tex
                     ↓
Citation：扫描引用 → 检查摘要证据 → 生成引用核查报告
                     ↓
Review：评审论文成品 ←→ Writing：按评审意见修改
                     ↓
Build：pdflatex → bibtex → pdflatex × 2 → main.pdf
```

MasterAgent 根据任务选择和调度各角色。支持 ReAct 与 Plan-Execute，可在 `config.yaml` 的 `agents.run_mode` 中按角色设置。