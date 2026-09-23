# writing_assistant

多 Agent ReAct 学术论文写作助手。输入种子论文 PDF、参考文献 PDF、创新点/实验描述等手稿，自动产出可编译的 LaTeX 论文（`main.tex` + `references.bib` + `main.pdf`）

## 文献库与 BibTeX 同步

结构化 `reference_library.json` 是文献维护的唯一入口：新增用 `add_reference`，
修正用 `update_reference(cite_key, changes)`，删除确认无用的条目用 `remove_reference`。
修改后 `write_library` 保存完整快照，`generate_bib_from_ref_library` 从当前库重新导出，
不再保留同名旧条目或复活已删除记录。通用 `write_file` 禁止直接写 `.bib` 和库 JSON。
导出会先保存库，再原子替换 `.bib`，旧 `.bib` 有变更时保留带内容摘要的 `.bak`。
两文件并非跨文件事务；导出失败时库可能已保存，修复文件权限等问题后重新导出即可。

`Paper.entry_type` 保存 article/book/inproceedings 等类型；`bib_fields` 保存 series、
booktitle 等扩展字段。显式导入旧 BibTeX 时保留原摘要和来源；导入器只接受项目生成的
花括号字段语法，其他 BibTeX 语法会报错而不会静默丢失。不会启动时自动覆盖用户修正。
本次现有库迁移前备份为 `example/library/reference_library.before-bib-sync.json.bak`。

## 功能特性

### 文献存在性验证规则

用户提供的 PDF 直接解析，不联网验证其存在；OCR/元数据不清楚时回看 PDF 或保留缺项，
只有用户明确要求补全/校对时才联网。实际搜索获得的文献不重复调用 `verify_paper`。
只有模型凭记忆提出的文献需要搜索验证，未确认的猜测不得作为已确认文献入库。

`verify_paper` 只核对规范化标题（忽略大小写、标点、连字符和多余空白），不做模糊相似匹配。
标题一致即 `verified=true`，作者、年份和 DOI 不参与判断；同名多版本返回候选列表，
不自动选择版本元数据。网络失败为 `search_failed`，空结果为 `not_found`，标题不同为
`unconfirmed`，三者均不等于文献不存在。通过只代表检索到同名文献，不代表引用论断正确。
新增文献由代码门禁检查来源：真实搜索结果、配置输入列表中的 PDF 解析结果、成功的标题验证
分别签发来源记录。`add_reference` 不再向模型暴露 `source` 参数；旧调用即使传入该字段
也不能绕过门禁。没有来源记录的标题在入库时自动验证，失败则拒绝入库；已知来源不会重复
联网。来源字段不能通过更新工具伪造，修改标题也必须重新通过门禁。
来源记录随新文献保存，旧库不因旧的 `source` 标签自动获得新凭据；已有历史数据未在此次
变更中批量重验或删除。此门禁针对 Agent 工具调用，不是对能直接修改本地文件的人的安全隔离。

### 引用证据与报告有效性

`validate_all_citations` 按每次引用出现位置分别核查，不再只保留同一引用键的最后一处。
结果分为“摘要支持、证据不足、存在矛盾、检查失败”；模型返回绿勾本身
不能判为通过。程序验证证据摘录确实出现在所提供材料中，并记录论断上下文、TeX行号、
摘要字段位置。语义是否真正支持仍是模型判断，需要人工复核。

Citation 仅使用共享文献库摘要（最多1500字符）与正文引用上下文，不读取参考文献PDF，
也不为报告指纹扫描PDF。摘要支持为通过；矛盾给出原因和修改建议；证据不足保留待确认；
检查失败报告技术问题，不作为正文有错的依据。后两者不会自动升级到全文核查。

工具自动生成 `citation_evidence.json` 和 `citation_report.txt`，通用文件工具不能覆盖它们。
正文、bib或文献库变化后，任务结束时程序会检查报告指纹、标记过期并在最终
回答前显示权威状态；重新调用核查工具只复用论断、书目信息与证据均未变化的结果。
旧版自由文本报告没有指纹，视为未验证；不会自动将旧报告升级为通过。
当前范围是指定的单个 TeX 文件，不展开 `input/include`，也不是穷尽全文检索或人工审稿。

### PDF 解析缓存与重试

`parse_pdf` 与 `parse_and_store` 共用按 PDF 内容 SHA-256 和解析版本标识的缓存。
成功元数据保存在 `.cache/pdf_metadata/`（不提交 Git），重启或改文件名仍可复用；
文件内容变化会重新解析。缓存包含论文元数据，不包含全文；可手动清理以强制重建。
旧运行没有此缓存，需要新版本成功解析一次后才能复用，不会从旧日志猜测元数据。

损坏/加密 PDF、无可提取文本、无效元数据及截断恢复失败不会原样重复调用模型。
连接异常、限流和服务端临时错误最多追加一次提取尝试；SDK 自带的网络重试与
LLM 层截断恢复仍各自有界，`attempts` 是提取尝试次数而非底层 HTTP 次数。
失败结果在当前解析器生命周期内保留，重复派发会直接报告失败；修复后重启可再试。
单文件失败不触发整批重跑，同进程并发的同内容文件只解析一次。

工具返回逐文件 `status`、`cache_hit`、`attempts`、`error_kind`，并写入 `pdf_parse`
结构化事件。`cached` 可包含之前的失败结果，不能当作成功入库数量；`stored` 表示
本次提交入库的有效记录数（文献库按引用键去重），不是新增唯一文献数。

- **多 Agent 协作**：6 个角色分工，由 MasterAgent 统一协调调度
  - **LiteratureAgent**：PDF 解析、Semantic Scholar 检索验证、文献入库、生成 `references.bib`（Plan-Execute 模式）
  - **WritingAgent**：读取期刊模板，按创新点/实验描述分节撰写 `main.tex`
  - **CitationAgent**：扫描 `\cite`，逐条比对引用与原文，产出引用核查报告
  - **ReviewAgent**：盲审视角评审（只读论文成品，无法访问创新点等作者内部材料）
  - **BuildAgent**：验证用户模板或内置模板，执行 `pdflatex → bibtex → pdflatex × 2` 编译并解析日志
- **两种运行范式可配**：ReAct / Plan-Execute，可按 Agent 单独配置
- **权限隔离的共享上下文**：可变数据只提供不可变快照，修改必须经过 `add_reference`、`set_section` 等授权命令
- **记忆系统**：按 Agent 独立持久化，使用 LLM 选择并在失败时降级为关键词召回
- **产出闸门**：Agent 结束前校验必需产出文件已写出且非空
- **并行工具**：PDF 批量解析、引用批量核查

## Quickstart

### 1. 准备环境

要求 Python 3.12，系统已安装 LaTeX（MiKTeX 或 TeX Live，需含 `pdflatex`、`bibtex`）

```bash
conda create -n agent python=3.12 -y
conda activate agent
pip install -r requirements.txt
# 开发环境（包含 pytest / Ruff / mypy）也可以使用：
pip install -e ".[dev]"
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
├── main.py                     # 最小可执行入口
├── config.yaml                 # 全局配置：路径、LLM 模型、各 Agent max_steps/run_mode、记忆参数
├── .env.example                # 环境变量模板（复制为 .env 并填入 API Key）
├── pyproject.toml              # 项目元数据、依赖与开发工具配置
├── requirements.txt            # 固定版本的运行依赖
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
    ├── config.py               # 类型化配置、默认值与启动前校验
    ├── bootstrap.py            # Context、Agent、权限和依赖装配
    ├── lifecycle.py            # 任务执行及异常退出时的持久化
    ├── cli.py                  # 参数解析和交互循环
    ├── agents/                 # 声明式业务 Agent（工具、Hook、产出规格）
    ├── core/                   # 框架核心
    │   ├── agent.py            #   Agent 公共状态、能力装配和任务生命周期
    │   ├── llm.py              #   LLM 调用层（DeepSeek + Anthropic 兼容接口）
    │   ├── run_modes.py        #   ReAct / Plan-Execute 运行器
    │   ├── utils.py            #   消息内容解析辅助函数
    │   └── message.py          #   Message 封装
    ├── domain/                 # 论文业务模型、权限与持久化
    │   ├── paper.py            #   Paper 数据类 + Source 枚举
    │   ├── paper_context.py    #   PaperContext + AgentContextView 权限隔离
    │   └── library.py          #   持久文献库读写与按 cite_key 合并去重
    ├── context/                # 对话上下文管理
    │   ├── context_compress.py #   四阶段压缩（80% 触发，64% 整理目标）
    │   └── token_counter.py    #   BPE 精确计数 + 启发式 fallback
    ├── observability/          # 日志与用量观测，不依赖业务 Agent
    │   ├── logging_setup.py    #   控制台、文本、JSONL 初始化与报告更新触发
    │   ├── telemetry.py        #   调用归属、token 统计和任务汇总
    │   ├── log_report.py       #   离线报告和日志首页生成
    │   └── log_report.html     #   自包含报告模板
    ├── memory/                 # 记忆系统：AgentMemory 持久化与召回
    ├── prompts/                # 各 Agent 系统提示词（.md，流程驱动）
    └── tools/                  # 工具系统
        ├── base.py             #   Tool 抽象基类
        ├── registry.py         #   ToolRegistry（每个 Agent 独立实例，权限隔离）
        └── builtin/            #   PDF、检索、文件、编译等内置工具
            ├── bib.py          #   旧版 BibTeX 统一导入的兼容入口
            └── citations/     #   引用与 BibTeX 工具
                ├── generation.py #  条目生成与论文总结
                ├── lookup.py     #  引用扫描与文献查询
                ├── validation.py #  引用比对与批量校验
                ├── storage.py    #  BibTeX 写入与文献入库
                └── common.py     #  共享转义函数
```

职责边界：`core/` 负责 Agent 执行机制；`domain/` 负责论文业务数据；
`context/` 负责对话长度与压缩；`observability/` 负责日志和用量报告。
新增工具按业务职责归组，Agent 仍通过 `tools.builtin` 的统一导出使用工具。

## 长期记忆开关

`config.yaml` 的 `memory.enabled` 为六个 Agent 分别提供布尔开关：

```yaml
memory:
  enabled:
    master: true
    literature: false
    writing: false
    citation: false
    review: false
    build: false
```

修改后重启生效。未配置的开关默认仅 master 开启；使用 `true` / `false`，
不要写成带引号的字符串。关闭的 Agent 不创建长期记忆实例，不加载、召回、提取、
整理或保存记忆；已有记忆文件保留，再开启时可继续使用。
当前对话历史、上下文压缩和持久文献库不受此开关影响。

## 模型用量日志

双击 `logs/index.html` 打开日志首页，选择一次运行进入离线 HTML 报告。
无需命令、服务器或网络连接。启动时补齐已有日志报告，每次任务结束（包括异常）
自动更新 `logs/reports/<运行名>.html`；已打开的页面刷新即可查看最新数据。
报告支持任务、Agent、模型、工具、用途、状态和时间筛选，以及调用 / 请求 ID、
错误类型搜索。统计随筛选更新，提供消耗排名、耗时分布、分页调用明细和原始 usage。
旧日志没有 `llm_call` 明细时会明确提示，不能追溯补出未记录的用量。

文本日志位于 `logs/`，逐行 JSON 日志位于 `logs/structured/`。
每次 `LLM.chat()` 或 `chat_with_tools()` 调用产生一个 `llm_call` 事件，包含：

- `task_id`、`call_id`、API `request_id`，以及模型、Agent、工具、用途。
- `input_tokens`、`output_tokens`、缓存读取/写入 token（仅记录 API 返回值）。
- 完整原始 `usage`、调用耗时 `duration`（秒）、成功/失败状态。

规划、总结、上下文压缩、记忆召回/提取/整理和工具内部模型调用均纳入记录。
并行工具保留任务和 Agent 归属。每次 CLI 输入的任务结束时（包括异常）输出
`task_usage` 汇总，按模型、Agent、用途和工具分别统计。

缺失的 usage 字段记录为 `null`，汇总记录各字段的 `*_unknown_calls`；
有已知值时显示已知部分之和，不能把它视为完整账单。缓存字段单列，不与输入
token 擅自相加，避免不同兼容 API 的统计口径造成重复计算。原有 `agent_step`
中的上下文 token 只用于观察上下文占用，不重复计入调用消耗。

每个事件对应一次 SDK 调用。SDK 内部自动重试的独立请求及其消耗不可见；应用层
显式重试则各自记录。失败且未返回 usage 的请求消耗为未知。不估算金额，也不把
本地 tokenizer 估算当作真实消费。新增用量事件不记录提示词、回答或密钥；原有
工具日志仍可能包含工具参数和内容片段。

## 开发与验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests
```

配置在构造 Agent 之前完成校验；非法运行模式、非正数限制、未知路径配置项和错误的记忆路径会在启动时直接报告。应用生命周期使用 `try/finally` 语义，即使任务异常退出也会尽量保存文献库和长期记忆。

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
# 输出截断保护

模型返回 `stop_reason=max_tokens` 时，日志将该次调用记为 `incomplete`，
不会执行其中的工具调用，也不会把截断文本作为完整结果返回。
一般调用最多增加预算恢复一次（恢复预算上限 32768 token；原预算更大时保持原值），
仍不完整则明确报错。恢复调用共享 `recovery_id`，`attempt` 与
`requested_max_tokens` 记录尝试次数及预算；所有尝试的 token 都计入用量。
`failed_calls` 统计请求异常，`incomplete_calls` 单独统计截断，不代表业务验证通过。

引用检查每组最多 4 条，批量失败只对该组逐条检查；单条预算为 2048 token，
截断恢复耗尽后不再叠加重试。缺项、重复引用键和无效结果不会作为有效检查结果。
修改后重启生效；不会自动重跑或修改已有论文和日志。
