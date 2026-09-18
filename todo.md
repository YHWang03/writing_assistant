# 待办事项

> 2026-09-17 评审意见已并入各项。标记 ❓ 的为待确认决策点，直接在其下写答案；未标记的建议项如不同意也可直接改写。

## 0. 压缩配对 bug（新增，最高优先级）

compress_messages 直接切 `messages[-keep_recent:]`，未保护 assistant(tool_use) / user(tool_result) 配对，keep_recent=6 时约一半概率产生孤立 tool_result 开头（前面接摘要 user 消息），API 会直接拒绝。compress_history 无此问题（历史中工具调用已转纯文本）。

修法：借鉴 s08 的切点回退（向前回退避开孤立 tool_result）。


## 1. 上下文压缩

原需求：参考 D:\Code\agent_learn\stage3\learn-claude-code\s08_context_compact 中的代码设计压缩上下文功能

评审意见：
- 现状已是"阈值到→LLM 摘要"，s08 的核心是先做零成本确定性操作，摘要只是最后手段
- 建议吸收（按收益排序）：
  1. 配对 bug 修复（见第 0 条）
  2. 旧 tool_result 替换为"文件路径指针"（已被消费过的 read_file/parse_pdf 结果替换为路径，需要时重读）——论文场景收益最大，累积大头就是旧 .tex/.pdf 内容
  3. reactive_compact：API 返回 prompt_too_long 时保留最近 5 条 + 摘要重试一次
- 不建议照搬：字符数触发（保留现有 BPE + API usage anchor 计数）、compact 主动工具、.transcripts 归档

❓1. 确认改造范围 = 配对修复 + 路径指针 + reactive 补救，不做 compact 工具 / transcript？

答：同意

## 2. 记忆系统

原需求：参考 D:\Code\agent_learn\stage3\learn-claude-code\s09_memory 中的代码设计记忆系统。 当前项目下 '记忆功能' 做成了tools, 由agent自行决定是否读取等。 应该参考claude中的实现, 在每一轮loop之前, 根据query查询相关联记忆, 拼接进上下文，在完整的loop结束之后，再调用llm提取事实/用户倾向的信息存入长期记忆中。 不再由agent自主决定。

整体memory只留一个长期记忆类型就可以了, 不需要区分情景记忆和事实记忆等等。
每一个agent有自己的记忆系统, 不需要全局共享。(例如master agent和writing agent各自有一个自己的memory.json, 两者不同)

提取时机: 在run()入口处召回一次, 不要在每一个react step都召回

评审意见：
- 同意 harness 层自动召回/提取；按 agent 分存储 → ReviewAgent 泄密问题随之解决
- 提取出口：_run_react 有三个结束出口（finish / text_only 强制 / max_steps），统一在出口提取，正好对应 hooks 的 Stop 事件
- 提取用 flash 模型（get_tool_llm 单例，max_tokens≈1000），成本低
- type 内容标签建议保留：s09 的分类有两个维度——架构层（episodic/long_term/working，删）和内容层 type 标签（user/feedback/project/reference），后者用于召回时 LLM 读目录选记忆，建议保留
- 存储格式：s09 用 md + YAML frontmatter，项目无 PyYAML 依赖。记忆文件不需要人工长文编辑，建议纯 JSON
- 整理（consolidate）：≥10 条触发、30 条上限、失败回滚快照，照搬 s09
- 删除面清单：memory_tool.py、6 个 Agent 的 MemoryTool 注册与 _sync_context_to_tools 注入、working.py（死代码）、episodic.py、long_term.py、manager.py、config 的 memory 段、main.py 的 _build_agent_memory

❓2. type 内容标签（user/feedback/project/reference）保留，只删架构层，对吗？

答：可以

❓3. 存储格式用纯 JSON（不加 PyYAML），可以吗？

答：可以

补充（已证实）：_run_react 三个结束出口各调一次 _save_to_memory（L422/L504/L550），plan_execute 结束又调一次（run_modes L88）——步骤片段会污染记忆。新设计中删除所有 _save_to_memory 调用，统一由 Stop hook（每次 run() 触发一次）负责提取，此问题顺带消除。

## 3. 搜索/下载tex模板

原需求：完全取消此功能，删除相关程序。 采用user提供的tex模板，若未提供，则使用默认模板。

评审意见：
- SearchTemplateTool / DownloadTemplateTool 本就是假实现（只返回"请手动下载"），删除零功能损失
- ValidateTemplateTool 是真实现，建议保留——用户提供的模板更需校验；但校验标准要从"必须有 .cls/.sty"改为"存在可编译的主 .tex"（现有 example/journal_tex 只有 template.tex，按现逻辑会被判 invalid）
- 默认模板目前不存在：resources/ 下只有 ds-v4 词表，"未提供则用默认"无法落地，需先备一个

❓4. 默认模板：你自己提供，还是我从 example/journal_tex/template.tex 整理一个内置到 resources/templates/default/？

答：创建一个默认模板, 放到 resources/templates/default/ 目录下, 直接把 example/journal_tex/template.tex 复制过去就行

## 4. 取消run_modes

原需求：删除run_modes.py文件，将其功能直接写入agent.py文件中

最终决定：react 和 plan_execute 两种范式直接并入 agent.py，用 hooks 让代码简洁；hooks 里具体挂什么对各个具体 agent 类分别设置。LiteratureAgent 保持 plan_execute（config 不变）。

评审意见（并入实现要求）：
- reflection 范式删除（config 未使用），只保留 react + plan_execute 两种；PLAN_STEP_MAX_STEPS=6、MAX_PLAN_STEPS=5 两个常量随迁入 agent.py
- ⚠ 已证实：plan_execute 下产出闸门全死——deadline 提示（agent.py L277）和 finish 闸门校验（L391）都要求 require_finish=True，而 PlanExecuteMode 每步 require_finish=False（L78-81），结束后也不查 _output_satisfied
- 修复方案：plan_execute 所有步骤执行完后，若声明了 required_output_exts 且未达标 → 注入"补写产出"提示，追加一轮带闸门的修正 react（require_finish=True）。written_paths 由 PostToolUse hook 持续收集（与 hooks 设计天然契合）
- 已证实：_save_to_memory 重复调用（见第 2 节补充），删除后由 Stop hook 统一负责
- 连带清理：main.py 的 resolve_run_mode 相关简化（只认 react/plan_execute，未知值回退 react）、config 的 run_mode 注释更新（去掉 reflection）

## 5. hooks

原需求：添加hooks功能, 来将run_loop, react等写简洁一些, 参考D:\Code\agent_learn\stage3\learn-claude-code\s04_hooks 中的代码设计。
然后这个hooks可以在每一个具体的agent类中再分别实现

评审意见：
- 与 s04 的关键差异：不做全局 HOOKS 字典，做实例级注册表（每 Agent 实例一份，同 ToolRegistry），基类提供触发点 + 默认 hook，子类在 _setup 时注册自己的
- 事件定义：UserPromptSubmit（run() 入口，挂记忆召回）、PreToolUse / PostToolUse（工具前后，挂日志、written_paths 收集）、Stop（每次 run() 结束触发一次，挂记忆提取与收尾）
- ⚠ Stop 语义必须明确：是 run() 整体结束，不是每个 react step 结束——plan_execute 一个任务含 3~5 个子步骤，Stop 只在最后触发一次，否则记忆会被提取多次
- 不建议 hook 化：finish 闸门、deadline_nudge、text_only 强制结束——强控制流留在循环里可读性更好
- 建议与记忆系统重写合并为一步做：记忆天然需要"入口召回 + Stop 提取"两个挂点，单独先做 hooks 会把 _run_react 白动一遍

## 建议执行顺序

1. [x] 修 compress_messages 配对 bug（独立、小、紧急）— _safe_cut 向前回退切点，验证通过
2. [x] 删模板假工具 + 复制默认模板到 resources/templates/default/ + 校验标准改为"存在主 .tex"
3. [x] 删 run_modes.py，react + plan_execute 并入 agent.py，补上 plan_execute 的产出闸门（步骤后统一校验 + 修正轮）
4. [x] hooks 实例级重构 + 记忆系统重写（合并一步）— 15/15 冒烟测试通过；顺带修复 TEMPORARY_MEMORY_MARKERS 未接入校验
5. [x] 压缩三层增强（路径指针 + reactive 补救）— 27/27 冒烟测试通过

## 完成记录（2026-09-18）

- 全部 5 步已实施并验证，临时测试脚本已清理，最终导入检查通过
- 新增/重写文件：src/memory/store.py（AgentMemory）、resources/templates/default/template.tex
- 核心改动：src/core/agent.py（react/plan_execute 合一 + hooks + 闸门 + reactive 补救）、
  src/core/context_compress.py（_safe_cut + _stub_large_tool_results + reactive_compact）、
  src/tools/builtin/template.py（只留 ValidateTemplateTool）、main.py、config.yaml
- 已删除：src/core/run_modes.py、src/memory/{base,episodic,long_term,working,manager}.py、
  src/tools/builtin/memory_tool.py 及各 Agent 的 SearchTemplate/DownloadTemplate/MemoryTool 注册
- 遗留：README.md 的目录结构描述未更新（不影响代码）
