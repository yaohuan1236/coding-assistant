# 智能编程助手 — 需求文档(草稿 v0.1)

> 本文档记录讨论中已锁定的决策与待定问题。每聊定一项,就更新一次。

## 1. 定位

**一句话**:面向中文开发者的轻量编程助手:定制技能包(代码审查/测试生成/中文文档)+ 定时自动评审报告 + 可回溯的会话分支。

## 2. 目标与约束

- **目的**:简历项目(求职 Agent 开发岗),要求能讲清自己的贡献边界
- **开源态度**:README 诚实标注"基于 nanobot 构建"(选型能力是加分项)
- **里程碑**:一周 MVP(2026-11-24 起,每天 3~4 小时),之后持续迭代
- **验证方式**:MVP 阶段用真实用例验证(以外卖平台 Java 项目为靶子,录演示);Terminal-Bench 类基准评测放 roadmap

## 3. 技术底座

- **完全自研核心**(2026-11-24 晚间决策,由"nanobot SDK 底座"转向):LLM 客户端、ReAct 主循环、工具、会话存储全部自己实现,**核心零框架依赖**(仅 rich/python-dotenv/httpx 这类基础设施库)
- **nanobot 定位变化**:`scripts/demo.py` 保留为对比素材,nanobot-ai 移到可选依赖 `[compare]`;设计灵感来源 = nanobot + my-pi-agent,记录于 REFERENCES.md
- **语言/环境**:Python 3.11+,asyncio + httpx,Windows 11 开发
- **模型**:DeepSeek(默认,MVP 期间钉死一个,减少变量)
- **亮点来源**:从 my-pi-agent 汲取(树状会话、上下文压缩、Never-Throw、工具并发调度、事件驱动、TDD),实现时逐个取舍;从 nanobot 汲取(Loop/Runner 职责切分、SKILL.md 格式、工具 schema 风格)

## 3.1 设计原则(锁定)

1. **最小里程碑优先,核心框架一次搭稳**:模块分层与接口第一天就定对(LLM 客户端 / 消息上下文 / 工具注册表 / 主循环 / 会话存储,每层一模块一职责);第 1 版实现从简,高级功能(并发、压缩、多 provider)通过预留接口逐步加入,不推倒重写
2. **借鉴设计思路 + 自研实现**:借鉴的是设计决策与交互范式,核心代码自己写;用户逐行理解每一处实现
3. **工程规范**:核心模块(llm/loop/session)配离线单元测试(mock LLM),参考 my-pi-agent 的 TDD 风格
4. **Never-Throw 原则**:工具 handler 异常一律转为语义化结果回填模型,不向循环抛异常

## 4. 交付形态

- **自研 CLI/TUI** —— 界面是自研贡献的主战场之一
- 目标:简洁但有质感的终端体验;不追求 1:1 像素级复刻 Pi-TUI,但交互质感要能撑起面试演示

## 5. 核心卖点与优先级

**保底线(第一周必须完成)**:

- **P1 定制技能包**:代码审查 / 测试生成 / 中文文档(具体定义待细化)
- **P2 定时自动任务**:每日定时代码审查报告(目标仓库、调度方式、推送渠道待定)

**冲刺线(做不完进 roadmap)**:

- **P3 会话分支/回退(轻量版)**:`/fork`、`/tree`、分支切换(基于 session_key 自研树状管理;不含 my-pi-agent 的 compaction 护栏)

## 6. 待定问题(open questions)

- [ ] 项目命名(暂定 码猫/CodeCat,用户未最终确认)
- [x] 技能包语言范围 → **语言通用**,Java 外卖项目仅作验证靶子
- [x] 技能数量 → 三个:code-review / test-gen / doc-gen,不增删
- [ ] 自动任务的调度方式与推送渠道
- [ ] CLI 命令集与交互设计
- [x] 分支/回退设计 → 自研后成为自然扩展:会话树数据结构从第一天预留(树节点=会话快照),第 1 版单链即可
- [x] ToolRegistry 契约位置 → 第 1 版临时定义在 loop.py;实现 core/tools/registry.py 时迁移为该处唯一权威定义,loop 改为从那里导入(2026-11-25 定)
- [ ] 面试叙事与 README 结构
- [ ] Tool schema 获取时机:loop 目前每 Step 调一次 schemas();真实 registry 落地后按静态/动态决定——静态优先 registry 内部缓存(或在 run 开头快照),动态保持每 Step 调用

## 7. 进度记录

| 日期 | 进展 |
|---|---|
| 2026-11-24 | 需求讨论定大方向;建目录与 git 仓库;Python 3.13.9 环境确认;项目骨架(pyproject/setuptools、src 布局、config/config.json、scripts/demo.py);`.venv` 安装 nanobot-ai 0.4.2;SDK 端到端验证通过(deepseek-flash 工具调用链,19 个工具注册,hello.txt 落盘);首次提交 a1c2847;`.env` 存真实 key 且已 gitignore;推送公开仓库 https://github.com/yaohuan1236/coding-assitant(用户手动建仓,名字拼写 assitant,可后续改名) |
| 2026-11-25 | 自研核心第一层:core/types.py(数据契约)、core/llm.py(httpx 直连 OpenAI 兼容接口、config.json+.env 加载、双向转换、明确异常)、loop.py(最小 ReAct 循环、注入式工具入口、max_steps)、tests/test_loop.py(6 个离线测试全过)、scripts/smoke_llm.py(真实 DeepSeek 冒烟通过);异步实现(asyncio+httpx,按第 3 节锁定);config.json 增 baseUrl;pytest 入 dev 依赖 |

