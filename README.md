# 码猫 CodeCat

> 面向中文开发者的轻量编程助手:**定制技能包**(代码审查 / 测试生成 / 中文文档)+ **定时自动评审报告** + **可回溯的会话分支**。

**基于 [nanobot](https://github.com/HKUDS/nanobot) 构建** —— 使用其 Python SDK 作为 agent 运行时底座,CLI/TUI、技能包与自动化工作流为自研部分。

## 状态

🚧 开发中(第一个里程碑:一周 MVP,2026-11-24 起)

- 需求文档:[REQ.md](./REQ.md)

## 快速开始

```bash
# 1. 环境(Windows / Python 3.11+)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .

# 2. 配置 DeepSeek API key
copy .env.example .env   # 填入 platform.deepseek.com 的 key

# 3. 验证环境
.venv/Scripts/python.exe scripts/demo.py "你好"
```

## Roadmap

- [x] 环境搭建 + nanobot SDK 跑通
- [ ] CLI/TUI(rich)
- [ ] 定制技能包:code-review / test-gen / doc-gen
- [ ] 定时自动任务:每日评审报告
- [ ] 会话分支/回退(轻量版:fork / tree / switch)
- [ ] Terminal-Bench 类基准评测

## 致谢

- [nanobot](https://github.com/HKUDS/nanobot)(HKUDS,MIT)—— agent 运行时底座
- [my-pi-agent](https://github.com/zxj-2023/my-pi-agent)—— 设计灵感来源(树状会话、上下文压缩、工程规范)
