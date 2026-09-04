# 笨办法学 Agent · 用 LangGraph 上线

> 仓库名 langgraph-in-action。

**任何人都可以写 agent。这个仓库教你把场景说清楚，剩下的交给 LangGraph 和你的 AI。**

这是《笨办法学 Agent》系列的第二本，用 LangGraph 把真实场景的 agent 做出来、放到线上
给人用，目的是上线。读过第一本的话，正文会不断指回对应的练习，每次指回都会顺带说清楚
那个练习解决的是什么问题，比如 checkpointer 对应第一本练习 11 到 13 的会话文件加压缩、
interrupt 对应练习 9 到 10 的权限闸门；没读过第一本也不影响跟着这本往下走。

## 这套书一共三本

《笨办法学 Agent》系列现在有三本，各自独立，不要求先后顺序：

- **[笨办法学 Agent · 亲手打造一个 harness](https://github.com/Leihb/learn-agent-the-hard-way)**——
  不用任何框架，32 个练习亲手写出一个 agent harness 的每一层，目的是看懂。
- **[笨办法学 Agent · 用 LangGraph 上线](https://github.com/Leihb/langgraph-in-action)（这一本）**——
  用 LangGraph 把真实场景的 agent 做出来、放到线上给人用，目的是上线。
- **[让 agent 替你干活 · 不写代码，用 octo 把活干完](https://github.com/Leihb/octo-at-work)**——
  不写一行代码，用装在自己电脑上的 octo 把日常的活干完，给不写代码的打工人。

三本共享同一句话：agent 没有秘密架构，会不会用，看你会不会把活/场景/工具边界说清楚。

## 📖 在线阅读

**https://leihb.github.io/langgraph-in-action/**

内容分五部分：

1. **基础**：三层判断、第一张图、工具与条件边、checkpointer、interrupt、Store、MCP、检索、skill、子图、观测。11 期。
2. **部署**：FastAPI 包一层、换真实存储与长任务、放到线上给别人用、评测。4 期。
3. **经典例子重做**：官方那批教程大半已归档或用了弃用 API，用 2026 年的 API 重做 7 个。
4. **企业空白**：企业案例里反复出现、但没有干净开源示例的 3 个场景，原创实现。
5. **读者场景**：评论区交场景，按模板筛选后实现，每期回到原评论下点名。

每一期的代码在 [`code/`](code/)，只依赖一个兼容 OpenAI 协议的模型端点和 Python 3.11+，
假数据全在仓库里，一条命令能跑。版本锁死在 `code/uv.lock`。

**这个仓库也是写给你的 AI 读的。** fork 之后对你的 Claude Code 或 Codex 说"改成我的场景"，
它会先读 [`AGENTS.md`](AGENTS.md)，那里写了哪些地方该改、哪些地方别碰。

## 目录与进度

| 部分 | 期 | 状态 |
|---|---|---|
| 前言 | 任何人都可以写 agent | ✅ 已发布 |
| Part 1 · 基础 | 第 1 到 11 期 | ✅ 已发布 |
| Part 2 · 部署 | 第 12 到 15 期 | ✅ 已发布 |
| Part 3 · 经典例子重做 | 7 个 | 🚧 2/7 |
| Part 4 · 企业空白 | 3 个 | 待开始 |
| Part 5 · 读者场景 | 持续 | 待开始 |

## 本地阅读

```bash
mdbook serve
```

## 许可

正文 CC BY-NC-SA 4.0，代码 MIT。见 [LICENSE](LICENSE)。
