# 给 AI 读的说明

你（Claude Code、Codex 或别的编码 agent）正在读一个教学仓库。它的读者会 fork 之后让你"改成我的场景"。
下面是你需要知道的全部约定。

## 这个仓库是什么

- 正文在 `src/`，mdBook 格式，中文。每一期一个文件，对应 `code/` 下同名目录里的可运行代码。
- 代码只用 Python 3.11+，框架是 LangGraph 1.2.x 和 langchain 1.3.x，版本锁在 `code/uv.lock`。**不要升级版本**，教程正文是按这些版本写的。
- 模型接入走一个兼容 OpenAI 协议的端点，用三个环境变量：`MODEL_BASE_URL`、`MODEL_API_KEY`、`MODEL_NAME`。读者本机通常是一个 litellm 网关，默认 `http://localhost:4477/v1`，模型别名 `chat-default`。代码里不要写死任何供应商。
- 观测走 Langfuse，环境变量 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`。没配就静默不上报，不能因为缺观测而跑不起来。
- 每一期一条命令能跑：`cd code && uv run python -m <期目录>.main`，假数据都在各期目录的 `data/` 里。

## 改成读者自己的场景时，改哪里

1. **工具**：各期 `tools.py`。换成读者自己系统的函数，保持一个工具一个函数、docstring 写清楚给模型看。
2. **系统提示词**：各期 `prompts.py`。
3. **状态定义**：各期 `state.py`，只加读者场景真正需要的字段，别顺手加"将来可能用"的。
4. **图的结构**：各期 `graph.py`。改之前先问读者三层判断的结果：单 prompt、预定义流程、还是模型自己决定下一步。答案不是最后一种就不要加 agent 循环。
5. **假数据**：`data/`，换成读者给的真实输入样例。

## 别碰的地方

- `code/common/`：模型客户端、Langfuse 接入、环境变量读取。所有期共用。
- `code/pyproject.toml` 的版本约束。
- `src/` 正文，除非读者明确要改教程本身。

## 写代码时的纪律

- 不要用 `langgraph.prebuilt.create_react_agent`，它已弃用。要现成 agent 用 `langchain.agents.create_agent`。
- 人工确认用节点内 `interrupt()` 加 `Command(resume=...)`，不用编译期 `interrupt_before`。interrupt 之前的代码在恢复时会整段重跑，副作用要幂等。
- 取运行结果用 `invoke(..., version="v2")`，不要按字典键取 `__interrupt__`。
- 任何会改外部系统的工具，默认要过一次人工确认，除非读者明确说不要。
- 不要给 agent 直接执行 shell 的工具。教程里没有，读者的场景多半也不需要，需要的话先问。

## 正文写作约定（只在被要求改教程时适用）

- 先描述行为再给术语的正式名字，不自造译名。
- 每一期必须有"跑起来"一节和真机跑出来的输出，不写没跑过的结果。
- 只写发现和结论，不写"试了 A 又试了 B"的调试过程。

- 流程图直接写 ```mermaid 代码块，站点会渲染（theme/mermaid-init.js），不用贴图片。
