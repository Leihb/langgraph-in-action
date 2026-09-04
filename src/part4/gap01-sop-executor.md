# 按 SOP 一步步执行的坐席助手

> Part 4 的三个例子网上没有现成的开源实现。这一篇的原型是电信、电商客服后台里给坐席用的
> "流程执行助手"（Vodafone 的 Super Agent 由一个 supervisor 判断"走标准排障流程"还是"开放
> 问答"，步骤存在图数据库里）。开源社区只有问答型的知识库助手，没有"按 SOP 一步步执行、
> 每步调接口、审批被拒能回退"的例子。
> 用到的机制：第 2 期（图、`Command` 路由）、第 4 期（checkpointer）、第 5 期（interrupt）、
> 例子 1（json_mode 结构化输出）、例子 3（状态机，这一篇是它的反面）。

场景来自一个真实的客服后台，名字和数字改过。客人因为商户漏发接送、包车迟到这类事要
补偿，坐席在后台走一条固定流程：查订单、核对能不能补、填金额和方式、看支付网关还能
原路退多少、不够的要客人收款账户、按金额分档审批、提交、写订单备注。八步，每一步都
有明确的规则和接口，坐席手册上写得清清楚楚。这条流程每天走几百遍。

这本书前面两个客服 agent 都是"模型在循环里自己决定下一步调什么工具"（第 3 期开始的那
个，例子 3 的状态机把它收窄到每步两三个工具）。这一篇反过来：**流程写成一份数据文件，
一张通用的图按文件一步步执行，模型只在三个地方出场**——听懂坐席要干什么、从坐席的
回答里抽字段、最后写小结。哪一步调哪个接口、什么条件下停、金额多大要谁批，全部由代码
按文件决定，模型没有"选工具"的权力。

## SOP 长什么样

```yaml
name: compensation
title: 发起补偿
description: 客人因为服务问题（商户漏发、行程取消、体验差）要给现金或积分补偿时走这条流程
fields: [order_id, amount, reason, comp_type]

steps:
  - id: load_order
    kind: call
    tool: get_order
    args: {order_id: order_id}
    save_as: order

  - id: eligibility
    kind: check
    rules: [order_paid, not_in_fraud_review, no_open_compensation]

  - id: collect
    kind: ask
    fields: [amount, reason, comp_type]
    prompt: 补偿金额（USD）、补偿原因、补偿方式（cash 现金原路退 / credit 积分）

  - id: gateway_balance
    kind: call
    when: comp_type == 'cash'
    tool: get_gateway_balance
    args: {order_id: order_id}
    save_as: gateway_balance

  - id: bank_info
    kind: ask
    when: comp_type == 'cash' and amount > gateway_balance
    fields: [bank_account]
    prompt: 网关可原路退的余额不够，超出部分要手工转账，请提供客人的收款账户

  - id: approval
    kind: approve
    tiers:
      - {max: 50, level: auto}
      - {max: 500, level: supervisor}
      - {max: 2000, level: manager}
      - {level: director}
    on_reject: {goto: collect, reset: [amount, bank_account]}

  - id: submit
    kind: call
    tool: apply_compensation
    args: {order_id: order_id, amount: amount, comp_type: comp_type, reason: reason, bank_account: bank_account, approver: approver}
    save_as: compensation

  - id: note
    kind: call
    tool: add_booking_note
    args: {order_id: order_id, compensation: compensation}
    save_as: note
```

四种步骤：

| kind | 做什么 | 停不停 |
|---|---|---|
| `call` | 调一个内部接口，结果存进 `facts[save_as]` | 不停 |
| `check` | 跑几条代码规则，任一条不过就结束、告诉坐席为什么 | 不停 |
| `ask` | `facts` 里缺字段就停下来问坐席，回答由模型抽成字段 | interrupt |
| `approve` | 按金额分档：`auto` 直接过，其他档停下来等审批人，拒绝就回退 | interrupt |

`when` 是可选条件，不成立就跳过这一步。`facts` 是一路收集的事实（订单、金额、余额、
审批人……），`args` 里写的是 `facts` 的字段名。**改流程改这个文件，图一行不动。**

## 敲进去

代码在 `code/gap01_sop_executor/`：`sops/compensation.yaml`（上面那份）、`sops/policy.md`
（坐席手册，给开放问答用）、`sop.py`（读文件）、`tools.py`（假接口）、`rules.py`（check
的规则）、`prompts.py`（三段提示词）、`graph.py`（执行器）、`main.py`。

### 图

```
plan ──qa──▶ qa ──▶ END
  └──sop──▶ step ──▶ step ──▶ ... ──▶ finish ──▶ END
             ▲ (approve 被拒 → 回退到指定步骤)
```

四个节点，三个用模型（`plan` / `qa` / `finish`），一个不用（`step`）。`step` 每次执行 SOP
里的一步，用 `Command(goto="step")` 把自己接回来，走完了去 `finish`。

```python
class SopState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    sop: str | None
    cursor: int                                  # 下一步的下标
    facts: dict                                  # 一路收集的事实
    trail: Annotated[list[str], operator.add]    # 执行记录，一步一行
    outcome: str | None                          # done / stopped / None
    model_calls: Annotated[int, operator.add]
```

### 模型出场一：听懂坐席要干什么

```python
def plan(state: SopState) -> Command[Literal["step", "qa"]]:
    text = state["messages"][-1].content
    sop_list = "\n".join(f"- {s['name']}：{s['description']}（字段：{', '.join(s['fields'])}）" for s in SOPS.values())
    raw = json_llm.invoke(PLAN.format(sops=sop_list, text=text))
    facts = {k: v for k, v in (raw.get("facts") or {}).items() if v not in (None, "", "null")}
    if raw.get("mode") == "sop" and raw.get("sop") in SOPS:
        return Command(goto="step", update={"mode": "sop", "sop": raw["sop"], "cursor": 0, "facts": facts, ...})
    return Command(goto="qa", update={"mode": "qa", ...})
```

坐席说一句话，模型判断：走哪条 SOP，还是在问问题；顺带把话里已有的字段抽出来（订单号、
金额、原因、方式）。这是 Vodafone 那个 supervisor 的位置。`json_llm` 是例子 1 的结论——
这台端点只有 `json_mode` 走得通——提示词里明确"没提到的一律 null，不要猜"。

### 不用模型：执行一步

```python
def step(state: SopState) -> Command[Literal["step", "finish"]]:
    sop = SOPS[state["sop"]]
    i = state["cursor"]
    if i >= len(sop["steps"]):
        return Command(goto="finish", update={"outcome": "done"})
    s = sop["steps"][i]
    facts = dict(state["facts"])
    nxt = {"cursor": i + 1}

    if not when_holds(s, facts):
        return Command(goto="step", update={**nxt, "trail": [f"{s['id']}：条件 `{s['when']}` 不成立，跳过"]})

    if s["kind"] == "call":
        result = TOOLS[s["tool"]](**resolve_args(s, facts))
        facts[s["save_as"]] = result
        return Command(goto="step", update={**nxt, "facts": facts, "trail": [...]})

    if s["kind"] == "check":
        for name in s["rules"]:
            ok, why = RULES[name](facts)
            if not ok:
                return Command(goto="finish", update={"outcome": "stopped", "trail": [f"{s['id']}：规则 {name} 不过——{why}"]})
        return Command(goto="step", update={**nxt, "trail": [...]})
```

`call` 和 `check` 一次模型都不调。`TOOLS` 是一个普通字典，值是普通函数——它们不是
LangChain 工具，模型看不见。`RULES` 同理：

```python
def order_paid(facts: dict) -> tuple[bool, str]:
    o = facts["order"]
    return o["payment_status"] == "paid", f"订单 {o['order_id']} 未支付，不能发起补偿"
```

给没付钱的订单打钱是会出事的，这种判断写成代码，不让模型"参考政策自己判断"。

### 停下来问：ask

```python
    if s["kind"] == "ask":
        missing = [f for f in s["fields"] if facts.get(f) in (None, "")]
        if not missing:
            return Command(goto="step", update={**nxt, "trail": [f"{s['id']}：字段齐了，不用问"]})
        answer = interrupt({"kind": "ask", "step": s["id"], "missing": missing, "prompt": s["prompt"]})
        raw = json_llm.invoke(EXTRACT.format(prompt=s["prompt"], text=answer, fields=", ".join(missing)))
        got = _clean_fields(raw, missing)
        facts.update(got)
        still = [f for f in missing if f not in got]
        return Command(goto="step", update={"facts": facts, "model_calls": 1, **({} if still else nxt), "trail": [...]})
```

`interrupt()` 之前只有纯读取——第 5 期说过恢复时节点从头重跑，这一行之前不能有副作用。
坐席的回答是一句自由文本（"300 美元，退回卡里"），这是模型第二次出场：把它抽成
`{"amount": 300, "comp_type": "cash"}`。`_clean_fields` 在代码里校验类型和取值。没抽全就
留在这一步（`cursor` 不动），下一轮接着问。

### 停下来等批：approve，以及回退

```python
    if s["kind"] == "approve":
        amount = float(facts["amount"])
        tier = next(t for t in s["tiers"] if amount <= t.get("max", float("inf")))
        if tier["level"] == "auto":
            facts["approver"] = "System"
            return Command(goto="step", update={**nxt, "facts": facts, "trail": [...]})
        decision = interrupt({"kind": "approve", "step": s["id"], "level": tier["level"], "summary": {...}})
        if str(decision).strip().lower().startswith("approve"):
            facts["approver"] = tier["level"]
            return Command(goto="step", update={**nxt, "facts": facts, "trail": [...]})
        back = s["on_reject"]
        for f in back.get("reset", []):
            facts.pop(f, None)
        return Command(goto="step", update={"cursor": step_index(sop, back["goto"]), "facts": facts, "trail": [...]})
```

分档是代码算的。自动档不停；其他档 `interrupt`，等审批人一句 `approve` 或 `reject 理由`。
拒绝了，`cursor` 拨回 SOP 文件里 `on_reject.goto` 指的那一步，把 `reset` 列的字段清掉——
金额清了，`collect` 会重新问金额，原因和方式还在不用再问。**回退是改一个下标**，没有
别的状态要收拾，因为每一步的产出都在 `facts` 里，且每一步只在自己的 `step` 执行里产生
副作用。

### 一步一次执行，是为了 interrupt

为什么不写成一个 `for` 循环把八步跑完？因为 `interrupt()` 恢复时是整个节点从头重跑。
如果八步在一个节点里，`bank_info` 那一步停下再恢复，前面的 `get_order` 会再调一次，`submit`
如果排在前面会再提交一次。一步一个节点执行，`interrupt` 只出现在 `ask`/`approve` 的第一行，
`call` 步骤永远在自己的那次执行里跑且只跑一次。

## 跑起来

```bash
cd code
uv run python -m gap01_sop_executor.main t1 "给 KL-778 补 80 美元现金，商户漏发接送"
uv run python -m gap01_sop_executor.main t1 "中国银行 6222 0000 1234，户名王女士"   # 回答上一步的提问
uv run python -m gap01_sop_executor.main t1 approve                                  # 审批人批准（或 "reject 理由"）
uv run python -m gap01_sop_executor.main --state t1                                  # 看 facts 和执行记录
```

thread 停在 interrupt 上时，发来的消息自动当作回答（例子 7 服务里的那个判断）。每一轮
打印执行记录、停在哪、最后打印这一轮调了几次模型。

## 你应该看到什么

### 一条完整的：余额不够、要账户、主管批

```
[t1] 坐席：给 KL-778 补 80 美元现金，商户漏发接送
  ── 新任务：给 KL-778 补 80 美元现金，商户漏发接送
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-778', 'amount': 80, 'reason': '商户漏发接送', 'comp_type': 'cash'}
  load_order：get_order → {'order_id': 'KL-778', 'customer': '王女士', 'product': '东京迪士尼一日票 x2', 'amount': 320, ..., 'payment_status': 'paid', 'fraud_status': 'pass'}
  eligibility：3 条规则全过
  collect：字段齐了，不用问
  gateway_balance：get_gateway_balance → 60
  ⏸ 问坐席：网关可原路退的余额不够，超出部分要手工转账，请提供客人的收款账户（缺 ['bank_account']）
  （这轮模型调用 1 次，累计 1）

[t1] 坐席：中国银行 6222 0000 1234，户名王女士   （作为对上一步提问的回答）
  bank_info：坐席答「中国银行 6222 0000 1234，户名王女士」→ 抽出 {'bank_account': '中国银行 6222 0000 1234，户名王女士'}
  ⏸ 等 supervisor 审批：{'order_id': 'KL-778', 'amount': 80, 'comp_type': 'cash', 'reason': '商户漏发接送', 'bank_account': '中国银行 6222 0000 1234，户名王女士'}
  （这轮模型调用 1 次，累计 2）

[t1] 坐席：approve   （作为对上一步提问的回答）
  approval：supervisor 审批通过
  submit：apply_compensation → {'compensation_no': 'RA260904153247', ..., 'resources': [{'kind': 'gateway', 'amount': 60}, {'kind': 'manual_transfer', 'amount': 20, 'account': '中国银行 6222 0000 1234，户名王女士'}]}
  note：add_booking_note → 补偿单 RA260904153247：cash 80 USD，原因：商户漏发接送，审批：supervisor
  助手：已为订单 KL-778（王女士）发起 80 美元现金补偿，原因“商户漏发接送”，主管审批已通过，补偿单 RA260904153247 已生成并写入订单备注。因网关余额不足，其中 60 美元走网关，剩余 20 美元需人工转账至王女士提供的中国银行账户（6222 0000 1234）。请坐席尽快安排完成这笔 20 美元的人工转账。
  （这轮模型调用 1 次，累计 3）
```

坐席一句话，plan 一次调用把四个字段全抽出来，`collect` 不用问。三轮一共调了三次模型：
plan、抽账户、写小结。八个步骤里的接口调用、规则、分档、拆"60 走网关 + 20 手工"，都是代码。

### 拦下来的两条

```
[t2] 坐席：KL-901 客人投诉，补 20 积分
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-901', 'amount': 20, 'reason': '客人投诉', 'comp_type': 'credit'}
  load_order：get_order → {..., 'payment_status': 'unpaid', ...}
  eligibility：规则 order_paid 不过——订单 KL-901 未支付，不能发起补偿
  助手：系统尝试为订单 KL-901 发起 20 积分补偿，但资格校验未通过：该订单尚未支付……请坐席先核实订单实际支付状态，再与客人沟通并决定后续如何处理。

[t7] 坐席：KL-315 补 40 美元
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-315', 'amount': 40}
  eligibility：规则 no_open_compensation 不过——订单 KL-315 已有待审批的补偿单 RA2609010001（30 USD），先处理它
  助手：……该订单已存在一张待审批的补偿单 RA2609010001，金额 30 USD，因此不能直接再发起新补偿。……请先处理这笔待审批的 30 美元补偿单，处理完后再判断是否需要补差额。
```

两条都在第二步停下，两次模型调用（plan 和小结），没有任何一次是模型"决定"停的。

### 缺字段、被拒、回退、再批

```
[t4] 坐席：给 KL-502 补偿一下，客人包车迟到两小时
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-502', 'reason': '客人包车迟到两小时'}
  load_order：get_order → {..., 'amount': 1800, ...}
  eligibility：3 条规则全过
  ⏸ 问坐席：补偿金额（USD）、补偿原因、补偿方式（cash 现金原路退 / credit 积分）（缺 ['amount', 'comp_type']）

[t4] 坐席：300 美元，退回卡里   （作为对上一步提问的回答）
  collect：坐席答「300 美元，退回卡里」→ 抽出 {'amount': 300, 'comp_type': 'cash'}
  gateway_balance：get_gateway_balance → 1800
  bank_info：条件 `comp_type == 'cash' and amount > gateway_balance` 不成立，跳过
  ⏸ 等 supervisor 审批：{'order_id': 'KL-502', 'amount': 300, 'comp_type': 'cash', 'reason': '客人包车迟到两小时', 'bank_account': None}

[t4] 坐席：reject 迟到两小时按政策最多补 150   （作为对上一步提问的回答）
  approval：supervisor 拒绝——reject 迟到两小时按政策最多补 150。回退到 collect，清掉 ['amount', 'bank_account']
  ⏸ 问坐席：补偿金额（USD）、补偿原因、补偿方式（cash 现金原路退 / credit 积分）（缺 ['amount']）
  （这轮模型调用 0 次，累计 2）

[t4] 坐席：那就 150   （作为对上一步提问的回答）
  collect：坐席答「那就 150」→ 抽出 {'amount': 150}
  gateway_balance：get_gateway_balance → 1800
  bank_info：条件 `comp_type == 'cash' and amount > gateway_balance` 不成立，跳过
  ⏸ 等 supervisor 审批：{'order_id': 'KL-502', 'amount': 150, ...}

[t4] 坐席：approve   （作为对上一步提问的回答）
  approval：supervisor 审批通过
  submit：apply_compensation → {'compensation_no': 'RA260904153402', ..., 'amount': 150, ...}
  助手：已为 KL-502 完成补偿流程：原申请的 300 美元被主管驳回，按政策调整为 150 美元后审批通过。……
```

拒绝那一轮**零次模型调用**：拨回 `collect`，清掉金额，重新问，全是代码。问第二次时只缺
`amount`，原因和方式没丢。整条走了五轮、四次模型调用。

### 一次都不停的

```
[t6] 坐席：KL-778 补 30 积分，导游迟到
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-778', 'amount': 30, 'reason': '导游迟到', 'comp_type': 'credit'}
  eligibility：3 条规则全过
  collect：字段齐了，不用问
  gateway_balance：条件 `comp_type == 'cash'` 不成立，跳过
  bank_info：条件 `comp_type == 'cash' and amount > gateway_balance` 不成立，跳过
  approval：30 USD 在自动审批档，审批人 System
  submit：apply_compensation → {'compensation_no': 'RA260904153307', ..., 'approver': 'System', ...}
  note：add_booking_note → 补偿单 RA260904153307：credit 30 USD，原因：导游迟到，审批：System
  助手：订单 KL-778 的 30 credit 补偿已通过系统自动审批并提交……坐席当前无需立即执行其他操作。
  （这轮模型调用 2 次，累计 2）
```

积分补偿跳过两个只对现金有意义的步骤，30 美元在自动档，八步一口气走完，两次模型调用。

### 问问题走另一条路

```
[t5] 坐席：先补偿了还能退款吗
  ── 提问：先补偿了还能退款吗
  助手：不可以。政策明确：先补偿再退款不可以，退款那边会拦。
```

### 模型猜了一个字段

第一版 plan 的提示词只说"没提到的一律 null，不要猜"。坐席说"KL-315 补 40 美元"，模型抽出
`{'order_id': 'KL-315', 'amount': 40, 'comp_type': 'cash'}`——坐席没说现金还是积分，"美元"
被当成了"现金"。这条要是没被资格规则拦住，`collect` 会看到字段齐了不问，一路走到提交。
补了一句"只说了 XX 美元不算说了方式，comp_type 填 null"，同一句话再跑，抽出的是
`{'order_id': 'KL-315', 'amount': 40}`；换一个能过资格的订单：

```
[t8] 坐席：KL-778 补 40 美元，商户漏发
  plan：走 SOP「发起补偿」，已知 {'order_id': 'KL-778', 'amount': 40, 'reason': '商户漏发'}
  ⏸ 问坐席：补偿金额（USD）、补偿原因、补偿方式（cash 现金原路退 / credit 积分）（缺 ['comp_type']）
[t8] 坐席：积分
  collect：坐席答「积分」→ 抽出 {'comp_type': 'credit'}
```

"美元→现金"是个合理的联想，但在这条流程里它决定钱从哪个口子出去，联想不算说了。**模型
出场的每个位置，它填的每个字段，都要想一遍"它会不会替坐席做决定"。**

## 发生了什么

**SOP 是数据，图是解释器。** 例子 3 的状态机把三步写进提示词和工具里，换一条流程要改
代码。这一篇八步写在一个 YAML 文件里，`step` 节点是一个按 `kind` 分发的解释器。再加一条
SOP（改单、退款）是再写一个文件，图不动，`plan` 的 SOP 列表自动多一行。真实客服后台有几十
上百条 SOP，"每条一个 agent"撑不住，"一个解释器 + N 个文件"撑得住。

**决定由代码做，模型只做翻译。** 八步里模型出场的位置：把坐席的自由文本翻成字段（两处），
把执行记录翻成给坐席看的话（一处）。能不能补、余额够不够、谁来批、拆多少走网关，这些
决定错了会出事，全部是代码。ep01 那段"LangGraph 的优势是更多确定性和更快更省"，这一篇
是它最直接的例子：t6 那条八步两次模型调用，t4 拒绝回退那轮零次。

**interrupt 决定了节点的粒度。** 一步一次节点执行，不是为了好看，是为了 `interrupt()`
"恢复时从头重跑"这条规则下，`call` 步骤永远只跑一次。第 5 期的 `cancel_order` 在一个节点
里，这一篇八步都要这个保证，所以粒度收到了"一步"。

**回退是拨下标，因为状态都在 facts 里。** 每一步的产出写进 `facts`，`cursor` 指向下一步。
审批拒绝了，拨回去、清几个字段，中间的步骤会按新字段重新执行（`gateway_balance` 又查了
一次，结果一样）。要是状态散在各个节点的局部变量里，回退就得写专门的清理代码。

**模型的每一次"不猜"都要写进提示词里验一遍。** "没提到的一律 null"不够，"美元"被理解成
"现金"是模型替坐席补了一个决定。这种事只能靠跑出来发现，第 15 期评测里"该问的问、不该
问的不问"这类正反成对的用例就是为它准备的。

## 常见问题

**这跟例子 3 的状态机有什么区别？** 例子 3 每一步是"模型在两三个工具里选"，步子怎么走
写在工具里；这一篇每一步"调哪个接口"写在文件里，模型不选。例子 3 适合"步骤固定但每步
里客人说什么都有可能"的对话；这一篇适合"坐席知道自己要做什么、要的是把八个动作按规矩
做对"的后台操作。两者都是状态机，自由度差一档。

**`when` 用 `eval`，安全吗？** 这里是演示——表达式只能看 `facts`，`__builtins__` 清空了——
但 SOP 文件是谁都能改的，真实系统应该换成白名单的比较运算，或者一个小的规则引擎。
文件里的 `when` 总共两种写法（等于、大于），够用了。

**坐席的回答抽错了怎么办？** `_clean_fields` 校验类型和取值，抽不出来就留在这一步再问；
抽出来但抽错了（"三百"抽成 30），这一篇挡不住。修法是 `approve` 前加一步 `confirm`：把
`facts` 打给坐席看一眼再往下走——加分练习 2。

**为什么小结要用模型？执行记录已经有了。** 可以不用，`trail` 直接打出来就行。用模型是
因为坐席要的是"接下来我要做什么"（t1 小结里的"请安排 20 美元人工转账"），这一句从记录
里推出来比模板拼出来自然。这是三次出场里最可有可无的一次，成本敏感就去掉。

**SOP 文件谁来写？** 坐席手册的作者。这份 YAML 的八步跟手册上的八步一一对应，`rules` 和
`tools` 的名字是开发给的词汇表，剩下的顺序、条件、分档是业务定的。前言里说"写代码的人换
了，差别在会不会把活说清楚"，这个文件就是把活说清楚的那份说明书。

## 加分练习

1. 再写一条 SOP `sops/refund.yaml`（退款：查订单→查取消政策→算可退金额→审批→提交），
   不改 `graph.py`。`plan` 要能分清"补偿"和"退款"。
2. 在 `approval` 前加一步 `kind: confirm`：把 `facts` 里的关键字段打给坐席确认，回答"对"
   继续，回答别的重新走 `collect`。`step` 里加一个分支。
3. 把 `when` 的 `eval` 换成白名单：只允许 `字段 运算符 字段/常量`，用 `operator` 模块查表。
4. 用第 15 期的方法给 `plan` 写十条用例：五条该抽出 `comp_type`，五条不该。跑一遍，看
   "美元→现金"这类猜测还有没有别的变体。
5. 把这张图挂进例子 7 的服务：interrupt 的两种 payload（`ask` / `approve`）前端要分别渲染
   成输入框和两个按钮。
