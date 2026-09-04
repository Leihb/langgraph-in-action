# 非结构化邮件到结构化工单

> 原型是物流公司 C.H. Robinson 的邮件建单（每天一万五千封运输邮件自动变成订单）和 Remote 的
> HR 数据迁移 agent（每个节点显式画出成功 / 失败 / 重试三条边）。开源只有官方的
> `extraction/retries` 笔记本讲"抽取失败把错误喂回去重试"，没有"读邮件→抽字段→校验→缺字段
> 回问客人→跟系统对不上转人工→同一订单归并"的完整例子。
> 用到的机制：例子 1（json_mode 结构化输出、`Command` 路由）、第 4 期（checkpointer，一条邮件线程
> 一个 thread）、第 5 期（interrupt）。

例子 1 读邮件是为了**回复**：分类、查资料、拟稿、发出去。这一篇读邮件是为了**落库**：客服系统
里每一个客人的问题都要变成一张工单——什么类型、哪个订单、要什么、多急、谁来管——坐席在工单
上干活，指标（多久解决、有没有超时）按工单算。邮件是自由文本，工单是十几个字段，中间那一步
今天大多是坐席手填。这一篇把它自动化，并且把"自动化之后什么时候该停下来"想清楚。

场景还是这本书的旅行客服。一个收件箱，九封邮件：改期、退款（没写订单号）、要赔偿（订单号
写成"KL 502"）、帮朋友问退款（发件人并非下单人）、客人对上一封的补充、表扬信、供应商发来的
团次取消通知、一封广告。

## 核心：抽出来的字段错了，错在哪一层

模型把邮件抽成字段，代码校验。校验出的问题分三类，走三条路：

| 类别 | 例子 | 谁能修 | 走哪条路 |
|---|---|---|---|
| **FORMAT** | 订单号少个横线、日期没写成 `YYYY-MM-DD`、类别不在枚举里 | 模型（它抽错了） | 把错误喂回去重抽，最多三次 |
| **MISSING** | 没写订单号、改期没说改到哪天 | 客人（邮件里就没有） | 回信问客人，这条线程等回复 |
| **MISMATCH** | 订单不存在、发件人并非下单人、改期日期已过 | 人（系统和邮件说的不一致） | 转坐席 |

这个三分是整篇的骨架。抽取失败分好几种：模型能修的不该去烦客人，客人没给的重抽一百次
也抽不出来，系统对不上的谁都不该自动决定。

```
read_email → extract → check ─┬─ FORMAT 且还有次数 → extract（带着错误重抽）
                 ▲             ├─ MISSING → ask_customer → END（等客人回信，同一线程下一封邮件接着跑）
                 │             ├─ MISMATCH → human_review（interrupt）→ file / discard
                 └─────────────┴─ 都没问题 → file（归并或新建工单，回执）→ END
```

## 敲进去

代码在 `code/gap02_email_to_ticket/`：`inbox.json`（九封邮件）、`orders.py`（订单假数据）、
`schema.py`（字段、校验、优先级）、`prompts.py`（一段提示词）、`tools.py`（工单库、发信）、
`graph.py`、`main.py`。

### 字段：模型抽的和代码填的分开

```python
class TicketDraft(TypedDict, total=False):
    ticket_type: str          # customer_demand / feedback / merchant_request / not_a_request
    category: str             # refund / amendment / cancellation / compensation / invoice / inquiry / praise / other
    order_id: str | None
    customer_name: str | None
    request: str
    target_date: str | None
    amount: float | None
    reason: str | None
    language: str
```

这九个是模型从邮件里抽的。工单上还有几个字段模型碰不到：`customer_email` 从邮件头取，
`priority` 和 `sla_hours` 代码算（退改类且出行不到 48 小时 → 紧急 12 小时；商户来的 → 高 24
小时；表扬 → 低 72 小时），`product` 从订单系统查。**模型评"这个多急"没有意义，出行日期减
收信日期才有意义。**

### 校验

```python
def validate(draft, sender, received_at) -> list[Problem]:
    problems = []
    if tt not in TICKET_TYPES: problems.append({"kind": "FORMAT", "field": "ticket_type", ...})
    if cat not in CATEGORIES:  problems.append({"kind": "FORMAT", "field": "category", ...})
    if tt in ("not_a_request", "feedback") or problems:
        return problems                      # 表扬不需要订单；枚举都错了先修枚举

    oid = draft.get("order_id")
    if oid is None:
        problems.append({"kind": "MISSING", "field": "order_id", "message": "邮件里没有订单号"})
    elif not ORDER_RE.match(str(oid)):
        problems.append({"kind": "FORMAT", "field": "order_id", "message": f"order_id 必须写成 KL-三位数字（如 KL-778），收到 {oid!r}"})
    else:
        order = get_order(oid)
        if order is None:
            problems.append({"kind": "MISMATCH", "field": "order_id", "message": f"系统里没有订单 {oid}"})
        elif tt == "customer_demand" and order["email"].lower() != sender.lower():
            problems.append({"kind": "MISMATCH", "field": "sender", "message": f"发件人 {sender} 不是订单 {oid} 的下单人（{order['email']}）"})
    ...
    for f in REQUIRED_FROM_CUSTOMER.get(cat, []):     # amendment 要 target_date，refund/compensation 要 reason
        if draft.get(f) in (None, ""):
            problems.append({"kind": "MISSING", "field": f, ...})
    return problems
```

每条问题带 `kind`，路由节点只看 `kind`。同一个字段 `order_id` 能出三类问题：没有（MISSING）、
格式不对（FORMAT）、查不到（MISMATCH）。

### 路由

```python
def check(state) -> Command[Literal["extract", "ask_customer", "human_review", "file", "__end__"]]:
    if d.get("ticket_type") == "not_a_request":
        return Command(goto=END, update={"status": "ignored", ...})
    problems = validate(d, sender=m["from"], received_at=m["received_at"])
    kinds = {p["kind"] for p in problems}
    if not problems:
        return Command(goto="file", ...)
    if "FORMAT" in kinds:
        if state["attempts"] < MAX_EXTRACT_ATTEMPTS:
            return Command(goto="extract", update={"problems": problems, ...})
        return Command(goto="human_review", ...)
    if "MISMATCH" in kinds:
        return Command(goto="human_review", update={"problems": problems, ...})
    return Command(goto="ask_customer", update={"problems": problems, ...})
```

先修格式（模型能修的先修，修完再看别的），再看对不上的，最后才问客人。重抽时 `extract` 把
问题列表拼进提示词末尾：

```python
    if state.get("problems"):
        feedback = FEEDBACK.format(problems="\n".join(f"- {p['field']}：{p['message']}" for p in state["problems"]))
    raw = json_llm.invoke(EXTRACT.format(thread=thread, feedback=feedback))
```

这就是官方 `extraction/retries` 那个模式：校验错误是给模型看的，写清楚"收到什么、要什么"。

### 等客人回信：不用 interrupt

```python
def ask_customer(state) -> dict:
    ...
    tools.send_email(m["from"], f"Re: {m['subject']}", body)
    return {"status": "waiting_customer", "trail": [...]}
```

`ask_customer` 发一封信就结束了（`END`），没有 `interrupt`。客人的回信是同一条线程的新一封
邮件，`main.py` 用 `conversation_id` 当 `thread_id` 再跑一次图：`read_email` 把新邮件追加进
`emails`（`operator.add`），`extract` 看的是整条线程，前一封说"想退款"、后一封说"订单号
KL-901"，合在一起抽。**跟第 5 期的差别**：interrupt 是"这个图停在这里等一个答案"，适合坐席
几秒内会回的审批；客人回信可能是三天后，也可能永远不回，图不该挂在那里等，该结束、把状态
留在 checkpoint 里，下一封邮件来了再接上。

### 归并

```python
def file(state) -> dict:
    if state.get("ticket_no"):                       # 这条线程已经有工单：追加
        tools.append_update(state["ticket_no"], ...)
        return {...}
    existing = tools.find_open_ticket(d.get("order_id"), d.get("category"))
    if existing:                                     # 另一条线程、同订单同类别、还没解决：归并
        tools.append_update(existing["ticket_no"], ...)
        return {"ticket_no": existing["ticket_no"], ...}
    ...                                              # 新建
```

客人常常发两封：第一封说要改期，隔一小时另起一封"补充一下"。两张工单两个坐席各处理一遍是
真实的浪费，归并规则是代码：同订单、同类别、状态未解决。

## 跑起来

```bash
cd code
uv run python -m gap02_email_to_ticket.main --reset
uv run python -m gap02_email_to_ticket.main inbox          # 按收信顺序处理九封
uv run python -m gap02_email_to_ticket.main c4 discard 非下单人来信，请本人联系
uv run python -m gap02_email_to_ticket.main --tickets
uv run python -m gap02_email_to_ticket.main --outbox
```

## 你应该看到什么

### 九封邮件，九次模型调用

```
[c1] m01
  ── 收到 m01（c1）wang.hui@example.com：东京迪士尼门票想改日期
  extract（第 1 次）→ {'ticket_type': 'customer_demand', 'category': 'amendment', 'order_id': 'KL-778', 'customer_name': '王慧', 'request': '要求将两张东京迪士尼门票从9月8日改到9月10日', 'target_date': '2026-09-10', 'reason': '行程有变', 'language': 'zh'}
  check：全部通过
  file：新建 T-0001  customer_demand/amendment  订单 KL-778  优先级 normal（SLA 48h）
[c2] m02
  ── 收到 m02（c2）chen.jun@example.com：申请退款
  extract（第 1 次）→ {'ticket_type': 'customer_demand', 'category': 'refund', 'customer_name': '陈俊', 'request': '申请退款（大阪周游卡）', 'reason': '我买的大阪周游卡用不上了', 'language': 'zh'}
  check：MISSING order_id → 问客人
  ask_customer：已回信问 ['order_id']，等回复
[c3] m03
  ── 收到 m03（c3）zhao.min@example.com：包车司机迟到两小时，要求赔偿
  extract（第 1 次）→ {'ticket_type': 'customer_demand', 'category': 'compensation', 'order_id': 'KL-502', 'customer_name': '赵敏', 'request': '要求赔偿至少三百美元', 'amount': 300, 'reason': '司机迟到了两个多小时，导致我们错过了预约的午餐。', 'language': 'zh'}
  check：全部通过
  file：新建 T-0002  customer_demand/compensation  订单 KL-502  优先级 high（SLA 24h）
[c4] m04
  ── 收到 m04（c4）zhang.wei@example.com：帮朋友问一下 KL-778 能不能退
  extract（第 1 次）→ {'ticket_type': 'customer_demand', 'category': 'refund', 'order_id': 'KL-778', 'customer_name': '张伟', 'request': '帮朋友询问KL-778东京迪士尼门票能否全额退款', 'reason': '她去不了了', 'language': 'zh'}
  check：MISMATCH sender → 转人工
  ⏸ 转人工：['发件人 zhang.wei@example.com 不是订单 KL-778 的下单人（wang.hui@example.com）']
     选项 ['file', 'discard']
[c2] m05
  ── 收到 m05（c2）chen.jun@example.com：Re: 申请退款
  extract（第 1 次）→ {'ticket_type': 'customer_demand', 'category': 'refund', 'order_id': 'KL-901', 'customer_name': '陈俊', 'request': '申请退款', 'reason': '原因是同行的人生病了，行程取消。', 'language': 'zh'}
  check：全部通过
  file：新建 T-0003  customer_demand/refund  订单 KL-901  优先级 urgent（SLA 12h）
[c5] m06
  ── 收到 m06（c5）wang.hui@example.com：补充：迪士尼改期
  extract（第 1 次）→ {..., 'category': 'amendment', 'order_id': 'KL-778', 'request': '客人希望改期迪士尼门票，优先9月10日，若不可则9月11日', 'target_date': '2026-09-10', ...}
  check：全部通过
  file：订单 KL-778 已有同类工单 T-0001（另一条线程），归并进去
[c6] m07
  ── 收到 m07（c6）liu.yang@example.com：表扬一下你们的客服
  extract（第 1 次）→ {'ticket_type': 'feedback', 'category': 'praise', 'customer_name': '刘洋', 'request': '希望转达对导游小林的表扬', 'language': 'zh'}
  check：全部通过
  file：新建 T-0004  feedback/praise  订单 None  优先级 low（SLA 72h）
[c7] m08
  ── 收到 m08（c7）ops@hokkaido-charter.co.jp：KL-315 明日团次取消通知
  extract（第 1 次）→ {'ticket_type': 'merchant_request', 'category': 'cancellation', 'order_id': 'KL-315', 'request': '请协助通知客人并安排改期或退款', 'language': 'zh'}
  check：全部通过
  file：新建 T-0005  merchant_request/cancellation  订单 KL-315  优先级 high（SLA 24h）
[c8] m09
  ── 收到 m09（c8）newsletter@travel-deals.example.net：本周特惠：东南亚机票低至 3 折
  extract（第 1 次）→ {'ticket_type': 'not_a_request', 'category': 'other', 'language': 'zh'}
  check：不是诉求，忽略

9 封邮件，模型调用 9 次
```

逐封看：

- **c2 两封**：第一封没订单号，回信问；第二封回了"KL-901"，两封合起来抽，一次过。退款、
  出行日 9 月 6 日、收信 9 月 4 日上午——不到 48 小时，代码算成 `urgent`。
- **c3**：邮件里写的是"KL 502"（没横线）和"至少三百美元"，模型抽出 `KL-502` 和 `300`，
  格式全对。客人同时要赔偿和发票，提示词里说"一封一个类别、要钱的优先"，抽成 `compensation`，
  发票那条丢了——见常见问题。
- **c4**：张伟替朋友问王慧的订单。发件人跟下单人不一致，这是 MISMATCH，转人工。模型抽的
  字段本身没错，错的是"这封信有没有资格建这张单"，那是代码查订单系统才知道的。
- **c5**：王慧另起一封补充。同订单、同类别、T-0001 还开着，归并，回信告诉她"已并入"。
- **c7**：供应商域名的邮件，模型抽成 `merchant_request`（代码也会按域名再钉一次），
  不查发件人身份。
- **c8**：广告，模型标 `not_a_request`，不建单不回信。

### 转人工之后

```
$ uv run python -m gap02_email_to_ticket.main c4 discard 非下单人来信，请本人联系
[c4] 坐席：discard 非下单人来信，请本人联系
  human_review：坐席决定丢弃——discard 非下单人来信，请本人联系
```

### 工单库和发出去的信

```
$ uv run python -m gap02_email_to_ticket.main --tickets
  T-0001  normal  customer_demand/amendment     订单 KL-778  wang.hui@example.com   要求将两张东京迪士尼门票从9月8日改到9月10日
  T-0002  high    customer_demand/compensation  订单 KL-502  zhao.min@example.com   要求赔偿至少三百美元
  T-0003  urgent  customer_demand/refund        订单 KL-901  chen.jun@example.com   申请退款
  T-0001  ↳ 更新（c5）：wang.hui@example.com：刚才那封邮件忘了说，如果 9 月 10 日没票，9 月 11 日也可以。
  T-0004  low     feedback/praise               订单 None    liu.yang@example.com   希望转达对导游小林的表扬
  T-0005  high    merchant_request/cancellation 订单 KL-315  ops@hokkaido-charter.co.jp  请协助通知客人并安排改期或退款

$ uv run python -m gap02_email_to_ticket.main --outbox
  → chen.jun@example.com  Re: 申请退款
    您好，我们收到了您的邮件。为了尽快处理，请补充以下信息： / - 您的订单号（形如 KL-778，在确认邮件里能找到） / 直接回复本邮件即可。
  → wang.hui@example.com  Re: 补充：迪士尼改期
    您好，您的来信已并入工单 T-0001，我们会一起处理。
  → chen.jun@example.com  Re: Re: 申请退款
    您好，已为您建立工单 T-0003，我们会在 12 小时内跟进。
  ……
```

九封邮件，五张工单、一次归并、一次问客人、一次转人工、一封忽略。

### 重抽那条路：九封真邮件没走到

`FORMAT → extract` 这条边在九封真邮件上一次都没触发——"KL 502"模型自己补了横线，两遍都是。
这条路的机制用注入的坏结果验证：给 `extract` 塞一个 `order_id: "KL 502"`、`target_date: "下周五"`
的假结果，看图怎么走：

```
  extract（第 1 次）→ {..., 'order_id': 'KL 502', 'target_date': '下周五', ...}
  check：FORMAT order_id；FORMAT target_date → 喂回去重抽
  [注入] 第二次提示词里带着反馈：- order_id：order_id 必须写成 KL-三位数字（如 KL-778），收到 'KL 502' | - target_date：target_date 必须是 YYYY-MM-DD 或 null，收到 '下周五'（收信日期 2026-09-04）
  extract（第 2 次）→ {..., 'order_id': 'KL-502', 'target_date': '2026-09-11', ...}
  check：全部通过
```

路是通的，但"这个模型在真邮件上多久会抽错一次格式"这一篇没有数据。要有数据，得第 15 期那种
几十条用例跑出来。

## 发生了什么

**抽取的失败要按"谁能修"分类。** 模型抽错格式，喂回去它自己能修；客人没给的字段，模型再聪明
也抽不出，只能问客人；邮件和系统对不上，谁都不该自动拍板。三条路的成本差一个量级：重抽是
一次模型调用，问客人是几小时到几天，转人工是一个坐席的时间。分错类要么烦客人（把格式错误
当缺字段去问），要么放过风险（把发件人不符当格式错误重抽，模型会把发件人"修"成下单人吗？
它修不了，但它可能把 `order_id` 改掉）。

**校验消息是写给模型看的。** `"order_id 必须写成 KL-三位数字（如 KL-778），收到 'KL 502'"`
——期望、例子、实际值，三样都有，模型第二次就能修。`"invalid order_id"` 就修不了。这跟第 15
期给裁判写 rubric 是一个道理。

**等客人和等坐席是两种等。** 坐席审批用 `interrupt`，图停着，几秒到几分钟。客人回信用"结束 +
下一封邮件重新进图"，checkpoint 里留着前面的邮件，等多久都行。判断标准是"等的这个人在不在
这个系统里"。

**优先级、SLA、归并，都是代码。** 出行日期减收信日期、同订单同类别未解决——这些规则写下来
比让模型"评估紧急程度"稳，也能改：SLA 从 48 改 24 是改一个数字。

**一封一单是这一篇的简化。** c3 要赔偿也要发票，只建了赔偿单。真实系统里一封邮件抽成多张
工单是常态，`draft` 要变成列表，归并要按每一张单查。这是加分练习 1。

## 常见问题

**跟例子 1 到底什么关系？** 例子 1 的 `classify` 只抽两个标签（意图、紧急度）用来决定回复
路径；这一篇抽九个字段用来落库。真实系统两者串着：先建单（这一篇），再按工单类别决定怎么
回（例子 1）。例子 1 里 `bug_tracking` 节点那一行 `create_ticket()`，展开就是这一篇。

**为什么 ask_customer 的回信为什么用模板、不用模型写？** 问的内容是确定的（缺哪几个字段），模板
够用，省一次调用。要按客人语言写、要语气好一点，换成模型一行的事，例子 1 的 `draft_response`
就是那个写法。

**发件人并非下单人就一定转人工？** 这一篇是。真实规则更细：同一邮箱域名的家庭成员、代订
的旅行社、客人在订单上留的备用联系人——这些都是订单系统里能查的事实，查得到就放行，
是给 `validate` 加规则，而非给模型加提示词。

**一条线程停在 human_review 上，客人又来一封怎么办？** `main.py` 里 inbox 模式直接跳过并
提示。真实系统里要排队：坐席处理完（file 或 discard）再把后面的邮件放进去。图本身不管
排队，那是图外面的事。

**模型抽错但格式没错怎么办？** 比如把 9 月 10 日抽成 9 月 11 日，校验挡不住。工单上保留
原始邮件（`emails` 都在 state 里），坐席打开工单时对照看。抽取的准确率要靠第 15 期那种
用例集量，不能靠校验。

## 加分练习

1. 一封多单：`draft` 变成列表，c3 那封抽出 `compensation` 和 `invoice` 两张，归并逐张查。
2. `REQUIRED_FROM_CUSTOMER` 加一条 `invoice: ["company_name"]`，写一封开发票的邮件，看
   ask_customer 问对了没有。
3. 给 `validate` 加规则：发件人跟下单人同域名（`@example.com`）且订单备注里有"允许家属
   联系"就放行。数据在 `orders.py` 加字段。
4. 用第 15 期的方法给 `extract` 写二十条用例，其中五条订单号故意写错格式（少横线、小写、
   多空格），量一量重抽这条路真实的触发率和修复率。
5. 把 `ask_customer` 发出去的信换成模型写的、用客人的语言（`language` 字段已经抽出来了）。
