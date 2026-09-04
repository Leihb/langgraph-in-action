# 对话式数据分析——SQL、图表、多轮细化

> 原型是 Inconvo 这类"嵌进产品里给业务用户问数"的服务：问一句话，返回一份图表的结构化描述
> （而非一段文字），还能接着说"拆成按品类""只看日本""换成柱状图"。开源里的 SQL agent 全是
> 一问一答出文本（例子 2 就是），缺的是"生成 SQL → 执行 → 选图 → 结构化输出 → 多轮细化"这一整条。
> 用到的机制：例子 2（json_mode 出 SQL、EXPLAIN 校验、只读连接）、第 4 期（checkpointer，一段对话
> 一个 thread）。

例子 2 的用户是会看 SQL 的人：模型写查询，人批准，跑，回一段话。这一篇的用户是业务同事：
不看 SQL，要看图，看完还要追问。三处不一样：**多轮**——state 里留着前几轮的问题、SQL 和结果，
用户说"拆成按品类"，模型在上一轮的 SQL 上改；**图表**——结果是一份结构化的图表规格（给前端）
加终端里的字符图（给人），选什么图由代码按结果的行列结构决定；**没有执行前审批**——只读连接、
SELECT-only、EXPLAIN 三道保险留着，审批那一步业务用户等不起。

数据是这本书自己的旅行订单：两张表（`products` 十二个产品，`bookings` 一万五千多条订单，
2025 年初到今天），固定种子生成，谁跑都是同一份。

## 敲进去

代码在 `code/gap03_conversational_analytics/`：`db.py`（造数据、schema、校验、只读执行，例子 2 的
那套）、`charts.py`（选图、画图，全是代码）、`prompts.py`（两段提示词）、`graph.py`、`main.py`。

### 图

```
understand（模型：SQL + 图表偏好）→ check（代码）→ run（代码）→ chart（代码：选图、画图）→ narrate（模型）
         ▲                              │ 校验/执行报错 ≤3 次
         └──────────────────────────────┘
mode=chart_only：跳过 check/run，拿上一轮的结果直接 chart
mode=cannot：直接 END
```

五个节点，模型两个：`understand` 把（多轮）问题翻成 SQL 和一个图表偏好，`narrate` 看着结果说
两句话。中间三个是代码。

### 多轮：把前几轮塞给模型

```python
class AnalyticsState(TypedDict):
    question: str
    turns: Annotated[list[Turn], operator.add]   # 已完成的轮次：问题、SQL、列、行、图表规格、两句话
    mode: str                                     # query / chart_only / cannot
    sql: str | None
    chart_hint: str | None
    ...


def understand(state):
    history = ""
    if state.get("turns"):
        recent = state["turns"][-3:]
        history = "前几轮的对话（最近的在最后）：\n" + "".join(
            HISTORY_ITEM.format(i=i + 1, question=t["question"], sql=t.get("sql"), n=len(t.get("rows", [])),
                                columns=", ".join(t.get("columns", []))) for i, t in enumerate(recent)) + "\n"
    if state.get("error"):
        history += f"你上一版 SQL 有问题：{state['error']}。请改正后重写。\n\n"
    out = json_llm.invoke(UNDERSTAND.format(today=db.TODAY.isoformat(), schema=db.load_schema(),
                                            history=history, question=state["question"]))
```

给模型看的是最近三轮的**问题、SQL、结果的列名和行数**，不给结果本身——结果几十行几百行，
塞进去是浪费，模型改 SQL 只需要知道上一条 SQL 长什么样。提示词里明说："用户可能在追问上一轮，
这时要在上一轮 SQL 的基础上改；如果只是要换图表样式、数据不用变，mode 填 chart_only。"

### 选图是代码

```python
def choose(columns, rows, hint):
    nums = _numeric_cols(columns, rows)
    cats = [i for i in range(len(columns)) if i not in nums]
    if len(columns) == 3 and len(cats) == 2 and len(nums) == 1:      # 三列长表 → 宽表
        columns, rows = pivot_long(columns, rows)
        ...
    if len(rows) == 1 and len(nums) == 1 and len(columns) == 1:
        kind = "number"
    elif len(cats) == 1 and cats[0] == 0 and 1 <= len(nums) <= 5 and len(rows) <= 40:
        kind = "line" if _looks_like_date(rows, 0) else "bar"
    else:
        kind = "table"
    if hint and hint in VALID_TYPES and hint != kind:
        ok = (hint == "pie" and kind == "bar" and len(nums) == 1 and len(rows) <= 8) \
             or (hint in ("bar", "line") and kind in ("bar", "line")) or hint == "table"
        kind = hint if ok else kind   # 不合适就忽略偏好，并在返回里说明
```

规则看结果的结构：一个数 → 大数字；一列类别加几列数字 → 柱状，类别像日期 → 折线；三列长表
（月份、品类、金额）先转宽表再画多序列；其他 → 表格。模型的偏好（用户说"换成饼图"）只在结构
允许时采纳。**模型知道用户想看什么，代码知道这份数据能画成什么。**

`pivot_long` 值得单独说：模型按 `GROUP BY month, category` 写出来的天然是长表，多序列图要宽表，
这一步转换是确定的，不该让模型"输出宽表"（它得写一堆 `CASE WHEN`，错的概率高得多）。

### 不完整的周期，代码来标

```python
def partial_period_caveat(columns, rows, today) -> str:
    if not rows or not _looks_like_date(rows, 0):
        return ""
    last = str(rows[-1][0])
    if last == today.strftime("%Y-%m"):
        return f"{last} 是当前月，只到 {today.day} 号，不是完整月份"
    ...
```

这一条是真机跑出来才加的，下面"你应该看到什么"里有它的来历。标注同时进图表规格（`spec["caveat"]`）、
终端渲染（图下面一行 ⚠）和 `narrate` 的提示词。

## 跑起来

```bash
cd code
uv run python -m gap03_conversational_analytics.main a1 "过去 12 个月每月的销售额"
uv run python -m gap03_conversational_analytics.main a1 "拆成按品类"
uv run python -m gap03_conversational_analytics.main a1 "只看日本"
uv run python -m gap03_conversational_analytics.main a1 "换成柱状图"
uv run python -m gap03_conversational_analytics.main --spec a1        # 最后一轮的图表规格 JSON
uv run python -m gap03_conversational_analytics.main --history a1
```

加 `--sql` 打印每轮的 SQL。

## 你应该看到什么

### 四轮追问

```
[a1] 用户：过去 12 个月每月的销售额
  understand #1 → SELECT strftime('%Y-%m', order_date) AS month, SUM(amount_usd) AS sales_amount FROM bookin
  check → 通过
  run → 13 行 × 2 列
  chart → line（13 行 × 2 列）
  SQL: SELECT strftime('%Y-%m', order_date) AS month, SUM(amount_usd) AS sales_amount FROM bookings WHERE status = 'confirmed' AND order_date >= date('now', '-12 months') GROUP BY month ORDER BY month

  ▍过去12个月每月销售额
      235,957 ┤                                  ●
              ┤                               ●
              ┤          ●
              ┤    ●           ●  ●  ●  ●
              ┤ ●           ●              ●
      104,870 ┤       ●
              ┤
              ┤
              ┤                                     ●
         0.00 ┤
              └ 09  10  11  12  01  02  03  04  05  06  07  08  09
                25-09 … 26-09

  最值得注意的是，2026年9月的销售额仅为14,241.34，远低于此前所有月份的10万以上水平，形成断崖式下滑。这可能说明该月数据不完整或出现异常，但仅从数据看，这是过去13个月中最突出的异常值。
  （这轮模型调用 2 次）

[a1] 用户：拆成按品类
  understand #2 → SELECT strftime('%Y-%m', b.order_date) AS month, p.category AS category, SUM(b.amount_usd)
  run → 64 行 × 3 列
  chart → line（13 行 × 6 列），已把长表转成宽表
  SQL: ... JOIN products p ON b.product_id = p.product_id WHERE b.status = 'confirmed' AND b.order_date >= date('now', '-12 months') GROUP BY month, category ORDER BY month, category

  ▍过去12个月各品类销售额趋势
      105,717 ┤                                  ▲
              ┤                   ▲     ▲
              ┤    ▲     ▲     ▲     ▲        ▲
              ┤             ▲
              ┤ ▲                                ●
       46,985 ┤       ▲  ●                 ▲  ●  ◇
              ┤    ●     ◇  ●  ●  ●  ●  ●  ●  ◇
              ┤ ✱  ◇  ●     ◇  ◇  ◇  ◇  ◇  ◇  ■
              ┤ ■  ■  ✱  ✱  ✱  ✱  ✱  ■  ■  ■  ◆  ✱
         0.00 ┤ ◆  ◆  ◆              ◆  ◆  ◆        ✱
              └ 09  10  11  12  01  02  03  04  05  06  07  08  09
     ● 一日游  ◆ 交通卡  ■ 体验  ▲ 包车  ◇ 景点门票

[a1] 用户：只看日本
  understand #3 → ... AND p.destination_country = '日本' GROUP BY month, category ...
  chart → line（13 行 × 6 列），已把长表转成宽表

[a1] 用户：换成柱状图
  understand #4 → 只换图：bar，数据沿用上一轮
  chart → bar（13 行 × 6 列）
  （这轮模型调用 1 次）
```

第二轮模型在第一轮的 SQL 上加了 `JOIN products` 和 `p.category`，64 行长表被代码转成 13 行 × 6 列
画多序列折线。第三轮又加了一个 `WHERE`。第四轮模型判断"数据不用变"，一次调用（没有 narrate），
拿上一轮的 64 行直接画柱状图。三次追问，SQL 一步步长，每一步都能在 `--history` 里看到。

### 第一轮那两句话是错的

第一轮的两句话全在讲"2026 年 9 月断崖式下滑"。9 月才过了 4 天。模型写的 `date('now', '-12 months')`
把当前月带了进来（13 行而非 12 行），然后它自己看着 13 个点，把不完整的最后一个点当成了
异常。它在第二句里犹疑了一下（"可能说明该月数据不完整"），但第一句已经说出去了。

这不该靠模型自己识破。代码知道今天是几号、知道第一列是月份、知道最后一行是当前月，加了
`partial_period_caveat` 之后：

```
[a4] 用户：今年每个月的销售额
  chart → line（9 行 × 2 列），标注：2026-09 是当前月，只到 4 号，不是完整月份

  ▍2026年每月销售额
      235,957 ┤                      ●
              ┤                   ●
              ┤    ●  ●  ●  ●
              ┤ ●              ●
              ┤                         ●
              └ 01  02  03  04  05  06  07  08  09
     ⚠ 2026-09 是当前月，只到 4 号，不是完整月份

  最值得注意的一点是：销售额从8月的峰值235,956.79骤降到9月仅14,241.34。不过2026-09目前只包含4天数据，不是完整月份，因此该月数值不宜与整月直接比较。
```

标注进了图表规格、进了图、进了 `narrate` 的提示词，模型这次的两句话跟着改了口。顺带一句：
同一个问题"过去 12 个月每月的销售额"在另一个 thread 里再问一次，模型写的是
`order_date >= '2025-09-01' AND order_date < '2026-09-01'`，12 行，不含当前月——同一个问题两种
SQL，"过去 12 个月"含不含本月模型自己没有定见。这类口径问题见"发生了什么"。

### 另一段对话：排行、饼图被拒、退款率、一个数、查不了

```
[a2] 用户：上个月哪个目的地卖得最好
  chart → bar（9 行 × 2 列）
  ▍上个月销售额最高的目的地排行
     札幌  ████████████████████████████████████████ 95,915
     东京  █████████████████████████ 60,839
     巴厘岛 ███████ 16,404
     釜山  ██████ 15,437
     ……
  从数据看，上个月卖得最好的目的地是札幌，销售额约9.59万美元，显著高于其他城市。它的销售额几乎是第二名东京的1.6倍……

[a2] 用户：换成饼图
  understand #2 → 只换图：pie，数据沿用上一轮
  chart → bar（9 行 × 2 列），用户想要 pie，但结果形状不适合，按规则用 bar
  （这轮模型调用 1 次）

[a2] 用户：退款率最高的品类是哪个
  SQL: SELECT p.category, ROUND(100.0 * SUM(CASE WHEN b.status = 'refunded' THEN b.amount_usd ELSE 0 END) / NULLIF(SUM(CASE WHEN b.status IN ('confirmed','refunded') THEN b.amount_usd ELSE 0 END), 0), 2) AS refund_rate_pct ... WHERE strftime('%Y-%m', b.order_date) = '2026-08' GROUP BY p.category ...
  ▍2026年8月各品类退款率
     包车   ████████████████████████████████████████ 14.05
     一日游  ████████████████████████████ 9.77
     体验   ████████████████████████ 8.28
     景点门票 ████████████ 4.17
     交通卡  ██████ 2.10

[a2] 用户：去年 8 月的总销售额是多少
  chart → number（1 行 × 1 列）
  ▍去年8月总销售额
     218,972  (total_sales_usd)

[a2] 用户：客人的年龄分布
  understand #5 → cannot：当前数据中没有客人的年龄字段，无法查询年龄分布。
  （这轮模型调用 1 次）
```

饼图被拒是代码的决定：九个目的地切饼没法看，规则是"饼图最多八片"，用户的偏好被忽略，trail 里
写了为什么。（模型给的标题"各目的地销售额占比"还留着——标题是模型起的、图是代码选的，两边没
对齐，见常见问题。）

第三轮值得多看一眼：用户问"退款率最高的品类"，没说时间，模型沿用了前两轮的"上个月"，
`WHERE ... = '2026-08'`；退款率它定义成**金额**口径（退款金额 / 成交加退款金额），而非笔数。
两个决定都不算错，但都是模型替用户做的。全时段、按笔数算，包车也是最高（12%），这次结论碰巧
一样，下次不一定。

## 发生了什么

**多轮细化的实现很朴素：把上几轮的 SQL 给模型看。** 不用什么"对话状态跟踪"，checkpointer 里
`turns` 列表就是对话状态，模型看着上一条 SQL 改一个 `JOIN`、加一个 `WHERE`，比从头理解"拆成按
品类"这四个字容易得多。给的是 SQL 和列名，不给结果——改查询不需要看数据。

**图表规格是产品，字符图是调试。** `spec` 那份 JSON（类型、x、系列、数据、标题、标注）是给前端
渲染的，这一篇的输出就是它；终端里的字符图是为了不搭前端也能看见结果对不对。真实产品里
`render` 整个可以删掉。

**选图交给代码，因为代码看得见结构。** 一列日期加五列数字，画折线；一列类别加一列数字、九行，
画柱状不画饼——这些判断的输入是结果的列类型和行数，代码手里都有，模型手里只有用户的一句话。
模型的偏好当建议，结构不合就否。

**"过去 12 个月"含不含本月，"退款率"按金额还是笔数，是口径，不是查询。** 口径应该在一个地方
写死，让模型查表，不让它每次现场决定。这一篇只做了最小的一步：代码在图上标出不完整的周期。
完整的做法是一份"指标字典"（销售额 = confirmed 的 amount_usd；退款率 = refunded 笔数 / 全部笔数；
"过去 N 个月"不含本月），既进提示词也进校验——加分练习 1。

**上下文会漏。** "退款率最高的品类"沿用了上一轮的"上个月"。这是多轮的代价：模型分不清用户是在
追问还是换了话题。修法不在模型这边——`understand` 的输出里加一个 `carried_filters` 字段，把沿用的
条件列出来，图的标题和标注里显示"（2026-08）"，用户一眼看到不对就会说"不是，全部时间"。让
沿用的条件**可见**，比让模型猜对更可靠。

## 常见问题

**为什么没有执行前审批？** 例子 2 讲过审批，这一篇的场景是业务同事随手问数，每问一句等人批
不现实。安全靠三道代码保险：只读连接（`mode=ro`，写不进去）、SELECT-only 规则、EXPLAIN 校验。
要是数据里有敏感列，加第四道：校验里查 SQL 有没有碰白名单外的列。

**标题是模型起的、图是代码选的，饼图被拒时标题还写着"占比"？** 是这一篇没收拾的毛边。修法：
`chart` 节点在改了图类型时把标题里的类型词也换掉，或者标题干脆由代码按"指标 × 维度 × 时间"
拼——加分练习 3。

**重试那条边走到了吗？** 十四轮真机没有一次校验或执行报错，`check → understand` 和
`run → understand` 两条边没走到。机制跟例子 2 一样（错误文本回给模型），例子 2 也没在真机上
触发过。这个模型在两千字符的 schema 上写单表和两表 JOIN 很少出错；表多了、列名怪了会不一样。

**结果几百行怎么办？** `ROW_CAP = 200`，超过截断；`choose` 里超过 40 行不画柱状折线、直接表格。
业务问数很少真的要看几百行，要看的时候是导出而非画图。

**`date('now')` 和提示词里的"今天"不一致怎么办？** 提示词给了 `today`，模型第一轮却用了 SQLite 的
`date('now')`——这台机器上两者相同，换台机器就不同。要严格，校验里禁掉 `now`，让模型只写字面日期。

## 加分练习

1. 写一份 `metrics.yaml`：每个指标的名称、定义（SQL 片段）、默认口径（含不含本月、按金额还是
   笔数），拼进 `UNDERSTAND`，再在 `check` 里校验模型用的口径跟字典一致。
2. `understand` 的输出加 `carried_filters: [...]`，把从上一轮沿用的条件列出来，显示在图标题里；
   用户说"全部时间"时能看到它被清掉。
3. 标题由代码拼：指标 × 维度 × 时间范围，模型只出这三样。
4. 把 `spec` 喂给一个真的图表库（前端 Vega-Lite 或 Python 的 matplotlib），看字段名要改哪几处。
5. 用第 15 期的方法写二十条问数用例，其中五条带相对时间（"上个月""今年""过去半年"），
   量一量模型的时间边界和 `TODAY` 对齐的比例。
