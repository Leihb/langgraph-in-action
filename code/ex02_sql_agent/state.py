"""例子 2 的状态。SQL 是一个字段，不是一条工具调用消息——图里没有 ToolNode。"""

import operator
from typing import Annotated, TypedDict


class SqlState(TypedDict, total=False):
    question: str
    schema: str            # 全库的 CREATE TABLE 语句，代码从 sqlite_master 读出来
    sql: str | None        # 模型生成的查询；None 表示模型判断这个问题查不了
    reason: str            # 模型给的一句说明（为什么这么写 / 为什么查不了）
    check_error: str       # 校验没过的原因，回给模型改
    run_error: str         # 执行报错的原因，回给模型改
    attempts: int          # 生成了几次，防止无限改
    approval: str          # 人工决定：accept / edit / reject
    columns: list[str]
    rows: list[list]
    answer: str
    trace: Annotated[list[str], operator.add]
