"""一个工具：upsert_memory。第 6 期的 remember_note 是"一人一条笔记、整条覆盖"；
这里是"一条记忆一个 key"，新增和修改分开——修改要带上原来的 id。"""

import uuid
from datetime import date

from langchain.tools import ToolRuntime, tool


@tool
def upsert_memory(content: str, context: str, memory_id: str | None = None, *, runtime: ToolRuntime) -> str:
    """把一条关于客人的长期信息存进记忆库，或者更新已有的一条。

    content：记忆本身，一句话，比如"客人出行需要无障碍安排，坐轮椅"。
    context：这条信息是在什么情况下说的，比如"预订东京行程时提到"。
    memory_id：只在更新已有记忆时传——客人纠正了之前说过的话、或者新信息跟某条
    旧记忆是同一件事，就传那条的 id 覆盖它，不要另存一条重复的。新记忆不传。
    """
    user_id = runtime.config["configurable"]["user_id"]
    key = memory_id or uuid.uuid4().hex[:8]
    runtime.store.put(("memories", user_id), key, {"content": content, "context": context, "updated": date.today().isoformat()})
    return f"{'已更新' if memory_id else '已新增'}记忆 {key}"
