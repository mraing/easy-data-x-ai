"""最小示例：流式工具调用 + pyseekdb 真实向量检索，配置读取 .env。

运行：uv run d1/d1_5_tool_use_seekdb.py

和 d1_4 的区别只有两点：
1. 用 stream，所以要先把分片累加成完整的 AIMessageChunk 才能拿到 tool_calls；
2. 知识库换成 pyseekdb 向量检索，不再是关键词子串匹配。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from seekdb_knowledge_base import search_knowledge_base

# 自动从当前文件所在目录逐级向上查找 .env
load_dotenv()


@tool
def query_knowledge_base(query: str) -> str:
    """从知识库中检索与用户问题相关的内容。当用户的问题涉及产品功能、技术文档、操作指南等需要查阅知识库的场景时使用。"""
    return search_knowledge_base(query)


llm = init_chat_model(
    os.environ["LLM_NAME"],
    model_provider="openai",
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

# 把工具绑定到模型上，模型才能在需要时「点名」要用哪个工具
llm_with_tools = llm.bind_tools([query_knowledge_base])

# 工具名 -> 工具对象，模型返回的是名字，程序要按名字找回真正的函数
tools_by_name = {query_knowledge_base.name: query_knowledge_base}

messages = [
    ("system", "你是一个技术文档助手。当用户问及产品相关问题时，请先查询知识库获取准确信息。"),
    ("user", "seekdb 有哪些查询方式？"),
]

max_rounds = 5
executed_tool_rounds = 0

while True:
    # 流式调用并累加分片：content 逐字打印，tool_calls 要等累加完才是完整的
    aggregated = None
    for chunk in llm_with_tools.stream(messages):
        aggregated = chunk if aggregated is None else aggregated + chunk
        if chunk.content:
            print(chunk.content, end="", flush=True)

    if not aggregated.tool_calls:
        break

    if executed_tool_rounds >= max_rounds:
        raise RuntimeError(f"工具调用已达到 {max_rounds} 轮上限")

    messages.append(aggregated)

    for tool_call in aggregated.tool_calls:
        print(f"\n[第 {executed_tool_rounds + 1} 轮] 模型请求工具：{tool_call['name']}({tool_call['args']})")
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        print(f"          工具返回：{result}")
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    executed_tool_rounds += 1

print()  # 换行
