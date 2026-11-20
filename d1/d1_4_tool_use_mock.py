"""最小示例：LangChain 手动工具调用（ReAct 循环的骨架版），配置读取 .env。

运行：uv run d1_4_tool_use_mock.py

关键点：
1. 必须用 bind_tools 绑定过的模型去对话，否则模型根本不知道有这个工具；
2. 模型返回 tool_calls 后，程序要自己执行工具、把结果包成 ToolMessage 塞回 messages，
   再问一次模型，直到模型不再要求调用工具为止。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from mock_knowledge_base import search_mock_knowledge_base

load_dotenv()  # 自动从当前文件所在目录逐级向上查找 .env


@tool
def query_knowledge_base(query: str) -> str:
    """从知识库中检索与用户问题相关的内容。当用户的问题涉及产品功能、技术文档、操作指南等需要查阅知识库的场景时使用。"""
    # 这里调用你的知识库检索逻辑
    return search_mock_knowledge_base(query)


llm = init_chat_model(
    os.environ["LLM_NAME"],
    model_provider="openai",  # 通过 OpenAI 兼容接口调用
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

# 把工具绑定到模型上：绑完之后模型才能在需要时「点名」要用哪个工具
llm_with_tools = llm.bind_tools([query_knowledge_base])

# 工具名 -> 工具对象，模型返回的是名字，程序要按名字找回真正的函数
tools_by_name = {
    query_knowledge_base.name: query_knowledge_base,
}

messages = [
    ("system", "你是一个技术文档助手。当用户问及产品相关问题时，请先查询知识库获取准确信息。"),
    ("user", "seekdb 支持哪些检索方式？"),
]

max_rounds = 5
executed_tool_rounds = 0

while True:
    # 每一轮都用「绑定了工具的模型」调用，并且带上到目前为止的完整对话
    response = llm_with_tools.invoke(messages)
    if not response.tool_calls:
        # 模型不再要求调用工具，说明它已经能直接回答了
        final_response = response
        break

    if executed_tool_rounds >= max_rounds:
        raise RuntimeError(f"工具调用已达到 {max_rounds} 轮上限")

    # 先把这个「带 tool_calls 的 AI 消息」放回历史，否则后面的 ToolMessage 对不上号
    messages.append(response)

    # 同一轮可能返回多个 tool call，要全部执行
    for tool_call in response.tool_calls:
        print(f"[第 {executed_tool_rounds + 1} 轮] 模型请求工具：{tool_call['name']}({tool_call['args']})")
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        print(f"          工具返回：{result}")
        # tool_call_id 必须和模型给的那个 id 一致，模型才能把结果对上号
        messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    executed_tool_rounds += 1

print("\n=== 最终回答 ===")
print(final_response.content)
