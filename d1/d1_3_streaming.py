"""最小示例：LangChain 通过 OpenAI 兼容接口调用模型，配置读取 .env。

运行：uv run d1_3_streaming.py

流式输出
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # 自动从 main.py 所在目录逐级向上查找 .env

llm = init_chat_model(
    os.environ["LLM_NAME"],
    model_provider="openai",  # 通过 OpenAI 兼容接口调用
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

for chunk in llm.stream([
    ("system", "你是一个技术助手。"),
    ("user", "用三句话解释什么是向量数据库。"),
]):
    print(chunk.content, end="", flush=True)

print()  # 换行