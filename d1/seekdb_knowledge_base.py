"""最小示例：用 pyseekdb 做真正的向量检索，替代 mock_knowledge_base。

单独测试：uv run d1/seekdb_knowledge_base.py

说明：每次运行都重建一个空库。macOS 上重开已有库会报
`seekdb_open failed: code=-1`（clog 初始化失败），所以这个 demo 不做持久化。
"""

import shutil
from pathlib import Path

import pyseekdb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).resolve().parent.parent / ".tmp" / "seekdb_kb"

DOCUMENTS = [
    "seekdb 支持关键词检索、语义向量检索和混合检索。",
    "seekdb 支持嵌入模式和服务器模式两种部署方式。",
    "pyseekdb 是 seekdb 的 Python SDK，支持 Schemaless API。",
]
IDS = ["doc1", "doc2", "doc3"]


class ChineseEmbeddingFunction(pyseekdb.EmbeddingFunction):
    """默认的 all-MiniLM-L6-v2 是英文模型，中文检索会排错，这里换成中文小模型。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def __call__(self, documents):
        return self.model.encode(list(documents), normalize_embeddings=True).tolist()

    def get_config(self) -> dict:
        return {}


_collection = None


def get_collection():
    """建库、建 collection、写入 3 条文档，返回可查询的 collection。"""
    global _collection
    if _collection is None:
        shutil.rmtree(DB_DIR, ignore_errors=True)
        client = pyseekdb.Client(path=str(DB_DIR))
        _collection = client.get_or_create_collection(
            "knowledge_base", embedding_function=ChineseEmbeddingFunction()
        )
        # 只给文本，向量由 embedding function 自动生成
        _collection.add(ids=IDS, documents=DOCUMENTS)
        # 写完后刷新索引，否则查询可能返回空
        _collection.refresh_index()
    return _collection


def search_knowledge_base(query: str, n_results: int = 2) -> str:
    """向量检索 top-k，返回拼接后的文档正文。"""
    result = get_collection().query(query_texts=query, n_results=n_results)
    return "\n\n".join(result["documents"][0])


if __name__ == "__main__":
    collection = get_collection()
    for query in ["有哪些查询方式", "怎么部署", "Python 的 SDK 叫什么"]:
        result = collection.query(query_texts=query, n_results=2)
        print(f"\nQ: {query}")
        for document, distance in zip(result["documents"][0], result["distances"][0]):
            print(f"  {distance:.4f}  {document}")
