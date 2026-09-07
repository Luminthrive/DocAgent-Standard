from typing import Optional, Dict, Any

from langchain_core.tools import tool

from agent_graph.schema import RagSearchInput
from config import config


def create_tools(hybrid_search_service, knowledge_service):
    """
    创建工具列表

    Args:
        hybrid_search_service: 混合检索服务（包含多路召回+Rerank）
        knowledge_service: 知识库服务
    """

    @tool(args_schema=RagSearchInput,
          description="检索知识库中的相关文档。当用户提出需要从知识库中查找信息的问题时调用此工具，返回与问题相关的文档片段")
    async def rag_search(query: str, kb_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """
        混合检索工具

        支持两种模式:
        - 单路检索: 当前默认
        - 多路召回: 通过环境变量 ENABLE_MULTI_REWRITE 控制
        """
        use_multi = config.enable_multi_rewrite

        docs = await hybrid_search_service.search(
            query=query,
            kb_id=kb_id or "default",
            top_k=top_k,
            use_multi_rewrite=use_multi,
        )

        return {"docs": docs, "count": len(docs)}

    @tool(description="列出当前用户拥有的全部知识库。当用户询问有哪些知识库、想切换知识库、或需要知道知识库信息时调用此工具")
    async def list_knowledge_bases(user_id: int) -> Dict[str, Any]:
        kbs = await knowledge_service.list_knowledge_bases(user_id=user_id, limit=100, offset=0)
        return {"knowledge_bases": kbs, "count": len(kbs)}

    return [rag_search, list_knowledge_bases]
