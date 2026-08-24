from typing import Optional, Dict, Any

from langchain_core.tools import tool

from agent_graph.schema import RagSearchInput


def create_tools(vector_service,knowledge_service):
    @tool(args_schema=RagSearchInput,
          description="检索知识库中的相关文档，根据用户问题在向量库中进行相似度检索，返回相关文档片段")
    async def rag_search(query: str, kb_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        docs = await vector_service.search(
            kb_id=kb_id or "default",
            query=query,
            top_k=top_k
        )
        return {"docs":docs,"count":len(docs)}

    @tool(description="列出当前用户拥有的全部知识库")
    async def list_knowledge_bases(user_id:int)->Dict[str,Any]:
        kbs=await knowledge_service.list_knowledge_bases(user_id=user_id,limit=100,offset=0)
        return {"knowledge_bases":kbs,"count":len(kbs)}

    return [rag_search,list_knowledge_bases]