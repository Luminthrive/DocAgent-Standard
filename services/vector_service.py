#向量处理服务
import uuid
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings
from loguru import logger
from qdrant_client.conversions.common_types import Distance, Filter
from qdrant_client.http.models import VectorParams, FieldCondition, MatchValue, PointStruct
from config import config

class VectorService:
    #注意当前需要在lifespan就初始化服务类
    def __init__(self,vector_client):
        self.embeddings=OpenAIEmbeddings(
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
            check_embedding_ctx_length=False,
            chunk_size=64,
            max_retries=3,
            request_timeout=300
        )
        self.client=vector_client

    async def ensure_collection(self):
        if not await self.client.collection_exists(config.qdrant_collection):
            self.client.create_collection(
                collection_name=config.qdrant_collection,
                vectors_config=VectorParams(
                    size=config.qdrant_collection_size,
                    distance=Distance.COSINE
                )
            )
        logger.info(f"qdrant collections已创建!")

    async def search(self,kb_id:str,query:str,top_k:int=5)->List[Dict[str,Any]]:
        logger.info(f"向量搜素：kb_id={kb_id},query={query[:30]},top_K={top_k}")
        await self.ensure_collection()
        query_vector =await self.embeddings.aembed_query(query)
        q_filter=Filter(must=[FieldCondition(key="kb_id",match=MatchValue(value=kb_id))])
        resp=await self.client.query_points(collection_name=config.qdrant_collection,
                                            query=query_vector,query_filter=q_filter,limit=top_k,with_payload=True)
        results=[]
        for p in resp.points:
            payload=p.payload or {}
            results.append({
                "text":payload.get("text",""),
                "score":round(float(p.score),4),
                "metadata":{
                    "kb_id":payload.get("kb_id",kb_id),
                    "doc_id":payload.get("doc_id",""),
                    "chunk_index":payload.get("chunk_index",0),
                    "file_name":payload.get("file_name",""),
                }
            })
        return results

    async def add_vector(self,kb_id:str,doc_id:str,texts:List[str],file_name:str="")->int:
        if not texts:
            return 0
        await self.ensure_collection()
        vectors=await self.embeddings.aembed_documents(texts)
        point_ids=[str(uuid.uuid4()) for _ in texts]
        points=[
            PointStruct(id=pid,vector=vec,payload={
                "kb_id":kb_id,
                "doc_id":doc_id,
                "chunk_index":idx,
                "file_name":file_name,
                "text":text,
            }) for pid ,text,vec,idx in zip(point_ids,texts,vectors,range(len(texts)))
        ]
        await self.client.upsert(collecton_name=config.qdrant_collection,points=points)
        logger.info(f"向量入库:kb_id={kb_id} doc_id={doc_id} count={len(points)}")
        return len(point_ids)

    async def delete_by_doc_id(self,kb_id:str,doc_id:str)->int:
        q_filter=Filter(must=[FieldCondition(key="doc_id",match=MatchValue(value=doc_id))])
        result=await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted=getattr(result,"status","")
        logger.info(f"删除文档向量:kb_id={kb_id} doc_id={doc_id} status={deleted}")
        return 1

    async def delete_by_kb_id(self,kb_id:str)->int:
        q_filter=Filter(must=[FieldCondition(key="kb_id",match=MatchValue(value=kb_id))])
        result=await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted=getattr(result,"status","")
        logger.info(f"删除知识库向量:kb_id={kb_id} status={deleted} ")
        return 1

    async def count(self,kb_id:str)->int:
        try:
            q_filter=Filter(must=[FieldCondition(key="kb_id",match=MatchValue(value=kb_id))])
            result=await self.client.count(
                collection_name=config.qdrant_collection,
                count_filter=q_filter,
                exact=True
            )
            return int(result.count)
        except Exception as e:
            logger.warning(f"向量计数失败: kb_id={kb_id} error={e}")
            return 0
