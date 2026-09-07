#向量处理服务 - 支持混合检索 (Dense + Sparse + RRF)
import uuid
from typing import List, Dict, Any

import httpx
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from qdrant_client.http.models.models import Distance, Filter
from qdrant_client.http.models import (
    VectorParams, SparseVectorParams, Modifier,
    FieldCondition, MatchValue, PointStruct,
    Query, Prefetch, Fusion
)
from config import config


class VectorService:
    """
    向量服务 - 支持混合检索
    - Dense: 通过模型 API 获取 (语义检索)
    - Sparse: 通过模型 API 获取 BM25 (关键词检索)
    - 融合: Qdrant 内置 RRF
    """

    def __init__(self, vector_client):
        # Dense embedding: 使用模型 API (OpenAI 兼容接口)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            base_url=f"{config.model_server_url}/v1",
            api_key="not-needed",  # 模型服务器不需要 API key
            check_embedding_ctx_length=False,
            chunk_size=64,
            max_retries=3,
            request_timeout=300
        )
        self.client = vector_client
        self.model_server_url = config.model_server_url
        self.sparse_timeout = 30.0

    async def _encode_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        通过 API 获取 sparse embedding
        """
        async with httpx.AsyncClient(timeout=self.sparse_timeout) as client:
            response = await client.post(
                f"{self.model_server_url}/v1/sparse_embeddings",
                json={"input": texts}
            )
            response.raise_for_status()
            result = response.json()

        # 解析 sparse vectors
        sparse_vecs = []
        for item in result.get("data", []):
            sparse_vecs.append({
                "indices": item["indices"],
                "values": item["values"],
            })
        return sparse_vecs

    async def ensure_collection(self):
        """创建集合 - 同时配置 dense 和 sparse vectors"""
        if not await self.client.collection_exists(config.qdrant_collection):
            self.client.create_collection(
                collection_name=config.qdrant_collection,
                # Dense vectors 配置
                vectors_config={
                    "dense": VectorParams(
                        size=config.qdrant_collection_size,
                        distance=Distance.COSINE
                    )
                },
                # Sparse vectors 配置
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF  # 启用 IDF 加权
                    )
                }
            )
        logger.info(f"qdrant collections已创建! (dense + sparse)")

    async def search(self, kb_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索: Dense + Sparse + RRF 融合
        """
        logger.info(f"混合检索：kb_id={kb_id}, query={query[:30]}, top_k={top_k}")
        await self.ensure_collection()

        # Dense query vector (通过 API)
        dense_vec = await self.embeddings.aembed_query(query)

        # Sparse query vector (通过 API)
        sparse_result = await self._encode_sparse([query])
        sparse_vec = sparse_result[0] if sparse_result else {"indices": [], "values": []}

        # Qdrant 原生混合检索 + 内置 RRF
        resp = await self.client.query_points(
            collection_name=config.qdrant_collection,
            query=Query(
                prefetch=[
                    # Dense 检索
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=top_k * 2,
                    ),
                    # Sparse 检索
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=top_k * 2,
                    ),
                ],
                fusion=Fusion.RRF,  # 内置 RRF 融合
                limit=top_k,
            ),
            query_filter=Filter(
                must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))]
            ),
            with_payload=True,
        )

        results = []
        for p in resp.points:
            payload = p.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "score": round(float(p.score), 4),
                "metadata": {
                    "kb_id": payload.get("kb_id", kb_id),
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "file_name": payload.get("file_name", ""),
                }
            })
        return results

    async def add_vector(self, kb_id: str, doc_id: str, texts: List[str], file_name: str = "") -> List[str]:
        """添加向量 - 同时写入 dense 和 sparse"""
        if not texts:
            return []
        await self.ensure_collection()

        # Dense vectors (通过 API)
        dense_vectors = await self.embeddings.aembed_documents(texts)

        # Sparse vectors (通过 API)
        sparse_results = await self._encode_sparse(texts)

        point_ids = [str(uuid.uuid4()) for _ in texts]
        points = []
        for pid, text, dense, sparse, idx in zip(
            point_ids, texts, dense_vectors, sparse_results, range(len(texts))
        ):
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": dense,
                    "sparse": sparse,
                },
                payload={
                    "kb_id": kb_id,
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "file_name": file_name,
                    "text": text,
                }
            ))

        await self.client.upsert(
            collection_name=config.qdrant_collection,
            points=points
        )
        logger.info(f"混合向量入库: kb_id={kb_id} doc_id={doc_id} count={len(points)}")
        return point_ids

    async def delete_by_doc_id(self, kb_id: str, doc_id: str) -> int:
        q_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        result = await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted = getattr(result, "status", "")
        logger.info(f"删除文档向量: kb_id={kb_id} doc_id={doc_id} status={deleted}")
        return 1

    async def delete_by_kb_id(self, kb_id: str) -> int:
        q_filter = Filter(must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))])
        result = await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted = getattr(result, "status", "")
        logger.info(f"删除知识库向量: kb_id={kb_id} status={deleted} ")
        return 1

    async def count(self, kb_id: str) -> int:
        try:
            q_filter = Filter(must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))])
            result = await self.client.count(
                collection_name=config.qdrant_collection,
                count_filter=q_filter,
                exact=True
            )
            return int(result.count)
        except Exception as e:
            logger.warning(f"向量计数失败: kb_id={kb_id} error={e}")
            return 0
