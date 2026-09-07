#向量处理服务 - 支持混合检索 (Dense + Sparse + RRF)
import uuid
from typing import List, Dict, Any

import httpx
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
    Dense + Sparse 均来自 BGE-M3 (单模型双路输出)
    融合: Qdrant 内置 RRF
    """

    def __init__(self, vector_client):
        self.client = vector_client
        self.model_server_url = config.model_server_url
        self.request_timeout = 60.0

    async def _encode(self, texts: List[str]) -> Dict[str, Any]:
        """
        通过 BGE-M3 API 获取 Dense + Sparse 向量 (单次调用)
        返回: {"dense": List[List[float]], "sparse": List[Dict]}
        """
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(
                f"{self.model_server_url}/v1/embeddings",
                json={"input": texts}
            )
            response.raise_for_status()
            result = response.json()

        dense_vecs = []
        sparse_vecs = []
        for item in result.get("data", []):
            dense_vecs.append(item["embedding"])
            sparse_vecs.append({
                "indices": item["sparse_indices"],
                "values": item["sparse_values"],
            })
        return {"dense": dense_vecs, "sparse": sparse_vecs}

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
        返回: [{"score": ..., "metadata": {...}, "content": {"chunk_text": ..., "parsed_raw": ...}}, ...]
        """
        logger.info(f"混合检索：kb_id={kb_id}, query={query[:30]}, top_k={top_k}")
        await self.ensure_collection()

        # BGE-M3 单次调用获取 Dense + Sparse
        encoded = await self._encode([query])
        dense_vec = encoded["dense"][0]
        sparse_vec = encoded["sparse"][0]

        # Qdrant 原生混合检索 + 内置 RRF
        resp = await self.client.query_points(
            collection_name=config.qdrant_collection,
            query=Query(
                prefetch=[
                    Prefetch(query=dense_vec, using="dense", limit=top_k * 2),
                    Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
                ],
                fusion=Fusion.RRF,
                limit=top_k,
            ),
            query_filter=Filter(
                must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))]
            ),
            with_payload=True,
        )

        results = []
        for p in resp.points:
            payload = p.payload or {}
            results.append({
                "score": round(float(p.score), 4),
                "metadata": payload.get("metadata", {}),
                "content": payload.get("content", {}),
            })
        return results

    async def add_vector(
        self,
        texts: List[str],
        payloads: List[Dict[str, Any]],
        file_name: str = "",
    ) -> List[str]:
        """
        添加向量 — 同时写入 dense + sparse
        payloads: 每个元素是完整的 {"metadata": {...}, "content": {...}} 结构
        texts: 与 payloads 一一对应的 embedding 文本（即 content.chunk_text）
        """
        if not texts:
            return []
        await self.ensure_collection()

        # BGE-M3 单次调用获取 Dense + Sparse
        encoded = await self._encode(texts)
        dense_vectors = encoded["dense"]
        sparse_results = encoded["sparse"]

        point_ids = [str(uuid.uuid4()) for _ in texts]
        points = []
        for pid, text, dense, sparse, idx, payload in zip(
            point_ids, texts, dense_vectors, sparse_results, range(len(texts)), payloads
        ):
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": dense,
                    "sparse": sparse,
                },
                payload=payload,  # 已是完整的 {metadata: {...}, content: {...}}
            ))

        await self.client.upsert(
            collection_name=config.qdrant_collection,
            points=points
        )
        logger.info(f"混合向量入库: count={len(points)}")
        return point_ids

    async def delete_by_doc_id(self, kb_id: str, doc_id: str) -> int:
        q_filter = Filter(must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))])
        result = await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted = getattr(result, "status", "")
        logger.info(f"删除文档向量: kb_id={kb_id} doc_id={doc_id} status={deleted}")
        return 1

    async def delete_by_kb_id(self, kb_id: str) -> int:
        q_filter = Filter(must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))])
        result = await self.client.delete(
            collection_name=config.qdrant_collection,
            points_selector=q_filter
        )
        deleted = getattr(result, "status", "")
        logger.info(f"删除知识库向量: kb_id={kb_id} status={deleted} ")
        return 1

    async def count(self, kb_id: str) -> int:
        try:
            q_filter = Filter(must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))])
            result = await self.client.count(
                collection_name=config.qdrant_collection,
                count_filter=q_filter,
                exact=True
            )
            return int(result.count)
        except Exception as e:
            logger.warning(f"向量计数失败: kb_id={kb_id} error={e}")
            return 0
