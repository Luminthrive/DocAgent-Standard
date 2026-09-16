#向量处理服务 - 支持混合检索 (Dense + Sparse + RRF)
import asyncio
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
        # 分批配置：从环境变量读取，默认 64
        self._batch_size = getattr(config, 'vector_batch_size', 64)
        # 重试配置
        self._max_retries = 3
        self._base_retry_delay = 2.0

    async def _encode(self, texts: List[str], batch_idx: int = 0, total_batches: int = 1) -> Dict[str, Any]:
        """
        通过 BGE-M3 API 获取 Dense + Sparse 向量 (单次调用)
        返回: {"dense": List[List[float]], "sparse": List[Dict]}
        自适应超时：根据批次大小动态调整超时时间
        """
        # 自适应超时：基础超时 + 每 16 个文本增加 10s
        adaptive_timeout = self.request_timeout + (len(texts) / 16) * 10.0
        logger.debug(f"编码批次 {batch_idx+1}/{total_batches}: count={len(texts)}, timeout={adaptive_timeout:.1f}s")

        last_error = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=adaptive_timeout) as client:
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
            except Exception as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    delay = self._base_retry_delay * (2 ** attempt)
                    logger.warning(f"编码失败 (attempt {attempt+1}/{self._max_retries}): {e}, {delay:.1f}s 后重试")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"编码最终失败: {e}")
                    raise

    async def ensure_collection(self):
        """创建集合 - 同时配置 dense 和 sparse vectors"""
        if not await self.client.collection_exists(config.qdrant_collection):
            await self.client.create_collection(
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
        添加向量 — 分批写入 dense + sparse，避免超大文档超时
        payloads: 每个元素是完整的 {"metadata": {...}, "content": {...}} 结构
        texts: 与 payloads 一一对应的 embedding 文本（即 content.chunk_text）
        """
        if not texts:
            return []
        await self.ensure_collection()

        all_point_ids = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_idx in range(0, len(texts), self._batch_size):
            batch_texts = texts[batch_idx:batch_idx + self._batch_size]
            batch_payloads = payloads[batch_idx:batch_idx + self._batch_size]

            # BGE-M3 编码（带自适应超时和重试）
            encoded = await self._encode(batch_texts, batch_idx // self._batch_size + 1, total_batches)
            dense_vectors = encoded["dense"]
            sparse_results = encoded["sparse"]

            point_ids = [str(uuid.uuid4()) for _ in batch_texts]
            points = []
            for pid, text, dense, sparse, idx, payload in zip(
                point_ids, batch_texts, dense_vectors, sparse_results, range(len(batch_texts)), batch_payloads
            ):
                points.append(PointStruct(
                    id=pid,
                    vector={
                        "dense": dense,
                        "sparse": sparse,
                    },
                    payload=payload,  # 已是完整的 {metadata: {...}, content: {...}}
                ))

            # upsert 带重试
            last_error = None
            for attempt in range(self._max_retries):
                try:
                    await self.client.upsert(
                        collection_name=config.qdrant_collection,
                        points=points
                    )
                    break
                except Exception as e:
                    last_error = e
                    if attempt < self._max_retries - 1:
                        delay = self._base_retry_delay * (2 ** attempt)
                        logger.warning(f"upsert 失败 (batch {batch_idx//self._batch_size+1}, attempt {attempt+1}): {e}, {delay:.1f}s 后重试")
                        await asyncio.sleep(delay)
                    else:
                        raise

            all_point_ids.extend(point_ids)
            logger.info(f"混合向量入库: batch={batch_idx//self._batch_size+1}/{total_batches}, count={len(points)}")

        logger.info(f"混合向量入库完成: total={len(all_point_ids)}")
        return all_point_ids

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
