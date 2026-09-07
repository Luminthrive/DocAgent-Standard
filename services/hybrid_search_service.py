#混合检索编排服务 - 多路改写 + 并行检索 + 合并去重 + Rerank精排
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
from config import config


class HybridSearchService:
    """
    混合检索编排服务

    支持两种模式：
    1. 单路检索: 直接使用原始 query 检索
    2. 多路召回: 多策略改写 → 并行检索 → 合并去重 → Rerank精排
    """

    def __init__(
        self,
        vector_service,
        rerank_service,
        query_rewrite_service,
    ):
        self.vector_service = vector_service
        self.rerank_service = rerank_service
        self.query_rewrite_service = query_rewrite_service
        logger.info("HybridSearchService初始化完成")

    async def search(
        self,
        query: str,
        kb_id: str,
        top_k: int = 5,
        use_multi_rewrite: bool = True,
        multi_rewrite_strategies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索入口

        Args:
            query: 用户查询
            kb_id: 知识库ID
            top_k: 返回结果数
            use_multi_rewrite: 是否启用多路改写
            multi_rewrite_strategies: 指定改写策略列表，None=全部

        Returns:
            检索结果列表
        """
        if use_multi_rewrite:
            return await self._multi_recall_search(query, kb_id, top_k, multi_rewrite_strategies)
        else:
            return await self._single_search(query, kb_id, top_k)

    async def _single_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """单路检索"""
        logger.info(f"单路检索: query='{query[:30]}' kb_id={kb_id}")
        docs = await self.vector_service.search(kb_id, query, top_k)
        return docs

    async def _multi_recall_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        strategies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        多路召回检索

        流程:
        1. 多策略改写生成多个 query
        2. 并行检索所有 query
        3. 合并去重
        4. Rerank精排
        """
        logger.info(f"多路召回: query='{query[:30]}' kb_id={kb_id}")

        # ═══════════════════════════════════════════════════════════
        # Step 1: 多策略改写
        # ═══════════════════════════════════════════════════════════
        rewrite_results = await self.query_rewrite_service.multi_rewrite(
            query=query,
            strategies=strategies,
        )

        # 改写结果列表（包含原始 query）
        queries = [query] + [r["rewritten"] for r in rewrite_results]
        # 去重
        queries = list(dict.fromkeys(queries))

        logger.info(f"改写结果: {len(queries)}个查询")

        # ═══════════════════════════════════════════════════════════
        # Step 2: 并行检索
        # ═══════════════════════════════════════════════════════════
        recall_per_query = max(3, top_k)  # 每个 query 召回的数量

        async def search_single(q: str) -> List[Dict[str, Any]]:
            try:
                docs = await self.vector_service.search(kb_id, q, recall_per_query)
                # 标记来源
                for doc in docs:
                    doc["source_query"] = q
                return docs
            except Exception as e:
                logger.warning(f"检索失败 query='{q[:20]}': {e}")
                return []

        tasks = [search_single(q) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # ═══════════════════════════════════════════════════════════
        # Step 3: 合并去重
        # ═══════════════════════════════════════════════════════════
        merged = self._merge_and_dedup(all_results)

        logger.info(f"合并去重: {sum(len(r) for r in all_results)} → {len(merged)}")

        if not merged:
            return []

        # ═══════════════════════════════════════════════════════════
        # Step 4: Rerank精排
        # ═══════════════════════════════════════════════════════════
        reranked = await self.rerank_service.rerank(
            query=query,
            documents=merged,
            top_k=top_k,
        )

        logger.info(f"多路召回完成: {len(reranked)}个结果")
        return reranked

    def _merge_and_dedup(
        self,
        all_results: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        合并多个检索结果并去重

        去重策略：基于文档内容前200字符 + doc_id
        """
        seen_keys = set()
        merged = []

        for results in all_results:
            for doc in results:
                # 生成去重 key
                text = doc.get("text", "")[:200]
                doc_id = doc.get("metadata", {}).get("doc_id", "")
                dedup_key = f"{doc_id}||{text}"

                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    merged.append(doc)

        return merged
