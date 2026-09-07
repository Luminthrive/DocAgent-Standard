#多路查询改写服务
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
from agent_graph.util import get_llm


# 多路改写策略 Prompt
REWRITE_STRATEGIES = {
    "keyword": """请将以下用户问题改写为更适合检索的形式：
原始问题: {original_query}
要求：去掉口语化表达，提取核心关键词，输出改写后的问题（只输出问题）：""",

    "decompose": """请将以下问题改写为更具体、更精确的检索问题：
原始问题: {original_query}
要求：输出一个更具体、更精确的检索问题（只输出问题）：""",

    "synonym": """请将以下问题用同义词/近义词替换关键术语进行改写：
原始问题: {original_query}
要求：用同义词/近义词替换关键术语，输出改写后的问题（只输出问题）：""",

    "expand": """请将以下问题扩展相关术语进行改写：
原始问题: {original_query}
要求：添加相关术语、上下位词，输出改写后的问题（只输出问题）：""",
}


class QueryRewriteService:
    """
    多路查询改写服务
    - 支持多种改写策略并行执行
    - 返回多个改写结果供后续检索使用
    """

    def __init__(self):
        self.strategies = REWRITE_STRATEGIES
        logger.info(f"QueryRewriteService初始化: {len(self.strategies)}种策略")

    async def _rewrite_single(
        self,
        strategy_name: str,
        strategy_prompt: str,
        query: str,
    ) -> Dict[str, Any]:
        """单策略改写"""
        try:
            prompt = strategy_prompt.format(original_query=query)
            resp = await get_llm(temperature=0.3).ainvoke(prompt)
            rewritten = str(resp.content).strip()

            # 清理：只保留问题本身，去掉多余内容
            if rewritten and "\n" in rewritten:
                rewritten = rewritten.split("\n")[0].strip()

            logger.debug(f"策略[{strategy_name}] 改写: '{query[:30]}' → '{rewritten[:30]}'")
            return {
                "strategy": strategy_name,
                "original": query,
                "rewritten": rewritten,
                "success": bool(rewritten),
            }
        except Exception as e:
            logger.warning(f"策略[{strategy_name}] 改写失败: {e}")
            return {
                "strategy": strategy_name,
                "original": query,
                "rewritten": query,  # 失败时返回原 query
                "success": False,
            }

    async def multi_rewrite(
        self,
        query: str,
        strategies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        多路改写：并行执行多种策略

        Args:
            query: 原始查询
            strategies: 要使用的策略列表，None 则使用全部

        Returns:
            改写结果列表
        """
        if strategies is None:
            strategies = list(self.strategies.keys())

        # 过滤有效策略
        valid_strategies = {
            k: v for k, v in self.strategies.items()
            if k in strategies
        }

        logger.info(f"多路改写: query='{query[:30]}' strategies={list(valid_strategies.keys())}")

        # 并行执行所有策略
        tasks = [
            self._rewrite_single(name, prompt, query)
            for name, prompt in valid_strategies.items()
        ]
        results = await asyncio.gather(*tasks)

        # 过滤成功的结果
        successful = [r for r in results if r["success"] and r["rewritten"] != query]

        # 如果所有策略都失败，返回原 query
        if not successful:
            logger.warning("所有改写策略失败，使用原始查询")
            return [{"strategy": "original", "original": query, "rewritten": query, "success": True}]

        # 去重（避免多个策略生成相同结果）
        seen = {query}  # 包含原始 query
        unique = []
        for r in successful:
            if r["rewritten"] not in seen:
                seen.add(r["rewritten"])
                unique.append(r)

        logger.info(f"多路改写完成: {len(unique)}个不同结果")
        return unique

    async def select_best(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        vector_service,
        kb_id: str,
    ) -> str:
        """
        选择最优改写结果：基于检索质量评分

        Args:
            query: 原始查询
            candidates: 候选改写结果
            vector_service: 向量服务
            kb_id: 知识库ID

        Returns:
            最优的改写 query
        """
        if not candidates:
            return query

        if len(candidates) == 1:
            return candidates[0]["rewritten"]

        # 并行检索所有候选
        async def score_candidate(candidate):
            rewritten = candidate["rewritten"]
            try:
                docs = await vector_service.search(kb_id, rewritten, top_k=3)
                if not docs:
                    return {"candidate": candidate, "score": 0.0}
                # 简单评分：平均 rerank 分数
                avg_score = sum(d.get("rerank_score", d.get("score", 0)) for d in docs) / len(docs)
                return {"candidate": candidate, "score": avg_score}
            except Exception:
                return {"candidate": candidate, "score": 0.0}

        tasks = [score_candidate(c) for c in candidates]
        scored = await asyncio.gather(*tasks)

        # 选择最高分
        best = max(scored, key=lambda x: x["score"])
        logger.info(
            f"选择最优改写: strategy={best['candidate']['strategy']} "
            f"score={best['score']:.3f} query='{best['candidate']['rewritten'][:30]}'"
        )

        return best["candidate"]["rewritten"]
