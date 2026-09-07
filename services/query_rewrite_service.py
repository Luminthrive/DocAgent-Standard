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

# 工具内部多路召回默认使用的策略（2-3个，控制 QPS）
DEFAULT_RECALL_STRATEGIES = ["keyword", "decompose"]


class QueryRewriteService:
    """
    多路查询改写服务（工具内部使用）
    - 每次 rag_search 自动执行
    - 并行生成 2-3 个改写结果
    - 返回多个 query 供后续并行检索
    """

    def __init__(self):
        self.strategies = REWRITE_STRATEGIES
        logger.info(f"QueryRewriteService初始化: 默认使用{DEFAULT_RECALL_STRATEGIES}")

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

            # 清理：只保留问题本身
            if rewritten and "\n" in rewritten:
                rewritten = rewritten.split("\n")[0].strip()

            return {
                "strategy": strategy_name,
                "rewritten": rewritten,
                "success": bool(rewritten),
            }
        except Exception as e:
            logger.warning(f"策略[{strategy_name}]改写失败: {e}")
            return {
                "strategy": strategy_name,
                "rewritten": query,
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
            strategies: 策略列表，None 使用默认策略

        Returns:
            改写结果列表 [{"strategy": str, "rewritten": str}]
        """
        if strategies is None:
            strategies = DEFAULT_RECALL_STRATEGIES

        valid_strategies = {
            k: v for k, v in self.strategies.items()
            if k in strategies
        }

        # 并行执行
        tasks = [
            self._rewrite_single(name, prompt, query)
            for name, prompt in valid_strategies.items()
        ]
        results = await asyncio.gather(*tasks)

        # 过滤成功且不同的结果
        successful = [r for r in results if r["success"] and r["rewritten"] != query]

        if not successful:
            return [{"strategy": "original", "rewritten": query, "success": True}]

        # 去重
        seen = {query}
        unique = []
        for r in successful:
            if r["rewritten"] not in seen:
                seen.add(r["rewritten"])
                unique.append(r)

        logger.debug(f"多路改写: {len(unique)}个结果")
        return unique
