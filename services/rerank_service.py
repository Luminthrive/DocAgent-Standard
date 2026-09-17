#Rerank 精排服务 - 通过 API 调用模型服务
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
from config import config


class RerankService:
    """
    Rerank 服务 - 通过 API 调用模型服务器
    """

    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.model_server_url
        self.model = config.rerank_model
        self.timeout = 30.0
        logger.info(f"RerankService初始化: base_url={self.base_url}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行精排

        Args:
            query: 用户查询
            documents: 混合检索返回的文档列表
            top_k: 返回前 top_k 个结果
            score_threshold: 分数阈值（[0,1] 归一化相关概率，模型服务端 CrossEncoder.predict 已做 Sigmoid），低于此分数的文档被过滤

        Returns:
            精排后的文档列表
        """
        if not documents:
            return []

        # 提取文档文本
        doc_texts = [doc.get("content", {}).get("chunk_text", "")[:512] for doc in documents]

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/rerank",
                    json={
                        "query": query,
                        "documents": doc_texts,
                        "top_n": len(doc_texts),  # 全部返回，后面过滤
                        "model": self.model,
                    }
                )
                response.raise_for_status()
                result = response.json()

            # 构建 index -> rerank_score 映射
            score_map = {
                item["index"]: item["score"]
                for item in result.get("results", [])
            }

            # 附加 rerank 分数到原始文档
            for i, doc in enumerate(documents):
                doc["rerank_score"] = round(score_map.get(i, 0.0), 4)

            # 按 rerank 分排序，过滤低分
            ranked = sorted(
                [d for d in documents if d["rerank_score"] > score_threshold],
                key=lambda x: x["rerank_score"],
                reverse=True,
            )

            logger.info(
                f"Rerank完成: query='{query[:30]}' "
                f"input={len(documents)} output={len(ranked[:top_k])}"
            )

            return ranked[:top_k]

        except httpx.HTTPError as e:
            logger.error(f"Rerank API调用失败: {e}")
            # 降级：返回原始排序
            return documents[:top_k]
        except Exception as e:
            logger.error(f"Rerank异常: {e}")
            return documents[:top_k]
