

from typing import Any
from loguru import logger
from config import config
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.checkpoint.memory import MemorySaver
async def create_checkpointer(redis_available: bool) -> Any:
    """
    创建 LangGraph Checkpointer。

    Redis 可用：
        AsyncRedisSaver

    Redis 不可用：
        MemorySaver
    """
    try:

        if redis_available and config.REDIS_URL:
            ttl_config = None
            if config.langgraph_checkpoint_ttl > 0:
                ttl_config = {
                    "default_ttl": config.langgraph_checkpoint_ttl // 60,
                }
            saver = AsyncRedisSaver(
                redis_url=config.REDIS_URL,
                ttl=ttl_config,
            )
            await saver.setup()
            logger.info(
                "LangGraph AsyncRedisSaver 初始化成功"
            )
            return saver

    except Exception as e:
        logger.warning(
            f"Redis Checkpointer 初始化失败，降级到 MemorySaver: {e}"
        )

    logger.warning(
        "LangGraph 使用 MemorySaver"
    )

    return MemorySaver()