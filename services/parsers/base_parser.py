import asyncio
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseParser(ABC):
    """文档解析器抽象基类 — 子类实现同步 _parse_sync，基类统一调度到线程池"""

    @abstractmethod
    def _parse_sync(self, file_path: str) -> List[Document]:
        """同步解析文档并分块（CPU/IO 密集，由基类调度到线程池，避免阻塞事件循环）"""

    async def parse(self, file_path: str) -> List[Document]:
        """对外保持 async 接口：同步解析体丢线程池执行，事件循环不被阻塞"""
        return await asyncio.to_thread(self._parse_sync, file_path)
