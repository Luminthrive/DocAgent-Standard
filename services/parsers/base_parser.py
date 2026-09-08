from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    async def parse(self, file_path: str) -> List[Document]:
        """解析文档并分块，返回最终的 Document 列表"""
