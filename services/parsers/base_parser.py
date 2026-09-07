from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


class BaseParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """加载文档，返回带元数据的 Document 列表（文件级元数据已写入每个 Document.metadata）"""

    @abstractmethod
    def get_splitter(self) -> TextSplitter:
        """返回该类型适用的分块器"""
