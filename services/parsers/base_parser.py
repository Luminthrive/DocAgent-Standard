from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


@dataclass
class ParseResult:
    """解析结果包装"""
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """加载文档，返回带元数据的 Document 列表"""

    @abstractmethod
    def get_splitter(self) -> TextSplitter:
        """返回该类型适用的分块器"""
