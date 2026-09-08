from typing import List

from langchain_core.documents import Document
from loguru import logger

from services.parsers import get_parser


class ParseService:
    """文档解析调度器 — 根据文件类型路由到对应解析策略"""

    async def parse_document(self, file_path: str, file_type: str) -> List[Document]:
        """解析文档并分块，返回最终的 Document 列表"""
        logger.info(f"解析文档: {file_path} type={file_type}")
        parser = get_parser(file_type)
        return await parser.parse(file_path)
