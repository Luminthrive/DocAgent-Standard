from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser


@register_parser(".docx")
class DocxParser(BaseParser):
    """Word 文档解析器 — 使用 Unstructured 保留标题/段落/表格结构"""

    def parse(self, file_path: str) -> ParseResult:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()

        # 为每个 chunk 标注 element_type
        for doc in docs:
            element_type = doc.metadata.get("category", "uncategorized")
            doc.metadata.update({
                "file_type": ".docx",
                "element_type": element_type,
                "heading_text": doc.metadata.get("category", ""),
            })

        logger.info(f"DOCX 解析完成: {len(docs)} elements")
        return ParseResult(
            documents=docs,
            metadata={"file_type": ".docx", "total_elements": len(docs)},
        )

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
