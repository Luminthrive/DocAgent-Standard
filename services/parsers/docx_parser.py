from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata


@register_parser(".docx")
class DocxParser(BaseParser):
    """Word 文档解析器 — 使用 Unstructured 保留标题/段落/表格结构"""

    # Unstructured 的标题/小标题类别
    _HEADING_CATEGORIES = {"Title", "Heading", "Subheading"}

    def parse(self, file_path: str) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()

        chunks = []
        heading_stack = []  # 跟踪标题层级，构建 section_path

        for doc in docs:
            category = doc.metadata.get("category", "uncategorized")
            text = doc.page_content.strip()
            if not text:
                continue

            is_heading = category in self._HEADING_CATEGORIES

            # 维护标题栈（遇到新标题时更新对应层级）
            if is_heading:
                # 简化：新标题替换栈顶
                heading_stack = [text]
                section_path = text
            else:
                section_path = " > ".join(heading_stack) if heading_stack else ""

            location_ref = f"章节：{section_path}" if section_path else ""

            meta = build_base_metadata(
                file_path, ".docx", chunk_index=len(chunks), total_chunks=0,
                location_ref=location_ref,
                section_path=section_path,
                extra={
                    "element_type": category,
                    "heading_text": text if is_heading else "",
                },
            )
            chunks.append(Document(page_content=text, metadata=meta))

        # 回填 total_chunks
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        logger.info(f"DOCX 解析完成: {len(chunks)} elements")
        return chunks

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
