from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser

# 文本密度阈值：每页文本少于此字数判定为图片型
OCR_TEXT_THRESHOLD = 50


@register_parser(".pdf")
class PDFParser(BaseParser):
    """PDF 解析器 — 自动检测文本型/图片型"""

    def parse(self, file_path: str) -> ParseResult:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 提取 PDF 级元数据
        pdf_title = ""
        if docs and docs[0].metadata.get("title"):
            pdf_title = docs[0].metadata["title"]

        # 检测每页是否为图片型（文本过少）
        for doc in docs:
            text_len = len(doc.page_content.strip())
            is_ocr = text_len < OCR_TEXT_THRESHOLD
            doc.metadata.update({
                "file_type": ".pdf",
                "page": doc.metadata.get("page", 0),
                "is_ocr": is_ocr,
                "pdf_title": pdf_title,
            })
            if is_ocr:
                logger.info(f"PDF 图片型页面: page={doc.metadata['page']}, text_len={text_len}")

        return ParseResult(
            documents=docs,
            metadata={
                "file_type": ".pdf",
                "pdf_title": pdf_title,
                "total_pages": len(docs),
            },
        )

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
