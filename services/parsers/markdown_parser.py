from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser


# 按标题层级切分的配置
MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


@register_parser(".md")
class MarkdownParser(BaseParser):
    """Markdown 解析器 — 按标题层级切分"""

    def parse(self, file_path: str) -> ParseResult:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = ".md"
        return ParseResult(
            documents=docs,
            metadata={"file_type": ".md"},
        )

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        """返回二级分块器（对 MarkdownHeaderTextSplitter 产出的超长块再细分）"""
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "，", " "],
        )

    def split_by_headers(self, docs):
        """一级切分：按标题层级拆分，每块携带 header_path"""
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT,
            strip_headers=False,
        )
        split_docs = []
        for doc in docs:
            split_docs.extend(header_splitter.split_text(doc.page_content))
        return split_docs
