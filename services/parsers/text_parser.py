from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata


@register_parser(".txt")
class TextParser(BaseParser):
    """纯文本解析器 — 按段落加载，递归字符分块"""

    def parse(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "file_type": "txt",
                "source": file_path,
            })
        return docs

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
