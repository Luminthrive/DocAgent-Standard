from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata

# 二级分块器
_SECONDARY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)


@register_parser(".txt")
class TextParser(BaseParser):
    """纯文本解析器 — 按段落加载，递归字符分块"""

    def _parse_sync(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "file_type": "txt",
                "source": file_path,
            })

        # 分块
        chunks = _SECONDARY_SPLITTER.split_documents(docs)

        # 回填 chunk_index 和 total_chunks
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = total

        return chunks
