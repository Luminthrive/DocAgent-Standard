from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser


@register_parser(".txt")
class TextParser(BaseParser):
    """纯文本解析器"""

    def parse(self, file_path: str) -> ParseResult:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        total_chars = sum(len(d.page_content) for d in docs)
        for doc in docs:
            doc.metadata.update({
                "file_type": ".txt",
                "total_chars": total_chars,
            })
        return ParseResult(
            documents=docs,
            metadata={"file_type": ".txt", "total_chars": total_chars},
        )

    def get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
