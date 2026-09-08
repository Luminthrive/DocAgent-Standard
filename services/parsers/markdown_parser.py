from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata


# 按标题层级切分的配置
MARKDOWN_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# 二级分块器（对超长标题段落再细分）
_SECONDARY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "，", " "],
)


@register_parser(".md")
class MarkdownParser(BaseParser):
    """Markdown 解析器 — 一级按标题层级切分，二级按长度细分"""

    async def parse(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        # 一级切分：按 # / ## / ### 标题层级拆分
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT,
            strip_headers=False,
        )
        header_chunks = []
        for doc in docs:
            header_chunks.extend(header_splitter.split_text(doc.page_content))

        # 构建带元数据的 Document 列表
        result = []
        for i, chunk in enumerate(header_chunks):
            # 从 MarkdownHeaderTextSplitter 的 metadata 提取标题路径
            header_path = []
            for level in ["h1", "h2", "h3"]:
                if level in chunk.metadata:
                    header_path.append(chunk.metadata[level])
            section_path = " > ".join(header_path) if header_path else ""
            location_ref = header_path[-1] if header_path else ""

            text = chunk.page_content
            meta = build_base_metadata(
                file_path, ".md", chunk_index=i, total_chunks=0,
                location_ref=location_ref,
                section_path=section_path,
            )
            # 额外保留 header_path 用于精确溯源
            meta["header_path"] = header_path
            result.append(Document(page_content=text, metadata=meta))

        # 二级切分：对超长标题段落再细分
        final = _SECONDARY_SPLITTER.split_documents(result)

        # 回填 chunk_index 和 total_chunks（二级切分后序号会变）
        total = len(final)
        for i, c in enumerate(final):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = total

        return final
