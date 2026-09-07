from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata

# 大行按行分块的阈值（与 ExcelParser 一致）
LARGE_TABLE_THRESHOLD = 50
ROWS_PER_CHUNK = 50

# 非表格元素的二级分块器
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)


@register_parser(".docx")
class DocxParser(BaseParser):
    """Word 文档解析器 — 标题/段落按元素拆分，表格走 Excel 策略（不切碎）"""

    _HEADING_CATEGORIES = {"Title", "Heading", "Subheading"}
    _TABLE_CATEGORIES = {"Table"}

    def parse(self, file_path: str) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()

        chunks = []
        heading_stack = []

        for doc in docs:
            category = doc.metadata.get("category", "uncategorized")
            text = doc.page_content.strip()
            if not text:
                continue

            is_heading = category in self._HEADING_CATEGORIES
            is_table = category in self._TABLE_CATEGORIES

            # 维护标题栈
            if is_heading:
                heading_stack = [text]
                section_path = text
            else:
                section_path = " > ".join(heading_stack) if heading_stack else ""

            location_ref = f"章节：{section_path}" if section_path else ""

            # 表格元素：走 Excel 策略（转 markdown，按行分块）
            if is_table:
                table_chunks = self._handle_table(
                    text, file_path, len(chunks), section_path, location_ref
                )
                chunks.extend(table_chunks)
            else:
                # 非表格元素：直接作为一个 chunk，交给 _TEXT_SPLITTER 二次分割
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

        # 二级：对非表格的超长 chunk 递归分割
        non_table_chunks = [c for c in chunks if c.metadata.get("element_type") != "Table"]
        table_chunks = [c for c in chunks if c.metadata.get("element_type") == "Table"]

        split_non_table = _TEXT_SPLITTER.split_documents(non_table_chunks) if non_table_chunks else []
        final_chunks = split_non_table + table_chunks  # 表格保持完整，放在后面

        # 回填 chunk_index 和 total_chunks
        total = len(final_chunks)
        for i, c in enumerate(final_chunks):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = total

        logger.info(f"DOCX 解析完成: {len(final_chunks)} chunks (含 {len(table_chunks)} 个表格)")
        return final_chunks

    def get_splitter(self):
        """parse() 已完成全部分块"""
        return None

    @staticmethod
    def _handle_table(
        html_text: str, file_path: str, chunk_index: int,
        section_path: str, location_ref: str,
    ) -> List[Document]:
        """将 Word 表格转为 markdown，按行分块（与 ExcelParser 策略一致）"""
        try:
            tables = pd.read_html(html_text)
            if not tables:
                return []
            df = tables[0]  # 取第一个表格
        except Exception:
            # 解析失败：作为纯文本返回
            meta = build_base_metadata(
                file_path, ".docx", chunk_index=chunk_index, total_chunks=0,
                location_ref=location_ref, section_path=section_path,
                extra={"element_type": "Table", "_parsed_raw": html_text},
            )
            return [Document(page_content=html_text, metadata=meta)]

        total_rows = len(df)
        docs = []

        if total_rows < LARGE_TABLE_THRESHOLD:
            # 小表：整表一个 chunk
            table_md = df.to_markdown(index=False)
            text = DocxParser._df_to_text(df)
            meta = build_base_metadata(
                file_path, ".docx", chunk_index=chunk_index, total_chunks=0,
                location_ref=location_ref, section_path=section_path,
                extra={
                    "element_type": "Table",
                    "row_range": f"1-{total_rows}",
                    "_parsed_raw": table_md,
                },
            )
            docs.append(Document(page_content=text, metadata=meta))
        else:
            # 大表：按行分块，每块带表头
            header_text = " | ".join(str(c) for c in df.columns)
            separator = " | ".join("---" for _ in df.columns)

            for start in range(0, total_rows, ROWS_PER_CHUNK):
                end = min(start + ROWS_PER_CHUNK, total_rows)
                chunk_df = df.iloc[start:end]
                table_md = chunk_df.to_markdown(index=False)
                text = f"{header_text}\n{separator}\n{DocxParser._df_to_text(chunk_df)}"

                meta = build_base_metadata(
                    file_path, ".docx", chunk_index=chunk_index + len(docs), total_chunks=0,
                    location_ref=f"{location_ref} 行{start + 1}-{end}" if location_ref else f"行{start + 1}-{end}",
                    section_path=section_path,
                    extra={
                        "element_type": "Table",
                        "row_range": f"{start + 1}-{end}",
                        "_parsed_raw": table_md,
                    },
                )
                docs.append(Document(page_content=text, metadata=meta))

        return docs

    @staticmethod
    def _df_to_text(df: pd.DataFrame) -> str:
        """DataFrame → 纯文本（用于 Embedding）"""
        lines = []
        header = " | ".join(str(c) for c in df.columns)
        lines.append(header)
        lines.append(" | ".join("---" for _ in df.columns))
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values))
        return "\n".join(lines)
