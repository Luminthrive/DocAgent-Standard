from typing import List

import pandas as pd
from langchain_core.documents import Document
from loguru import logger

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser

# 大行按行分块的阈值
LARGE_SHEET_THRESHOLD = 50
ROWS_PER_CHUNK = 50


@register_parser(".xlsx")
class ExcelParser(BaseParser):
    """Excel 解析器 — 每个 Sheet 独立处理，大表按行分块"""

    def parse(self, file_path: str) -> ParseResult:
        docs = []
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                continue

            total_rows = len(df)
            chunk_docs = self._sheet_to_chunks(df, sheet_name, total_rows)
            docs.extend(chunk_docs)

        logger.info(f"XLSX 解析完成: {len(sheet_names)} sheets, {len(docs)} chunks")
        return ParseResult(
            documents=docs,
            metadata={
                "file_type": ".xlsx",
                "total_sheets": len(sheet_names),
                "sheet_names": sheet_names,
            },
        )

    def _sheet_to_chunks(self, df: pd.DataFrame, sheet_name: str, total_rows: int) -> List[Document]:
        """将一个 Sheet 转换为 Document 列表"""
        docs = []

        if total_rows < LARGE_SHEET_THRESHOLD:
            # 小表：整表作为一个 chunk
            text = df.to_markdown(index=False)
            docs.append(Document(
                page_content=text,
                metadata={
                    "file_type": ".xlsx",
                    "sheet_name": sheet_name,
                    "row_range": f"1-{total_rows}",
                    "total_rows": total_rows,
                },
            ))
        else:
            # 大表：按行分块，每块保留表头
            header = df.columns.tolist()
            header_text = " | ".join(str(h) for h in header)
            separator = " | ".join("---" for _ in header)

            for start in range(0, total_rows, ROWS_PER_CHUNK):
                end = min(start + ROWS_PER_CHUNK, total_rows)
                chunk_df = df.iloc[start:end]
                rows_text = chunk_df.to_markdown(index=False)
                text = f"{header_text}\n{separator}\n{rows_text}"

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "file_type": ".xlsx",
                        "sheet_name": sheet_name,
                        "row_range": f"{start + 1}-{end}",
                        "total_rows": total_rows,
                    },
                ))

        return docs

    def get_splitter(self):
        """Excel 不使用 LangChain TextSplitter，返回 None"""
        return None
