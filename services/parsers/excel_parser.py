from typing import List

import pandas as pd
from langchain_core.documents import Document
from loguru import logger

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata

# 大行按行分块的阈值
LARGE_SHEET_THRESHOLD = 50
ROWS_PER_CHUNK = 50


@register_parser(".xlsx")
class ExcelParser(BaseParser):
    """Excel 解析器 — 每个 Sheet 独立处理，大表按行分块"""

    def _parse_sync(self, file_path: str) -> List[Document]:
        docs = []
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            if df.empty:
                continue

            total_rows = len(df)
            chunk_docs = self._sheet_to_chunks(file_path, df, sheet_name, total_rows, len(sheet_names))
            docs.extend(chunk_docs)

        logger.info(f"XLSX 解析完成: {len(sheet_names)} sheets, {len(docs)} chunks")
        return docs

    def _sheet_to_chunks(
        self, file_path: str, df: pd.DataFrame, sheet_name: str,
        total_rows: int, total_sheets: int,
    ) -> List[Document]:
        docs = []

        if total_rows < LARGE_SHEET_THRESHOLD:
            # 小表：整表作为一个 chunk
            table_md = df.to_markdown(index=False)
            text = self._df_to_text(df)
            meta = build_base_metadata(
                file_path, ".xlsx", chunk_index=len(docs), total_chunks=0,
                location_ref=f"工作表：{sheet_name}，行1-{total_rows}",
                extra={
                    "sheet_name": sheet_name,
                    "row_range": f"1-{total_rows}",
                    "total_rows": total_rows,
                    "total_sheets": total_sheets,
                    "_parsed_raw": table_md,
                },
            )
            docs.append(Document(page_content=text, metadata=meta))
        else:
            # 大表：按行分块，每块保留表头
            for start in range(0, total_rows, ROWS_PER_CHUNK):
                end = min(start + ROWS_PER_CHUNK, total_rows)
                chunk_df = df.iloc[start:end]
                table_md = chunk_df.to_markdown(index=False)
                text = self._df_to_text(chunk_df)

                meta = build_base_metadata(
                    file_path, ".xlsx", chunk_index=len(docs), total_chunks=0,
                    location_ref=f"工作表：{sheet_name}，行{start + 1}-{end}",
                    extra={
                        "sheet_name": sheet_name,
                        "row_range": f"{start + 1}-{end}",
                        "total_rows": total_rows,
                        "total_sheets": total_sheets,
                        "_parsed_raw": table_md,
                    },
                )
                docs.append(Document(page_content=text, metadata=meta))

        return docs

    @staticmethod
    def _df_to_text(df: pd.DataFrame) -> str:
        """DataFrame → 纯文本（用于 Embedding），保留关键信息"""
        lines = []
        header = " | ".join(str(c) for c in df.columns)
        lines.append(header)
        lines.append(" | ".join("---" for _ in df.columns))
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values))
        return "\n".join(lines)
