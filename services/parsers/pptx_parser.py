from typing import List

from pptx import Presentation
from langchain_core.documents import Document
from loguru import logger

from services.parsers.base_parser import BaseParser, ParseResult
from services.parsers import register_parser


@register_parser(".pptx")
class PPTXParser(BaseParser):
    """PPT 解析器 — 每张幻灯片一个 chunk"""

    def parse(self, file_path: str) -> ParseResult:
        prs = Presentation(file_path)
        docs = []

        for idx, slide in enumerate(prs.slides):
            text_parts = []
            slide_title = ""
            has_notes = False

            # 提取幻灯片上的文字
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        para_text = para.text.strip()
                        if para_text:
                            text_parts.append(para_text)
                            # 第一个有文字的 shape 通常视为标题
                            if not slide_title:
                                slide_title = para_text

            # 提取备注
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                notes_text = notes_frame.text.strip()
                if notes_text:
                    has_notes = True
                    text_parts.append(f"\n--- 备注 ---\n{notes_text}")

            page_content = "\n".join(text_parts)
            if not page_content.strip():
                continue

            docs.append(Document(
                page_content=page_content,
                metadata={
                    "file_type": ".pptx",
                    "slide_number": idx + 1,
                    "slide_title": slide_title,
                    "has_notes": has_notes,
                },
            ))

        logger.info(f"PPTX 解析完成: {len(docs)} slides")
        return ParseResult(
            documents=docs,
            metadata={
                "file_type": ".pptx",
                "total_slides": len(docs),
            },
        )

    def get_splitter(self):
        """PPT 不使用 LangChain TextSplitter，返回 None"""
        return None
