import base64
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata
from config import config

# 文本密度阈值：每页文本少于此字数判定为图片型
OCR_TEXT_THRESHOLD = 50

# Vision OCR 提示词
_OCR_PROMPT = (
    "请识别并提取这张图片中的所有文字内容。"
    "要求：保持原始排版格式，包括标题、段落、列表、表格结构。"
    "如果是表格，用 Markdown 表格格式输出。"
    "只输出识别到的文字内容，不要添加额外说明。"
)

# 懒加载的 Vision LLM 实例
_vision_llm: ChatOpenAI | None = None


def _get_vision_llm() -> ChatOpenAI:
    """懒加载 Vision LLM（避免未配置时报错）"""
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = ChatOpenAI(
            model=config.vision_model,
            base_url=config.vision_api_url,
            api_key=config.vision_api_key,
            max_tokens=4096,
            temperature=0.0,
        )
    return _vision_llm


@register_parser(".pdf")
class PDFParser(BaseParser):
    """PDF 解析器 — 文本型直接提取，图片型调 Vision LLM OCR"""

    def parse(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 提取 PDF 级元信息
        pdf_title = ""
        if docs and docs[0].metadata.get("title"):
            pdf_title = docs[0].metadata["title"]

        chunks = []
        for page_doc in docs:
            page_num = page_doc.metadata.get("page", 0)
            text = page_doc.page_content
            text_len = len(text.strip())
            is_ocr = text_len < OCR_TEXT_THRESHOLD

            # 图片型页面：渲染 → Vision LLM OCR
            if is_ocr:
                logger.info(f"PDF 图片型页面: page={page_num}, text_len={text_len}，调用 Vision OCR")
                ocr_text = self._ocr_page(file_path, page_num)
                if ocr_text:
                    text = ocr_text
                    logger.info(f"Vision OCR 成功: page={page_num}, ocr_len={len(ocr_text)}")
                else:
                    logger.warning(f"Vision OCR 未返回内容: page={page_num}")

            meta = build_base_metadata(
                file_path, ".pdf", chunk_index=len(chunks), total_chunks=0,
                doc_title=pdf_title,
                location_ref=f"第{page_num + 1}页",
                extra={
                    "page": page_num,
                    "is_ocr": is_ocr,
                    "total_pages": len(docs),
                },
            )
            chunks.append(Document(page_content=text, metadata=meta))

        # 回填 total_chunks
        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)

        return chunks

    def get_splitter(self):
        """PDF 已按页切分，不再二次切分"""
        return None

    @staticmethod
    def _ocr_page(file_path: str, page_num: int) -> str:
        """渲染 PDF 页面为图片 → 调 Vision LLM 识别文字"""
        # 1. PyMuPDF 渲染页面为 PNG
        try:
            import fitz
            pdf_doc = fitz.open(file_path)
            if page_num >= len(pdf_doc):
                pdf_doc.close()
                return ""
            page = pdf_doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            pdf_doc.close()
        except Exception as e:
            logger.error(f"PDF 页面渲染失败: page={page_num} error={e}")
            return ""

        # 2. 调 Vision LLM（ChatOpenAI，原生支持图片）
        if not config.vision_api_key:
            logger.warning("VISION_API_KEY 未配置，跳过 Vision OCR")
            return ""

        try:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            llm = _get_vision_llm()
            response = llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    }},
                ])
            ])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Vision OCR 失败: page={page_num} error={e}")
            return ""
