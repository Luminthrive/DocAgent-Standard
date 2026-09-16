import asyncio
import base64
from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from services.parsers.base_parser import BaseParser
from services.parsers import register_parser
from services.parsers.metadata_utils import build_base_metadata
from config import config

# 文本密度阈值：每页文本少于此字数判定为图片型
OCR_TEXT_THRESHOLD = 50

# OCR 并发控制：从环境变量读取，默认 3
OCR_SEMAPHORE_LIMIT = getattr(config, 'ocr_concurrency_limit', 3)
# OCR 重试配置
OCR_MAX_RETRIES = 3
OCR_BASE_RETRY_DELAY = 2.0

# 二级分块器（超长页自动分割，短页不受影响）
_SECONDARY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)

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
            api_key=SecretStr(config.vision_api_key or ""),
            temperature=0.0,
            model_kwargs={"max_tokens": 4096},
        )
    return _vision_llm


def _render_page(file_path: str, page_num: int) -> bytes:
    """渲染 PDF 页面为 PNG 图片字节"""
    import fitz
    pdf_doc = fitz.open(file_path)
    if page_num >= len(pdf_doc):
        pdf_doc.close()
        return b""
    page = pdf_doc[page_num]
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    pdf_doc.close()
    return img_bytes


async def _ocr_page_with_retry(file_path: str, page_num: int, semaphore: asyncio.Semaphore) -> str:
    """渲染 PDF 页面为图片 -> 调 Vision LLM 识别文字（带并发控制和指数退避重试）"""
    if not config.vision_api_key:
        logger.warning("VISION_API_KEY 未配置，跳过 Vision OCR")
        return ""

    # 渲染页面（同步操作，放到线程池避免阻塞事件循环）
    try:
        img_bytes = await asyncio.get_running_loop().run_in_executor(
            None, _render_page, file_path, page_num
        )
        if not img_bytes:
            return ""
    except Exception as e:
        logger.error(f"PDF 页面渲染失败: page={page_num} error={e}")
        return ""

    # 并发控制：获取信号量
    async with semaphore:
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        llm = _get_vision_llm()

        for attempt in range(OCR_MAX_RETRIES):
            try:
                # Vision LLM 调用（同步，放到线程池）
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: llm.invoke([
                        HumanMessage(content=[
                            {"type": "text", "text": _OCR_PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            }},
                        ])
                    ])
                )
                return response.content.strip()
            except Exception as e:
                if attempt < OCR_MAX_RETRIES - 1:
                    delay = OCR_BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Vision OCR 失败 (page={page_num}, attempt {attempt+1}/{OCR_MAX_RETRIES}): {e}, {delay:.1f}s 后重试")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Vision OCR 最终失败: page={page_num} error={e}")
                    return ""


@register_parser(".pdf")
class PDFParser(BaseParser):
    """PDF 解析器 - 一级按页切分，二级递归分割超长页，图片型调 Vision OCR（并发+重试）"""

    async def parse(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        docs = await asyncio.to_thread(loader.load)

        # 提取 PDF 级元信息
        pdf_title = ""
        if docs and docs[0].metadata.get("title"):
            pdf_title = docs[0].metadata["title"]

        # 一级：按页切分
        page_chunks = []
        ocr_tasks: List[Tuple[int, int]] = []  # (page_num, chunk_index)

        for page_doc in docs:
            page_num = page_doc.metadata.get("page", 0)
            text = page_doc.page_content
            text_len = len(text.strip())
            is_ocr = text_len < OCR_TEXT_THRESHOLD

            chunk_idx = len(page_chunks)

            if is_ocr:
                logger.info(f"PDF 图片型页面: page={page_num}, text_len={text_len}，标记为 OCR")
                ocr_tasks.append((page_num, chunk_idx))
                text = ""

            meta = build_base_metadata(
                file_path, ".pdf", chunk_index=chunk_idx, total_chunks=0,
                doc_title=pdf_title,
                location_ref=f"第{page_num + 1}页",
                extra={
                    "page": page_num,
                    "is_ocr": is_ocr,
                    "total_pages": len(docs),
                },
            )
            page_chunks.append(Document(page_content=text, metadata=meta))

        # 并发执行 OCR（带信号量限流 + 指数退避重试）
        if ocr_tasks:
            semaphore = asyncio.Semaphore(OCR_SEMAPHORE_LIMIT)
            logger.info(f"开始并发 OCR: {len(ocr_tasks)} 页, 并发限制={OCR_SEMAPHORE_LIMIT}")

            ocr_results = await asyncio.gather(*[
                _ocr_page_with_retry(file_path, page_num, semaphore)
                for page_num, _ in ocr_tasks
            ])

            # 回填 OCR 结果
            for (page_num, chunk_idx), ocr_text in zip(ocr_tasks, ocr_results):
                if ocr_text:
                    page_chunks[chunk_idx].page_content = ocr_text
                    logger.info(f"Vision OCR 成功: page={page_num}, ocr_len={len(ocr_text)}")
                else:
                    logger.warning(f"Vision OCR 未返回内容: page={page_num}")

        # 二级：递归分割（短页不受影响，超长页自动切分）
        final_chunks = await asyncio.to_thread(_SECONDARY_SPLITTER.split_documents, page_chunks)

        # 回填 chunk_index 和 total_chunks
        total = len(final_chunks)
        for i, c in enumerate(final_chunks):
            c.metadata["chunk_index"] = i
            c.metadata["total_chunks"] = total

        return final_chunks