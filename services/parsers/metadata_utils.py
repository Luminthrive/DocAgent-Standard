"""元数据工具 — 所有 parser 共用的 metadata 生成逻辑"""

import os
from typing import Dict, Any, Optional

# file_type 缩写映射（统一命名）
_EXT_MAP = {
    ".md": "md", ".txt": "txt", ".pdf": "pdf",
    ".docx": "docx", ".xlsx": "xlsx", ".pptx": "ppt",
}


def normalize_file_type(ext: str) -> str:
    """扩展名 → 统一短名（.pdf → pdf）"""
    return _EXT_MAP.get(ext.lower(), ext.lstrip("."))


def file_timestamps(file_path: str) -> Dict[str, int]:
    """获取文件创建/修改时间戳（Unix int）"""
    try:
        create_ts = int(os.path.getctime(file_path))
        update_ts = int(os.path.getmtime(file_path))
    except OSError:
        create_ts = update_ts = 0
    return {"create_ts": create_ts, "update_ts": update_ts}


def build_base_metadata(
    file_path: str,
    file_type: str,
    chunk_index: int,
    total_chunks: int,
    *,
    doc_title: str = "",
    location_ref: str = "",
    section_path: str = "",
    offset_start: Optional[int] = None,
    offset_end: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建通用 metadata（不含 doc_id/chunk_id，由 document_service 注入）。
    调用方只需传文件类型特有的字段。
    """
    ft = normalize_file_type(file_type)
    ts = file_timestamps(file_path)
    meta = {
        "source_file_name": os.path.basename(file_path),
        "file_type": ft,
        # doc_id / chunk_id 由 document_service 注入，这里先占位
        "doc_id": "",
        "chunk_id": "",
        "chunk_index": chunk_index,
        "location_ref": location_ref,
        "offset_start": offset_start,
        "offset_end": offset_end,
        "doc_title": doc_title or os.path.splitext(os.path.basename(file_path))[0],
        "section_path": section_path,
        **ts,
    }
    if extra:
        meta.update(extra)
    return meta


def build_payload_content(
    chunk_text: str,
    parsed_raw: str = "",
) -> Dict[str, str]:
    """构建 content 层（送入 Embedding / 返回给上层）"""
    return {
        "chunk_text": chunk_text,
        "parsed_raw": parsed_raw,
    }
