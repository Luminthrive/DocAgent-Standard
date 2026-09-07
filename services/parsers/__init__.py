from typing import Dict, Type

from services.parsers.base_parser import BaseParser

PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {}


def register_parser(file_type: str):
    """装饰器：注册解析器到注册表"""
    def decorator(cls: Type[BaseParser]):
        PARSER_REGISTRY[file_type] = cls
        return cls
    return decorator


def get_parser(file_type: str) -> BaseParser:
    """根据文件类型获取解析器实例"""
    if file_type not in PARSER_REGISTRY:
        raise ValueError(f"不受支持的文件类型: {file_type}")
    return PARSER_REGISTRY[file_type]()


# 导入所有解析器以触发 @register_parser 装饰器注册
from services.parsers import text_parser  # noqa: F401, E402
from services.parsers import markdown_parser  # noqa: F401, E402
from services.parsers import pdf_parser  # noqa: F401, E402
from services.parsers import docx_parser  # noqa: F401, E402
from services.parsers import excel_parser  # noqa: F401, E402
from services.parsers import pptx_parser  # noqa: F401, E402
