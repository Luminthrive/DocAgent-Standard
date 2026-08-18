from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CreateKnowledgeBaseRequest(BaseModel):
    """
    创建知识库请求
    """
    name:str =Field(...,min_length=1,max_length=100,description="知识库名称")
    description:Optional[str] =Field(None,max_length=500,description="描述")


class KnowledgeBaseResponse(BaseModel):
    """
    操作知识库相关响应
    """
    id:str
    name:str
    description:Optional[str]
    doc_count:int
    chunk_count:int
    created_at:datetime

class UploadResponse(BaseModel):
    """
    上传响应
    """
    doc_id:str
    file_name:str
    status:str
    message:str

class DocumentResponse(BaseModel):
    """
    文档响应
    """
    id:str
    file_name:str
    file_type:str
    file_size:int
    chunk_count:int
    status:str
    error_msg:Optional[str]
    created_at:datetime