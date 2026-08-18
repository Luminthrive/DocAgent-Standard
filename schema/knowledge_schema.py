from datetime import datetime
from typing import Optional, Dict, Any, List

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

class RetrieveRequest(BaseModel):
    """
    独立检索请求
    """
    query:str=Field(...,min_length=1,max_length=1000,description="检索查询")
    top_k:int=Field(5,ge=1,le=20,description="返回数量")

class RetrieveResult(BaseModel):
    """
    单条检索结果
    """
    score:float=Field(...,description="相似度分数0-1")
    text:str=Field(...,description="文档内容")
    metadata:Dict[str,Any]=Field(default_factory=dict,description="元数据")

class RetrieveResponse(BaseModel):
    """
    检索响应
    """
    query:str
    results:List[RetrieveResult]
    count:int