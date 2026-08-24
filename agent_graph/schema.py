from typing import Optional

from pydantic import Field, BaseModel


class HumanReviewRequest(BaseModel):
    question:str=Field(description="向用户提出问题")
    context:str=Field(description="检索到的低质量上下文")
    confirmation_text:str=Field(description="LLM生成的确认提示文本")
    self_rag_score:float=Field(default=0.0,description="触发时的self-rag评分")
    status:str=Field(default="pending",description="pending/confirmed/cancelled")


class RagSearchInput(BaseModel):
    query:str=Field(description="检索查询文本")
    kb_id:Optional[str]=Field(default=None,description="知识库id,不指定使用默认知识库")
    top_k:int =Field(default=5,description="返回文档数量，默认是5")



