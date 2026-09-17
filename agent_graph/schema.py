from typing import Optional

from pydantic import Field, BaseModel


class HumanReviewRequest(BaseModel):
    question:str=Field(description="向用户提出问题")
    context:str=Field(description="检索到的低质量上下文")
    confirmation_text:str=Field(description="LLM生成的确认提示文本")
    self_rag_score:float=Field(default=0.0,description="触发时的self-rag评分")
    status:str=Field(default="pending",description="pending/confirmed/cancelled")


class RagSearchInput(BaseModel):
    query:str=Field(description="用户的检索查询，用于在知识库中搜索相关文档。请将用户问题转化为适合检索的查询文本")
    kb_id:Optional[str]=Field(default=None,description="系统根据当前会话自动注入的知识库ID，无需填写，传入也会被忽略")
    top_k:int =Field(default=5,description="返回文档数量。简单问题用3-5条，复杂问题或需要更多上下文时用8-10条")



