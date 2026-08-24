from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    kb_id:Optional[str]=Field(None,description="关联知识库id（可选）")
    title:Optional[str]=Field(None,description="会话标题,可选可自动生成")

class SessionResponse(BaseModel):
    """会话响应"""
    id:str
    title:str
    kb_id:Optional[str]
    created_at:datetime
    updated_at:Optional[datetime]

class MessageResponse(BaseModel):
    """消息响应"""
    id:str
    role:str
    content:str
    tool_calls:Optional[List[Dict[str,Any]]]
    tool_results:Optional[Dict[str,Any]]
    created_at:datetime

class RunAgentRequest(BaseModel):
    """执行agent问答请求"""
    message:str=Field(...,min_length=1,max_length=10000,description="用户消息")
    stream:bool=Field(False,description="是否使用SSE流式输出")
    enable_trace:bool=Field(False,description="是否开启Trace值")
    human_confirmation:Optional[Dict[str,Any]]=Field(None,
          description="人机协同确认：{status:'confirmed'|'cancelled',answer?:str}")

class HumanReviewRequired(BaseModel):
    """需要人工审核"""
    trace_id:str
    need_human_review:bool=True
    session_id:str
    question:str
    context:str
    confirmation_text:str
    self_rag_score:float
    message:str="检索质量不足，请确认是否继续"

class AgentRunResponse(BaseModel):
    """agent执行响应（同步模式）"""
    trace_id:str
    answer:str
    session_id:str
    self_rag_score:Optional[float]
    need_human_review:bool
    latency_ms:int