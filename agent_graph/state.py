from typing import Optional, Annotated, Any, Dict, List

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field




class AgentState(MessagesState):
    session_id:Optional[str]
    user_id:Optional[str]
    kb_id:Optional[str]

    current_query:Optional[str]

    self_rag_score:Optional[float]

    retry_count:int=0

    # HITL: 人工介入模式 ("edit_query" | "force_answer")
    hitl_mode:Optional[str]

    # HITL: 人工修改后的query
    human_edited_query:Optional[str]

    # HITL: 人工确认结果（保留兼容）
    human_confirmation:Optional[Dict[str,Any]]

    trace_id:Optional[str]
