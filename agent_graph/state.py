from typing import Optional, Annotated, Any, Dict, List

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field




class AgentState(MessagesState):
    session_id:Optional[str]
    user_id:Optional[str]
    kb_id:Optional[str]

    current_query:Optional[str]

    self_rag_score:Optional[float]

    retry_count:int

    human_confirmation:Optional[Dict[str,Any]]

    trace_id:Optional[str]
