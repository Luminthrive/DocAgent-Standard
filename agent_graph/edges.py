from typing import Literal

from langchain_core.messages import AIMessage

from config import config
from agent_graph.state import AgentState


def router(state:AgentState)->Literal["tools","answer"]:
    retry_count=state.get("retry_count") or 0
    if retry_count >= config.max_agent_iterations:
        return "answer"
    messages=state.get("messages") or []
    last_msg=messages[-1] if messages else None
    if isinstance(last_msg,AIMessage) and last_msg.tool_calls:
        return "tools"
    return "answer"


def tools_router(state:AgentState)->Literal["self_rag","assistant"]:
    messages=state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg,AIMessage) and msg.tool_calls:
            if any(call.get("name")=="rag_search" for call in msg.tool_calls):
                return "self_rag"
            return "assistant"
    return "assistant"

def self_rag_router(state:AgentState)->Literal["rewrite_and_retry","human_intervention","answer"]:
    """
    Self-RAG 路由决策
    使用混合评分分数进行决策:
    - score < low_threshold: 人工干预
    - low_threshold <= score < mid_threshold: 改写查询重试
    - score >= mid_threshold: 直接回答
    """
    score=state.get("self_rag_score") or 0.0
    retry=state.get("retry_count") or 0

    low_threshold=config.rag_score_low_threshold
    mid_threshold=config.rag_score_mid_threshold

    if score < low_threshold:
        return "human_intervention"

    if score < mid_threshold and retry < config.max_self_rag_retries:
        return "rewrite_and_retry"

    return "answer"

def compress_router(state:AgentState)->Literal["compress","end"]:
    messages=state.get("messages") or []
    if len(messages) > config.max_history_turns*2:
        return "compress"
    return "end"

def human_review_router(state:AgentState)->Literal["reset_and_retry","end"]:
    confirmation=state.get("human_confirmation") or {}
    status=confirmation.get("status","pending")
    if status=="confirmed":
        return "reset_and_retry"
    return "end"
