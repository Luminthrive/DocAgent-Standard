from gc import enable

from fastapi import APIRouter, Depends, Query, HTTPException
from scripts.regsetup import description
from starlette.responses import StreamingResponse

from core.auth import get_current_user_id
from schema.agent_schema import CreateSessionRequest, SessionResponse, MessageResponse, RunAgentRequest, \
    HumanReviewRequired, AgentRunResponse
from services.agent_service import AgentService
from services.session_service import SessionService

router=APIRouter()

agent_service=AgentService()
session_service=SessionService()

@router.post("/sessions",response_model=SessionResponse,summary="创建会话")
async def create_session(
        request:CreateSessionRequest,
        user_id:str=Depends(get_current_user_id)
):
    """
    创建会话
    :param request:
    :param user_id:
    :return:
    """
    session=await session_service.create_session(
        user_id=user_id,
        kb_id=request.kb_id,
        title=request.title
    )
    return session

@router.get("/sessions",response_model=list[SessionResponse],summary="获取会话列表")
async def list_sessions(
        limit:int=Query(20,ge=1,le=100,description="每页数量"),
        offset:int=Query(0,ge=0,description="偏移量"),
        user_id:str=Depends(get_current_user_id)
):
    """
    获取用户会话列表
    :param limit:
    :param offset:
    :param user_id:
    :return:
    """
    sessions=await session_service.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    return sessions

@router.get("/sessions/{session_id}/messages",response_model=list[MessageResponse],summary="获取会话消息")
async def get_message(
        session_id:str,
        user_id:str=Depends(get_current_user_id)
):
    """
    获取指定会话的会话历史
    :param session_id:
    :param user_id:
    :return:
    """
    session=await session_service.get_session(session_id,user_id)
    if not session:
        raise HTTPException(status_code=404,detail="会话不存在")
    messages=await session_service.list_messages(session_id)
    return messages

@router.delete("/sessions/{session_id}",summary="删除会话")
async def delete_session(
        session_id:str,
        user_id:str=Depends(get_current_user_id)
):
    """删除指定会话及其所有消息"""
    session=await session_service.get_session(session_id,user_id)
    if not session:
        raise HTTPException(status_code=404,detail="会话不存在")
    await session_service.delete_session(session_id)
    return {"status":"deleted","session_id":session_id}

@router.post("/sessions/{session_id}/run",response_model=AgentRunResponse|HumanReviewRequired,summary="执行agent问答")
async def run_agent(
        session_id:str,
        request:RunAgentRequest,
        user_id:str=Depends(get_current_user_id)
):
    session=await session_service.get_session(session_id,user_id)
    if not session:
        raise HTTPException(status_code=404,detail="会话不存在")
    if request.human_confirmation:
        result=await agent_service.handle_human_confirmation(
            session_id=session_id,
            user_id=user_id,
            confirmation=request.human_confirmation
        )
        return result
    if request.stream:
        return StreamingResponse(
            agent_service.run_stream(
                session_id=session_id,
                user_id=user_id,
                message=request.message
            ),
            media_type="text/event-stream"
        )

    result=await agent_service.run(
        session_id=session_id,
        user_id=user_id,
        message=request.message,
        enable_trace=request.enable_trace
    )
    if result.get("need_human_review"):
        return HumanReviewRequired(
            trace_id=result.get("trace_id",""),
            need_human_review=True,
            session_id=session_id,
            question=result.get("question",""),
            context=result.get("context",""),
            confirmation_text=result.get("confirmation_text",""),
            self_rag_score=result.get("self_rag_score",0.0),
            message="检索质量不足，请确认是否继续"
        )

    if not request.enable_trace:
        result["react_steps"]=[]

    return AgentRunResponse(**result)
