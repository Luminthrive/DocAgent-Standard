import json
import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt
from loguru import logger

from agent_graph.state import AgentState


class AgentService:
    def __init__(self,session_service,graph):
        self.session_service=session_service
        self.graph=graph

    @staticmethod
    def generate_trace_id():
        timestamp=time.strftime("%Y%m%d_%H%M%S")
        short_uuid=uuid.uuid4().hex[:8]
        return f"tr_{timestamp}_{short_uuid}"

    async def _prepare_run(self,session_id:str,user_id:str,message:str):
        trace_id=self.generate_trace_id()
        logger.info(f"[trace:{trace_id}] 新建请求:session={session_id}")
        await self.session_service.create_message(
            session_id=session_id,
            role="user",
            content=message
        )
        session=await self.session_service.get_session(
            session_id=session_id,
            user_id=user_id
        )
        session_kb_id=session.get("kb_id") if session else None
        initial_state:AgentState={
            "messages": [HumanMessage(content=message)],
            "current_query": message,
            "session_id": session_id,
            "user_id": user_id,
            "kb_id": session_kb_id,
            "trace_id": trace_id,
            "self_rag_score": 0.0,
            "retry_count": 0,
            "human_confirmation": None,
        }
        return initial_state,trace_id

    @staticmethod
    def _extract_answer(messages)->str:
        for message in reversed(messages or []):
            if isinstance(message,AIMessage) and message.content:
                return str(message.content)
        return ""

    @staticmethod
    def _sse(event_type:str,**fields)->str:
        payload={
            "type":event_type,
            **fields
        }
        return f"data:{json.dumps(payload,ensure_ascii=False)}\n\n"

    @staticmethod
    def _step_summary(node_name:str,node_out:Any)->dict:
        info={
            "node":node_name
        }
        if isinstance(node_out,dict):
            info['updated_fields']=list(node_out.keys())
            messages = node_out.get("messages")
            if isinstance(messages, list):
                info["message_count"] = len(messages)

        return info

    async def handle_human_confirmation(self, session_id, user_id, confirmation):
        status = confirmation.get("status", "pending")
        answer = confirmation.get("answer", "")

        if status == "confirmed" and answer:
            await self.session_service.create_message(
                session_id=session_id,
                role="user",
                content=f"[人工回答] {answer}",
            )

        from langgraph.types import Command

        config = {
            "recursion_limit": 50,
            "configurable": {
                "thread_id": session_id or "default",
            },
        }

        result = await self.graph.ainvoke(
            Command(resume=confirmation),
            config=config,
        )

        answer_text = self._extract_answer(
            result.get("messages", [])
        )

        if answer_text:
            await self.session_service.create_message(
                session_id=session_id,
                role="assistant",
                content=answer_text,
            )
        else:
            answer_text = (
                "已取消"
                if status != "confirmed"
                else answer
            )

        return {
            "trace_id": self.generate_trace_id(),
            "answer": answer_text,
            "session_id": session_id,
            "self_rag_score": result.get(
                "self_rag_score"
            ),
            "need_human_review": False,
            "latency_ms": 0,
        }

    async def run_stream(self, session_id, user_id, message):
        yield self._sse(
            "start",
            message="开始处理...",
        )

        t0 = time.time()

        initial_state, trace_id = await self._prepare_run(
            session_id,
            user_id,
            message,
        )

        config = {
            "recursion_limit": 50,
            "configurable": {
                "thread_id": session_id or "default",
            },
        }

        final_state = None

        try:
            async for chunk in self.graph.astream(
                    initial_state,
                    config,
                    stream_mode=["updates", "values"],
            ):
                mode, data = chunk

                if mode == "values":
                    final_state = data
                    continue

                # HITL
                if "__interrupt__" in data:
                    interrupts = data.get("__interrupt__") or []

                    if interrupts:
                        payload = getattr(
                            interrupts[0],
                            "value",
                            interrupts[0],
                        )

                        logger.info(
                            f"[trace:{trace_id}] "
                            f"流式需人工确认"
                        )

                        yield self._sse(
                            "human_review",
                            data={
                                "question": payload.get(
                                    "question",
                                    "",
                                ),
                                "context": payload.get(
                                    "context",
                                    "",
                                ),
                                "confirmation_text": payload.get(
                                    "confirmation_text",
                                    "",
                                ),
                                "self_rag_score": payload.get(
                                    "self_rag_score",
                                    0.0,
                                ),
                            },
                        )

                        yield self._sse(
                            "done",
                            latency_ms=int(
                                (time.time() - t0) * 1000
                            ),
                        )

                        return

                # Node step
                for node_name, node_out in data.items():
                    if node_name == "__interrupt__":
                        continue

                    yield self._sse(
                        "step",
                        data=self._step_summary(
                            node_name,
                            node_out,
                        ),
                    )

            # 某些情况下 values 没有拿到最终状态
            if final_state is None:
                snapshot = await self.graph.aget_state(config)
                final_state = snapshot.values

            messages = (
                final_state.get("messages", [])
                if isinstance(final_state, dict)
                else []
            )

            answer_text = self._extract_answer(messages)

            if answer_text:
                await self.session_service.create_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                )

            yield self._sse(
                "answer",
                data=answer_text,
            )

            yield self._sse(
                "done",
                latency_ms=int(
                    (time.time() - t0) * 1000
                ),
            )

        except Exception as e:
            logger.error(
                f"[trace:{trace_id}] 流式异常: {e}",
                exc_info=True,
            )

            yield self._sse(
                "error",
                message=str(e),
            )

    async def run(self, session_id, user_id, message, enable_trace):
        start=time.time()
        initial_state,trace_id=await self._prepare_run(
            session_id,user_id,message
        )
        config={
            "recursion_limit":50,
            "configurable":{
                "thread_id":session_id or "default"
            }
        }
        try:
            result=await self.graph.ainvoke(
                initial_state,
                config=config
            )
            interrupts=result.get("__interrupt__") or []
            if interrupts:
                payload=interrupts[0].value
                logger.info(
                    f"[trace:{trace_id}] 需人工确认: "
                    f"question={str(payload.get('question', ''))[:40]}"
                )
                return {
                    "trace_id": trace_id,
                    "need_human_review": True,
                    "question": payload.get("question", ""),
                    "context": payload.get("context", ""),
                    "confirmation_text": payload.get(
                        "confirmation_text",
                        "",
                    ),
                    "self_rag_score": payload.get(
                        "self_rag_score",
                        0.0,
                    ),
                    "latency_ms": int(
                        (time.time() - start) * 1000
                    ),
                }
            answer_text=self._extract_answer(
                result.get("messages",[])
            )
            if answer_text:
                await self.session_service.create_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer_text
                )
            latency_ms = int(
                (time.time() - start) * 1000
            )
            logger.info(
                f"[trace:{trace_id}] 请求完成: "
                f"latency={latency_ms}ms"
            )
            return {
                "trace_id": trace_id,
                "answer": answer_text,
                "session_id": session_id,
                "self_rag_score": result.get("self_rag_score"),
                "need_human_review": False,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            latency_ms = int(
                (time.time() - start) * 1000
            )

            logger.error(
                f"[trace:{trace_id}] 系统异常: {e}",
                exc_info=True,
            )

            return {
                "trace_id": trace_id,
                "answer": "抱歉，系统内部错误，请稍后重试。",
                "session_id": session_id,
                "self_rag_score": 0.0,
                "need_human_review": False,
                "latency_ms": latency_ms,
            }
