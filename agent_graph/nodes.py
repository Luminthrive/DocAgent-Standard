import json
import time
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from agent_graph.prompts import ASSISTANT_PROMPT, ANSWER_PROMPT, HUMAN_CONFIRMATION_PROMPT, QUERY_REWRITE_PROMPT, \
    COMPRESS_SUMMARY_PROMPT
from agent_graph.state import AgentState
from loguru import logger
from agent_graph.util import build_context, get_llm, get_last_user_message, get_last_tool_result, truncate_tool_result
from config import config


def create_node(tools):
    async def assistant_node(state: AgentState) -> dict:
        trace_id = state.get("trace_id", "")
        query = state.get("current_query", "") or ""
        user_id=state.get("user_id","")
        logger.info(
            f"[trace:{trace_id}] [assistant_node] "
            f"开始推理 query='{query[:50]}...' tools={len(tools)}"
        )

        context=build_context(state)
        effective_query=get_last_user_message(state["messages"]) or query

        prompt_messages=ASSISTANT_PROMPT.format_messages(
            context=context,query=effective_query,user_id=user_id
        )
        resp=await get_llm(temperature=0.1).bind_tools(tools).ainvoke(prompt_messages)

        return {"messages":[resp]}

    async def tools_node(state:AgentState)->dict:

        trace_id=state.get("trace_id","")
        messages=state["messages"]
        last_calls=[]
        for msg in reversed(messages):
            if isinstance(msg,AIMessage) and msg.tool_calls:
                last_calls=msg.tool_calls
                break
        if not last_calls:
            return {}

        tool_map={tool.name:tool for tool in tools}
        result_messages=[]

        for call in last_calls:
            tool_name=call["name"]
            tool_args=call["args"]
            call_id=call.get("id",f"call_{tool_name}")
            tool_func=tool_map.get(tool_name)
            if not tool_func:
                logger.error(f"[trace:{trace_id}]",f"[tools_node] 未知工具:{tool_name}")
                result_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "error":f"未知工具:{tool_name}"
                            },
                            ensure_ascii=False
                        ),
                        name=tool_name,
                        tool_call_id=call_id
                    )
                )
                continue


            if (tool_name=="rag_search" and state.get("current_query")):
                tool_args["query"]=state["current_query"]

            if (tool_name=="rag_search" and not tool_args.get("kb_id")):
                session_kb_id=state.get("kb_id")
                if session_kb_id:
                    tool_args["kb_id"]=session_kb_id

            if tool_name=="list_knowledge_bases":
                user_id=state.get("user_id")
                if user_id is not None:
                    tool_args["user_id"]=user_id
            logger.info(
                f"[trace:{trace_id}] [tools_node] "
                f"执行工具 name={tool_name} args={tool_args}"
            )
            message=None

            try:
                start_ms = time.time() * 1000
                result=await tool_func.ainvoke(tool_args)
                if (not result or (isinstance(result,str) and not result.strip())):
                    logger.warning(
                        f"[trace:{trace_id}] [tools_node] "
                        f"工具返回空结果 name={tool_name}"
                    )
                    result={
                        "warning":(
                            f"工具{tool_name}返回了空结果，请使用其他方式回答"
                        )
                    }
                message=ToolMessage(
                    content=json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str
                    ),
                    name=tool_name,
                    tool_call_id=call_id
                )
                logger.info(
                    f"[trace:{trace_id}] [tools_node] "
                    f"工具执行成功 name={tool_name} duration={time.time()*1000-start_ms}ms"
                )
            except Exception as e:
                error_msg = (
                    f"工具 {tool_name} 执行失败: {str(e)}。"
                    "请尝试其他方式回答用户问题。"
                )

                message = ToolMessage(
                    content=json.dumps(
                        {
                            "error": error_msg
                        },
                        ensure_ascii=False,
                    ),
                    name=tool_name,
                    tool_call_id=call_id,
                )

                logger.error(
                    f"[trace:{trace_id}] [tools_node] "
                    f"工具执行失败 name={tool_name} error={e}"
                )

            result_messages.append(message)

        return {
            "messages":result_messages
        }

    def self_rag_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []
        query=state.get("current_query") or ""

        if not docs:
            self_rag_score=0.0
        else:
            scores=[]
            query_words=set(query.lower().split())
            for doc in docs[:-3]:
                text=doc.get("text","")[:500].lower()
                overlap=len(query_words & set(text.split()))/max(len(query_words),1)
                vec_score=doc.get("score",0.5)
                doc_score=overlap*0.3+vec_score*0.7
                scores.append(doc_score)
            self_rag_score=sum(scores)/len(scores)

        score=round(self_rag_score,3)
        logger.info(
            f"[trace:{trace_id}] [self_rag_node] "
            f"评分完成 score={score} docs={len(docs)} "
            f"duration={time.time()*1000-start_ms}ms"
        )

        if score < 0.3:
            decision = "human_intervention"
        elif score < 0.5:
            decision = "rewrite_query"
        else:
            decision = "answer"

        logger.info(
            f"[trace:{trace_id}] [self_rag_node] "
            f"决策={decision} score={score}"
        )

        return {"self_rag_score":score}

    async def answer_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""
        rag_result=get_last_tool_result(state["messages"],"rag_search")
        effective_query=get_last_user_message(state["messages"]) or query

        history_context=build_context(state)
        context=f"用户问题：\n{effective_query}\n\n"
        if rag_result:
            context+=f"[rag_search]{truncate_tool_result('rag_search',rag_result)}"
        else:
            context+=(
                "【注意】当前没有调用过任何工具,没有任何操作被实际执行。"
                "请诚实地告诉用户你尚未执行任何操作,不要编造\"已完成\"、\"已创建\"等虚假结果。"
            )

        prompt_messages=ANSWER_PROMPT.format_messages(history=history_context,context=context,query=effective_query)
        resp=await get_llm(temperature=0.7).ainvoke(prompt_messages)

        logger.info(
            f"[trace:{trace_id}] [answer_node] "
            f"生成回答: len={len(resp.content)} duration=..."
        )
        return {"messages":[AIMessage(content=resp.content)]}

    async def human_intervention_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []

        context=""
        if docs:
            context_parts=[]
            for i,doc in enumerate(docs[:3],1):
                text=doc.get("text","")[:300]
                score=doc.get("score",0)
                context_parts.append(f"[{i}] (相关度:{score:.2f}) {text}")
            context="\n".join(context_parts)
        else:
            context="未检索到相关文档"

        prompt=HUMAN_CONFIRMATION_PROMPT.format(query=query,context=context)
        resp=await get_llm(temperature=0.7).ainvoke(prompt)
        confirmation_text=str(resp.content)

        user_confirmation=interrupt({
            "question":query,
            "contexst":context,
            "confirmation_text":confirmation_text,
            "self_rag_score":state.get("self_rag_score",0.0),
            "status":"pending"
        })

        logger.info(
            f"[trace:{trace_id}] [human_intervention] "
            f"等待人工确认 status={...} "
            f"duration={time.time()*1000-start_ms}ms"
        )
        return {"human_confirmation":user_confirmation}

    async def rewrite_query_node(state:AgentState)->dict:
        retry=(state.get("retry_count") or 0)+1
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}

        original=state["current_query"]
        docs = rag_result.get("docs") or []
        docs_text = "\n".join(
            d.get("text", "")[:200]
            for d in docs[:3]
        )
        prompt=QUERY_REWRITE_PROMPT.format(
            original_query=original,
            result_summary=docs_text
        )
        current_query=original
        try:
            resp=await get_llm(temperature=0.3).ainvoke(prompt)
            new_query=str(resp.content).strip()
            if new_query:
                current_query=new_query
            logger.info(
                f"[trace:{state.get('trace_id')}] [rewrite_query_node] "
                f"查询改写完成 retry={retry} "
                f"new_query='{current_query[:50]}'"
            )
        except Exception:
            logger.warning(
                f"[trace:{state.get('trace_id')}] [rewrite_query_node] "
                f"查询改写失败，使用原始查询 retry={retry}"
            )

        return {"retry_count":retry,"current_query":current_query}

    def human_review_reset_node(state:AgentState)->dict:
        return {"human_confirmation":None,"self_rag_score":0.5}

    async def compress_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        messages=state.get("messages") or []
        threshold=config.max_history_turns*2
        if len(messages)<=threshold:
            return {}
        logger.info(
            f"[trace:{state.get('trace_id')}] [compress_node] "
            f"触发压缩 messages={len(messages)} threshold={threshold}"
        )
        to_summarize=messages[:-threshold]
        keep_recent=messages[-threshold:]
        lines=[]
        for msg in to_summarize:
            if hasattr(msg,"type"):
                role="用户" if msg.type=="human" else "助手"
                content=str(msg.content)
            else:
                role="未知"
                content=str(msg)
            lines.append(f"{role}:{content.strip()[:300]}")
        conversation_text="\n".join(lines)
        prompt=COMPRESS_SUMMARY_PROMPT.format(conversation=conversation_text)
        resp=await get_llm(temperature=0.1).ainvoke(prompt)
        summary=str(resp.content).strip()
        if summary and len(summary)>=10:
            new_messages=[AIMessage(content=f"[对话摘要] {summary}")]+keep_recent
        else:
            new_messages=keep_recent
        logger.info(
            f"[trace:{state.get('trace_id')}] [compress_node] "
            f"压缩完成 messages={len(messages)} "
            f"kept={len(keep_recent)} "
            f"duration={time.time()*1000-start_ms}ms"
        )

        return {"messages":new_messages}

    return {
        "assistant": assistant_node,
        "tools": tools_node,
        "self_rag": self_rag_node,
        "answer": answer_node,
        "human_intervention": human_intervention_node,
        "rewrite_query": rewrite_query_node,
        "human_review_reset": human_review_reset_node,
        "compress": compress_node,
    }




