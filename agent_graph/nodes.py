import json
import time
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from agent_graph.prompts import ASSISTANT_PROMPT, ANSWER_PROMPT, HUMAN_CONFIRMATION_PROMPT, COMPRESS_SUMMARY_PROMPT
from agent_graph.state import AgentState
from loguru import logger
from agent_graph.util import build_context, get_llm, get_last_user_message, get_last_tool_result
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
        effective_query=query or get_last_user_message(state["messages"]) or ""

        prompt_messages=ASSISTANT_PROMPT.format_messages(
            context=context,query=effective_query,user_id=user_id
        )
        messages=prompt_messages+state.get("messages",[])

        logger.info(
            f"[trace:{trace_id}] [assistant_node] "
            f"effective_query='{effective_query[:80]}'"
        )
        resp=await get_llm(temperature=0.1).bind_tools(tools).ainvoke(messages)

        logger.info(
            f"[trace:{trace_id}] [assistant_node] "
            f"resp.tool_calls={resp.tool_calls} "
            f"content='{str(resp.content)[:150]}'"
        )

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

        # 重复调用防护：检测上一轮 assistant 是否调用了完全相同的工具和参数
        prev_call_keys=set()
        skip_current=True
        for msg in reversed(messages):
            if isinstance(msg,AIMessage) and msg.tool_calls:
                if skip_current:
                    skip_current=False
                    continue
                for pc in msg.tool_calls:
                    prev_call_keys.add((pc["name"],json.dumps(pc["args"],sort_keys=True,ensure_ascii=False)))
                break
            if isinstance(msg,ToolMessage):
                continue
            break

        for call in last_calls:
            tool_name=call["name"]
            tool_args=call["args"]
            call_id=call.get("id",f"call_{tool_name}")

            call_key=(tool_name,json.dumps(tool_args,sort_keys=True,ensure_ascii=False))
            if call_key in prev_call_keys:
                logger.warning(
                    f"[trace:{trace_id}] [tools_node] "
                    f"阻止重复调用 name={tool_name} args={tool_args}"
                )
                result_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {"error":f"禁止重复调用相同工具和参数: {tool_name}"},
                            ensure_ascii=False
                        ),
                        name=tool_name,
                        tool_call_id=call_id
                    )
                )
                continue

            tool_func=tool_map.get(tool_name)
            if not tool_func:
                logger.error(f"[trace:{trace_id}] [tools_node] 未知工具:{tool_name}")
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


            if (tool_name=="rag_search" and not tool_args.get("query")):
                tool_args["query"]=state["current_query"]

            if (tool_name=="rag_search" and not tool_args.get("kb_id")):
                session_kb_id=state.get("kb_id")
                if session_kb_id:
                    tool_args["kb_id"]=session_kb_id

            if tool_name=="list_knowledge_bases":
                user_id=state.get("user_id")
                if user_id is not None:
                    tool_args["user_id"]=int(user_id)
            logger.info(
                f"[trace:{trace_id}] [tools_node] "
                f"执行工具 name={tool_name} args={tool_args}"
            )

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
        """
        Self-RAG 混合信号融合评分策略
        主信号: Reranker 分数 (权重 50%)
        辅助信号:
          - 文档覆盖度 (权重 25%): 多个 chunk 命中同一主题
          - 重试次数惩罚 (权重 15%): 重试越多，分数越低
          - 文档数量奖励 (权重 10%): 有效文档越多越好
        """
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []
        query=state.get("current_query") or ""
        retry_count=state.get("retry_count") or 0

        if not docs:
            return {"self_rag_score":0.0}

        # ═══════════════════════════════════════════════════════════
        # 信号1: Reranker 分数 (主信号，权重 50%)
        # ═══════════════════════════════════════════════════════════
        rerank_scores=[doc.get("rerank_score",0) for doc in docs]
        # 归一化到 [0, 1] (rerank score 通常是 logits，范围约 [-10, 10])
        import math
        normalized_rerank=[1/(1+math.exp(-s)) for s in rerank_scores]
        avg_rerank=sum(normalized_rerank)/len(normalized_rerank) if normalized_rerank else 0

        # ═══════════════════════════════════════════════════════════
        # 信号2: 文档覆盖度 (权重 25%)
        # 衡量多个 chunk 是否命中同一主题，而不是孤证
        # ═══════════════════════════════════════════════════════════
        def compute_coverage(docs, query):
            """计算文档覆盖度：基于文档来源多样性和内容重叠"""
            if len(docs) <= 1:
                return 0.3  # 只有一个文档，覆盖度较低

            # 统计不同文档来源
            doc_ids=set()
            file_names=set()
            for doc in docs:
                meta=doc.get("metadata",{})
                if meta.get("doc_id"):
                    doc_ids.add(meta["doc_id"])
                if meta.get("file_name"):
                    file_names.add(meta["file_name"])

            # 来源多样性: 来自不同文档/文件越多越好
            source_diversity=min(1.0, (len(doc_ids)*0.5+len(file_names)*0.5)/3)

            # 内容重叠: 多个文档包含相似关键词
            query_words=set(query.lower().split())
            overlap_counts=0
            for doc in docs:
                text=doc.get("text","")[:500].lower()
                doc_words=set(text.split())
                if len(query_words & doc_words) > 0:
                    overlap_counts+=1
            content_coverage=overlap_counts/len(docs) if docs else 0

            return source_diversity*0.6+content_coverage*0.4

        coverage=compute_coverage(docs,query)

        # ═══════════════════════════════════════════════════════════
        # 信号3: 重试次数惩罚 (权重 15%)
        # 重试越多，分数越低，防止无限循环
        # ═══════════════════════════════════════════════════════════
        max_retries=config.max_self_rag_retries
        retry_penalty=max(0, 1.0-(retry_count/max_retries)*0.8)

        # ═══════════════════════════════════════════════════════════
        # 信号4: 文档数量奖励 (权重 10%)
        # 有效文档越多，信息越充分
        # ═══════════════════════════════════════════════════════════
        effective_docs=sum(1 for s in normalized_rerank if s>0.3)
        doc_count_score=min(1.0, effective_docs/3)

        # ═══════════════════════════════════════════════════════════
        # 加权融合
        # ═══════════════════════════════════════════════════════════
        final_score=(
            avg_rerank*0.50+           # Reranker 主信号
            coverage*0.25+             # 文档覆盖度
            retry_penalty*0.15+        # 重试惩罚
            doc_count_score*0.10       # 文档数量
        )

        score=round(final_score,4)

        # ═══════════════════════════════════════════════════════════
        # 详细日志
        # ═══════════════════════════════════════════════════════════
        logger.info(
            f"[trace:{trace_id}] [self_rag_node] "
            f"混合评分: final={score} "
            f"rerank={avg_rerank:.3f} coverage={coverage:.3f} "
            f"retry_penalty={retry_penalty:.3f} doc_count={doc_count_score:.3f} "
            f"docs={len(docs)} retry={retry_count} "
            f"duration={time.time()*1000-start_ms}ms"
        )

        return {"self_rag_score":score}

    async def answer_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""

        history_context=build_context(state)
        effective_query=get_last_user_message(state["messages"]) or query

        prompt_messages=ANSWER_PROMPT.format_messages(history=history_context,query=effective_query)
        messages=prompt_messages+state.get("messages",[])
        resp=await get_llm(temperature=0.7).ainvoke(messages)

        logger.info(
            f"[trace:{trace_id}] [answer_node] "
            f"生成回答: len={len(resp.content)} duration={time.time()*1000-start_ms}ms"
        )
        return {"messages":[AIMessage(content=resp.content)]}

    async def human_intervention_node(state:AgentState)->dict:
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []

        if docs:
            context_parts=[]
            for i,doc in enumerate(docs[:3],1):
                text=doc.get("text","")[:300]
                # 优先使用 rerank_score，降级到 score
                score=doc.get("rerank_score") or doc.get("score",0)
                score_label="rerank" if doc.get("rerank_score") is not None else "vec"
                context_parts.append(f"[{i}] ({score_label}:{score:.2f}) {text}")
            context="\n".join(context_parts)
        else:
            context="未检索到相关文档"

        prompt=HUMAN_CONFIRMATION_PROMPT.format(query=query,context=context)
        resp=await get_llm(temperature=0.7).ainvoke(prompt)
        confirmation_text=str(resp.content)

        user_confirmation=interrupt({
            "question":query,
            "context":context,
            "confirmation_text":confirmation_text,
            "self_rag_score":state.get("self_rag_score",0.0),
            "status":"pending"
        })

        logger.info(
            f"[trace:{trace_id}] [human_intervention] "
            f"等待人工确认 status=pending "
            f"duration={time.time()*1000-start_ms}ms"
        )
        return {"human_confirmation":user_confirmation}

    async def rewrite_query_node(state:AgentState)->dict:
        """
        简化查询改写节点

        职责：
        1. 增加重试计数
        2. 保留原始 query（改写由多路召回服务自动处理）

        流程：
        self_rag评分低 → rewrite_query_node(只加计数) → assistant → rag_search(多路召回自动处理)
        """
        retry=(state.get("retry_count") or 0)+1
        original=state.get("current_query") or ""

        logger.info(
            f"[trace:{state.get('trace_id')}] [rewrite_query_node] "
            f"重试计数+1 retry={retry} query='{original[:50]}' "
            f"(改写由多路召回服务自动处理)"
        )

        return {"retry_count":retry,"current_query":original}

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




