import json
import time
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt
from agent_graph.prompts import ASSISTANT_PROMPT, ANSWER_PROMPT, HUMAN_CONFIRMATION_PROMPT, QUERY_REWRITE_PROMPT, COMPRESS_SUMMARY_PROMPT
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
        Self-RAG 三维融合评分策略
        - Reranker 分数 (60%): 检索结果与查询的相关性
        - 多源有效文档 (25%): 有效文档占总文档的比例
        - 内容覆盖度 (15%): chunk 与查询的关键词匹配度
        """
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []
        query=state.get("current_query") or ""

        if not docs:
            return {"self_rag_score":0.0}

        # ═══════════════════════════════════════════════════════════
        # 信号1: Reranker 分数 (主信号，权重 60%)
        # ═══════════════════════════════════════════════════════════
        rerank_scores=[doc.get("rerank_score",0) for doc in docs]
        # 归一化到 [0, 1] (rerank score 通常是 logits，范围约 [-10, 10])
        import math
        normalized_rerank=[1/(1+math.exp(-s)) for s in rerank_scores]
        avg_rerank=sum(normalized_rerank)/len(normalized_rerank) if normalized_rerank else 0

        # ═══════════════════════════════════════════════════════════
        # 信号2: 多源有效文档 (权重 25%)
        # 来自多少篇不同文档，且 Reranker 分数超过阈值
        # ═══════════════════════════════════════════════════════════
        def compute_multi_source_effective(docs, normalized_rerank):
            """计算多源有效文档数：有效文档占总文档的比例"""
            if not docs:
                return 0.0

            doc_ids=set()
            effective_doc_ids=set()

            for doc, score in zip(docs, normalized_rerank):
                meta=doc.get("metadata",{})
                doc_id=meta.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)
                    # 有效文档：Reranker 分数 > 0.3
                    if score > 0.3:
                        effective_doc_ids.add(doc_id)

            return len(effective_doc_ids)/len(doc_ids) if doc_ids else 0.0

        multi_source_score=compute_multi_source_effective(docs, normalized_rerank)

        # ═══════════════════════════════════════════════════════════
        # 信号3: 内容覆盖度 (权重 15%)
        # 有多少 chunk 包含查询关键词
        # ═══════════════════════════════════════════════════════════
        import jieba
        import re

        def tokenize(text):
            """中英文混合分词：jieba 切词 + 英文整词保留"""
            tokens = set()
            # 先用正则提取完整英文单词（如 FastAPI, API, Python）
            for match in re.finditer(r'[a-zA-Z][a-zA-Z0-9_]+', text):
                tokens.add(match.group().lower())
            # jieba 切分中文
            for word in jieba.cut(text):
                word = word.strip().lower()
                if word and len(word) > 1 and not word.isascii():
                    tokens.add(word)
            return tokens

        def compute_content_coverage(docs, query):
            query_words = tokenize(query)
            if not query_words:
                return 0
            overlap_counts = 0
            for doc in docs:
                text = doc.get("content", {}).get("chunk_text", "")[:500]
                doc_words = tokenize(text)
                if query_words & doc_words:
                    overlap_counts += 1
            return overlap_counts / len(docs) if docs else 0

        content_coverage=compute_content_coverage(docs, query)

        # ═══════════════════════════════════════════════════════════
        # 加权融合（三维评分）
        # ═══════════════════════════════════════════════════════════
        final_score=(
            avg_rerank*0.60+           # Reranker 主信号
            multi_source_score*0.25+   # 多源有效文档
            content_coverage*0.15      # 内容覆盖度
        )

        score=round(final_score,4)

        # ═══════════════════════════════════════════════════════════
        # 详细日志
        # ═══════════════════════════════════════════════════════════
        logger.info(
            f"[trace:{trace_id}] [self_rag_node] "
            f"混合评分: final={score} "
            f"rerank={avg_rerank:.3f} multi_source={multi_source_score:.3f} "
            f"content_cov={content_coverage:.3f} "
            f"docs={len(docs)} "
            f"duration={time.time()*1000-start_ms}ms"
        )

        return {"self_rag_score":score}

    async def answer_node(state:AgentState)->dict:
        """
        Answer节点：生成最终回答

        兜底逻辑：
        - 如果检索文档质量差，输出"信息不足"模板而非强行编造
        """
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""

        history_context=build_context(state)
        effective_query=get_last_user_message(state["messages"]) or query

        # 检查检索文档质量
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []

        # 文档质量检查
        has_quality_docs=False
        if docs:
            # 检查是否有高质量文档（rerank_score > 0.3）
            for doc in docs:
                score=doc.get("rerank_score") or doc.get("score",0)
                if score>0.3:
                    has_quality_docs=True
                    break

        # 如果文档质量差，生成兜底回答
        if not has_quality_docs and docs:
            # 构建参考文档摘要
            doc_snippets=[]
            for i,doc in enumerate(docs[:3],1):
                text=doc.get("content",{}).get("chunk_text","")[:200]
                doc_snippets.append(f"【片段{i}】{text}")
            docs_summary="\n".join(doc_snippets)

            fallback_response=(
                f"当前检索未找到足够匹配的信息，以下为检索到的相关参考内容：\n\n"
                f"{docs_summary}\n\n"
                f"您可以尝试调整问题描述重新提问。"
            )
            logger.info(
                f"[trace:{trace_id}] [answer_node] "
                f"文档质量不足，返回兜底回答 duration={time.time()*1000-start_ms}ms"
            )
            return {"messages":[AIMessage(content=fallback_response)]}

        # 正常回答流程
        prompt_messages=ANSWER_PROMPT.format_messages(history=history_context,query=effective_query)
        messages=prompt_messages+state.get("messages",[])
        resp=await get_llm(temperature=0.7).ainvoke(messages)

        logger.info(
            f"[trace:{trace_id}] [answer_node] "
            f"生成回答: len={len(resp.content)} duration={time.time()*1000-start_ms}ms"
        )
        return {"messages":[AIMessage(content=resp.content)]}

    async def human_intervention_node(state:AgentState)->dict:
        """
        HITL人工介入节点（简化设计）

        功能：
        - 展示检索结果和评分
        - 提供两个选择：
          1. edit_query: 修改query后继续检索（保留所有状态）
          2. force_answer: 直接生成回答（跳过检索）
        """
        start_ms=time.time()*1000
        trace_id=state.get("trace_id","")
        query=state.get("current_query") or ""
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []

        if docs:
            context_parts=[]
            for i,doc in enumerate(docs[:3],1):
                text=doc.get("content",{}).get("chunk_text","")[:300]
                score=doc.get("rerank_score") or doc.get("score",0)
                score_label="rerank" if doc.get("rerank_score") is not None else "vec"
                context_parts.append(f"[{i}] ({score_label}:{score:.2f}) {text}")
            context="\n".join(context_parts)
        else:
            context="未检索到相关文档"

        prompt=HUMAN_CONFIRMATION_PROMPT.format(query=query,context=context)
        resp=await get_llm(temperature=0.7).ainvoke(prompt)
        confirmation_text=str(resp.content)

        # interrupt返回用户选择：{mode: "edit_query"|"force_answer", edited_query?: string}
        user_choice=interrupt({
            "question":query,
            "context":context,
            "confirmation_text":confirmation_text,
            "self_rag_score":state.get("self_rag_score",0.0),
            "options":["edit_query","force_answer"],
            "status":"pending"
        })

        hitl_mode=user_choice.get("mode","edit_query") if isinstance(user_choice,dict) else "edit_query"
        human_edited_query=user_choice.get("edited_query") if isinstance(user_choice,dict) else None

        logger.info(
            f"[trace:{trace_id}] [human_intervention] "
            f"人工介入选择 mode={hitl_mode} "
            f"duration={time.time()*1000-start_ms}ms"
        )
        return {
            "hitl_mode":hitl_mode,
            "human_edited_query":human_edited_query
        }

    async def rewrite_query_node(state:AgentState)->dict:
        """
        Agent 改写节点（状态层面）

        职责：
        1. 基于当前 query 和检索结果生成新的 query
        2. 增加重试计数
        3. 更新 state 中的 current_query

        流程：
        self_rag评分中等 → rewrite_query_node(改写query) → assistant → rag_search(多路召回)
        """
        retry=(state.get("retry_count") or 0)+1
        original=state.get("current_query") or ""

        # 获取上一轮检索结果作为参考
        rag_result=get_last_tool_result(state["messages"],"rag_search") or {}
        docs=rag_result.get("docs") or []
        docs_text="\n".join(d.get("content",{}).get("chunk_text","")[:200] for d in docs[:3])

        current_query=original
        try:
            prompt=QUERY_REWRITE_PROMPT.format(
                original_query=original,
                result_summary=docs_text
            )
            resp=await get_llm(temperature=0.3).ainvoke(prompt)
            new_query=str(resp.content).strip()
            if new_query:
                current_query=new_query
            logger.info(
                f"[trace:{state.get('trace_id')}] [rewrite_query_node] "
                f"Agent改写完成 retry={retry} "
                f"original='{original[:30]}' new='{current_query[:30]}'"
            )
        except Exception as e:
            logger.warning(
                f"[trace:{state.get('trace_id')}] [rewrite_query_node] "
                f"改写失败，使用原始查询 retry={retry} error={e}"
            )

        return {"retry_count":retry,"current_query":current_query}

    async def edit_query_node(state:AgentState)->dict:
        """
        人工修改query节点

        处理人工介入时用户修改的query：
        - 更新 current_query
        - 不递增 retry_count（人工修改不计入自动重试）
        - 保留全部状态
        """
        trace_id=state.get("trace_id","")
        human_edited_query=state.get("human_edited_query")
        original_query=state.get("current_query") or ""

        # 优先使用人工编辑的query，否则保留原query
        new_query=human_edited_query if human_edited_query else original_query

        logger.info(
            f"[trace:{trace_id}] [edit_query_node] "
            f"人工修改query original='{original_query[:30]}' "
            f"new='{new_query[:30]}'"
        )
        return {
            "current_query":new_query,
            "human_edited_query":None,  # 清除临时字段
            "hitl_mode":None  # 清除临时字段
        }

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
        "edit_query": edit_query_node,
        "compress": compress_node,
    }




