from typing import Any

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from loguru import logger
from agent_graph.edges import router, tools_router, self_rag_router, human_review_router, compress_router
from agent_graph.state import AgentState


async def build_graph(nodes:dict,checkpointer:Any=None):
    workflow=StateGraph(state_schema=AgentState)
    workflow.add_node("assistant",nodes["assistant"])
    workflow.add_node("tools",nodes["tools"])
    workflow.add_node("self_rag",nodes["self_rag"])
    workflow.add_node("answer",nodes["answer"])
    workflow.add_node("human_intervention",nodes["human_intervention"])
    workflow.add_node("rewrite_query",nodes["rewrite_query"])
    workflow.add_node("human_review_reset",nodes["human_review_reset"])
    workflow.add_node("compress",nodes["compress"])

    workflow.add_edge(START,"assistant")
    workflow.add_conditional_edges("assistant",router,{
        "tools":"tools",
        "answer":"answer"
    })
    workflow.add_conditional_edges("tools",tools_router,{
        "self_rag":"self_rag",
        "assistant":"assistant"
    })
    workflow.add_conditional_edges("self_rag",self_rag_router,{
        "rewrite_and_retry":"rewrite_query",
        "human_intervention":"human_intervention",
        "answer":"answer"
    })
    workflow.add_edge("rewrite_query","assistant")
    workflow.add_conditional_edges("human_intervention",human_review_router,{
        "reset_and_retry":"human_review_reset",
        "end":END
    })
    workflow.add_edge("human_review_reset","assistant")
    workflow.add_conditional_edges("answer",compress_router,{
        "compress":"compress",
        "end":END
    })
    workflow.add_edge("compress",END)
    graph=workflow.compile(checkpointer=checkpointer)
    logger.info("LangGraph StateGraph 构建完成（Agentic RAG, 8 nodes）")
    return graph