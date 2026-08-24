from fastapi import FastAPI

from agent_graph.checkpointer import create_checkpointer
from agent_graph.graph_builder import build_graph
from agent_graph.nodes import create_node
from agent_graph.tools import create_tools
from services.agent_service import AgentService
from services.document_service import DocumentService
from services.knowledge_service import KnowledgeService
from services.parse_service import ParseService
from services.session_service import SessionService
from services.vector_service import VectorService


async def init_knowledge_service(app:FastAPI):
    kb_service=KnowledgeService(app.state.db_session_factory,app.state.vector_service)
    app.state.kb_service=kb_service

async def init_vector_service(app:FastAPI):
    vector_service=VectorService(app.state.vector_client)
    app.state.vector_service=vector_service

async def init_parse_service(app:FastAPI):
    parse_service=ParseService()
    app.state.parse_service=parse_service

async def init_document_service(app:FastAPI):
    document_service=DocumentService(
        app.state.parse_service,
        app.state.vector_service,
        app.state.db_session_factory,
    )
    app.state.document_service=document_service

async def init_session_service(app:FastAPI):
    session_service=SessionService(
        app.state.db_session_factory
    )
    app.state.session_service=session_service

async def init_graph(app:FastAPI):
    app.state.tools=create_tools(app.state.vector_service,app.state.kb_service)
    app.state.nodes=create_node(app.state.tools)
    app.state.checkpointer=await create_checkpointer(redis_available=app.state.redis_client is not None)
    app.state.graph=await build_graph(app.state.nodes,app.state.checkpointer)

async def init_agent_service(app:FastAPI):
    agent_service=AgentService(app.state.session_service,app.state.graph)
    app.state.agent_service=agent_service


async def init_services(app:FastAPI):
    await init_vector_service(app)
    await init_knowledge_service(app)
    await init_parse_service(app)
    await init_document_service(app)
    await init_session_service(app)
    await init_graph(app)
    await init_agent_service(app)

async def get_knowledge_service(app:FastAPI):
    return app.state.kb_service

async def get_vector_service(app:FastAPI):
    return app.state.vector_service

async def get_parse_service(app:FastAPI):
    return app.state.parse_service

async def get_document_service(app:FastAPI):
    return app.state.document_service

async def get_session_service(app:FastAPI):
    return app.state.session_service

async def get_agent_service(app:FastAPI):
    return app.state.agent_service