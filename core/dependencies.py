from fastapi import FastAPI

from services.document_service import DocumentService
from services.knowledge_service import KnowledgeService
from services.parse_service import ParseService
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


async def init_services(app:FastAPI):
    await init_vector_service(app)
    await init_knowledge_service(app)
    await init_parse_service(app)
    await init_document_service(app)

async def get_knowledge_service(app:FastAPI):
    return app.state.kb_service

async def get_vector_service(app:FastAPI):
    return app.state.vector_service

async def get_parse_service(app:FastAPI):
    return app.state.parse_service

async def get_document_service(app:FastAPI):
    return app.state.document_service