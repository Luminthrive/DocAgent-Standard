from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app=FastAPI(
    title="DocAgent",
    description="一个企业级知识库系统，包含RAG，Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

