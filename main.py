from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import time

from aiohttp.abc import HTTPException
from fastapi import FastAPI,Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from loguru import  logger

from api import system_router
from config import config
from core.clients import init_db, init_redis, init_vector

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.db_client,app.state.db_session=await init_db()
    app.state.redis_client=await init_redis()
    app.state.vector_client=await init_vector()
    yield
    await app.state.db_client.close()
    await app.state.redis_client.close()
    await app.state.vector_client.close()
app=FastAPI(
    title="DocAgent",
    description="一个企业级知识库系统，包含RAG，Agent",
    version="1.0.0",
    lifespan=lifespan
)



#按照固定窗口计数
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self,app:FastAPI,windows,counts):
        super().__init__(app)
        self.windows = windows
        self.counts=counts
    async def dispatch(self,request:Request,call_next):
        client_ip=request.client.host
        redis_client=request.app.state.redis_client
        key=f"rate_limit:{client_ip}"
        count=await redis_client.incr(key)
        if count==1:
            await redis_client.expire(key,self.windows)
        if count>self.counts:
            return JSONResponse(
                status_code=429,
                content={
                    "error":True,
                    "detail":"请求过于频繁，请稍后重试"
                }
            )
        return await call_next(request)

class RequsetLogMiddleware(BaseHTTPMiddleware):
    def __init__(self,app:FastAPI):
        super().__init__(app)
    async def dispatch(self,request:Request,call_next):
        start=time.time()
        response=call_next(request)
        end=time.time()
        logger.info(f"{request.method}{request.url.path}->{response.status_code}{(end-start)*1000}ms")
        return response

app.add_middleware(RateLimitMiddleware,windows=10,counts=20)
app.add_middleware(RequsetLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def register_exeception_handlers(app:FastAPI)->None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request:Request,exc:HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error":True,
                "detail":exc.detail
            }
        )
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request:Request,exc:Exception):
        logger.error(
            f"未捕获异常:{request.method}{request.url.path} error={exc}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "error":True,
                "detail":"服务器内部错误"
            }
        )

register_exeception_handlers(app)

app.include_router(system_router,prefix="/api/health",tags=["系统"])

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host=config.backend_host,port=config.backend_port,reload=True,reload_dirs=["."])

