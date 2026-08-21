
from fastapi import Request, FastAPI
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from config import config
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine


async def init_db():
    db_engine=create_async_engine(
        url=config.DB_URL,
        pool_size=10,
        max_overflow=20,
    )
    session_factory=async_sessionmaker(bind=db_engine)
    return db_engine,session_factory

async def init_redis():
    redis_client=Redis.from_url(config.REDIS_URL)
    return redis_client

async def init_vector():
    vector_client=AsyncQdrantClient(config.QDRANT_URL)
    return vector_client

async def init_client(app:FastAPI):
    app.state.db_engine, app.state.db_session_factory = await init_db()
    app.state.redis_client = await init_redis()
    app.state.vector_client = await init_vector()

async def close_client(app:FastAPI):
    await app.state.db_engine.dispose()
    await app.state.redis_client.close()
    await app.state.vector_client.close()


async def get_redis(request:Request):
    return request.app.state.redis_client

async def get_vector(request:Request):
    return request.app.state.vector_client

async def get_db_session(request:Request):
    session_factory=request.app.state.db_session_factory
    async with session_factory() as session:
        yield session