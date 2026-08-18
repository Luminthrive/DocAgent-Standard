from urllib.request import Request

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
    session=async_sessionmaker(bind=db_engine)
    return db_engine,session

async def init_redis():
    redis_client=Redis.from_url(config.REDIS_URL)
    return redis_client

async def init_vector():
    vector_client=AsyncQdrantClient(config.QDRANT_URL)
    return vector_client

async def get_db_client(request:Request)->AsyncEngine:
    return request.state.db_engine

async def get_redis(request:Request)->Redis:
    return request.state.redis_client

async def get_vector(request:Request)->AsyncQdrantClient:
    return request.state.vector_client

async def get_db_session(request:Request)->AsyncSession:
    session_factory=request.state.db_session
    async with session_factory() as session:
        yield session