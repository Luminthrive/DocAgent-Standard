
from fastapi import APIRouter,Request
from sqlalchemy import text
from schema.common_schema import HealthResponse

router =APIRouter()

@router.get("",response_model=HealthResponse,summary="健康检查")
async def health_check(request:Request):
    result = {"services": {}}
    try:
        async with request.app.state.db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        result["services"]["db"]="healthy"
    except Exception as e:
        result["services"]["db"]=f"error:{e}"

    try:
        redis_client=request.app.state.redis_client
        await redis_client.ping()
        result["services"]["redis"]="healthy"
    except Exception as e:
        result["services"]["redis"]=f"error{e}"

    try:
        vector_client=request.app.state.vector_client
        await vector_client.get_collections()
        result["services"]["vector"]="healthy"
    except Exception as e:
        result["services"]["vector"]=f"error{e}"

    status="ok" if all(v=="healthy" for v in result["services"].values()) else "unhealthy"
    return HealthResponse(
        status=status,
        result=result
    )

