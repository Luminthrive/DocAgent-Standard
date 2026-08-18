from typing import Optional

from fastapi import Header, HTTPException

from config import config


async def get_current_user_id(x_api_key:Optional[str]=Header(None))->int:
    api_key=config.auth_api_key
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(
            status_code=401,
            detail="缺少x-api-key header或值错误",
        )
    return 1