import json
from datetime import datetime
from typing import Dict, Any
from loguru import logger
from sqlalchemy import select, delete

from db.models import AgentSession, AgentMessage


class SessionService:
    def __init__(self,db_session_factory):
        self.db_session_factory=db_session_factory
    @staticmethod
    def to_dict(row:AgentSession)->Dict[str,Any]:
        return {
            "id":str(row.id),
            "title":row.title,
            "kb_id":str(row.kb_id) if row.kb_id is not None else None,
            "created_at":row.created_at,
            "updated_at":row.updated_at
        }
    @staticmethod
    def message_to_dict(row:AgentMessage)->Dict[str,Any]:
        return {
            "id":str(row.id),
            "session_id":str(row.session_id),
            "role":row.role,
            "content":row.content or "",
            "tool_calls":json.loads(row.tool_calls) if row.tool_calls else None,
            "tool_results":json.loads(row.tool_results) if row.tool_results else None,
            "created_at":row.created_at
        }

    async def create_session(self, user_id, kb_id, title):
        if not title:
            title=f"会话{datetime.now().strftime('%Y-%m-%d %H:%M')}"

        async with self.db_session_factory() as db_session:
            row=AgentSession(
                user_id=user_id,
                kb_id=int(kb_id) if kb_id else None,
                title=title
            )
            db_session.add(row)
            await db_session.commit()
            await db_session.refresh(row)
            logger.info(f"创建会话:id={row.id} user_id={user_id} kb_id={kb_id}")
            return self.to_dict(row)

    async def list_sessions(self, user_id, limit, offset):
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(AgentSession)
                .where(AgentSession.user_id==user_id)
                .order_by(AgentSession.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self.to_dict(r) for r in result.scalars().all()]

    async def get_session(self, session_id, user_id):
        async with self.db_session_factory() as db_session:
            result = await db_session.execute(
                select(AgentSession).where(
                    AgentSession.id == int(session_id),
                    AgentSession.user_id==user_id
                )
            )
            row=result.scalar_one_or_none()
            return self.to_dict(row) if row else None

    async def delete_session(self, session_id):
        async with self.db_session_factory() as db_session:
            await db_session.execute(
                delete(AgentMessage)
                .where(AgentMessage.session_id==int(session_id))
            )
            await db_session.execute(
                delete(AgentSession)
                .where(AgentSession.id==int(session_id))
            )
            await db_session.commit()
        logger.info(f"删除会话: session_id={session_id} ")
        return True

    async def list_messages(self, session_id):
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(AgentMessage)
                .where(AgentMessage.session_id==int(session_id))
                .order_by(AgentMessage.created_at.asc())
            )
            return [self.message_to_dict(r) for r in result.scalars().all()]

    async def create_message(self,session_id,role,content,tool_calls=None,tool_results=None):
        async with self.db_session_factory() as db_session:
            row=AgentMessage(
                session_id=int(session_id),
                role=role,
                content=content,
                tool_calls=json.dumps(tool_calls,ensure_ascii=False) if tool_calls else None,
                tool_results=json.dumps(tool_results,ensure_ascii=False) if tool_results else None
            )
            db_session.add(row)
            await db_session.commit()
            await db_session.refresh(row)
            return self.message_to_dict(row)



