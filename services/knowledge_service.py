from typing import Dict, Any

from loguru import logger
from sqlalchemy import select, update, delete

from db.models import KnowledgeBase, AgentSession, Chunk, Document
from services.vector_service import VectorService


#知识库管理服务
class KnowledgeService:
    def __init__(self,db_session_factory,vector_service):
        self.vector_service=vector_service
        self.db_session_factory=db_session_factory

    @staticmethod
    def to_dict(row:KnowledgeBase)->Dict[str,Any]:
        return {
            "id":str(row.id),
            "name":row.name,
            "description":row.description,
            "doc_count":row.doc_count or 0,
            "chunk_count":row.chunk_count or 0,
            "created_at":row.created_at
        }

    async def create_knowledge_base(self, user_id, name, description):
        async with self.db_session_factory() as db_session:
            kb=KnowledgeBase(user_id=user_id,name=name,description=description)
            db_session.add(kb)
            await db_session.commit()
            await db_session.refresh(kb)
            logger.info(f"创建知识库:id={kb.id} name={name} user_id={user_id}")
            return self.to_dict(kb)

    async def list_knowledge_bases(self, user_id, limit, offset):
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(KnowledgeBase).where(KnowledgeBase.user_id==user_id).order_by(KnowledgeBase.created_at.desc())
                .limit(limit).offset(offset)
            )
            rows=result.scalars().all()
            return [self.to_dict(r) for r in rows]


    async def delete_knowledge_base(self, kb_id):
        """需要删除知识库及其所有文档，分块，向量"""
        await self.vector_service.delete_by_kb_id(int(kb_id))
        async with self.db_session_factory() as db_session:
            await db_session.execute(
                update(AgentSession)
                .where(AgentSession.kb_id==int(kb_id))
                .values(kb_id=None)
            )
            await db_session.execute(
                delete(Chunk).where(Chunk.kb_id==int(kb_id))
            )
            await db_session.execute(
                delete(Document).where(Document.kb_id==int(kb_id))
            )
            await db_session.execute(
                delete(KnowledgeBase).where(KnowledgeBase.id==int(kb_id))
            )
            await db_session.commit()
        logger.info(f"删除知识库:kb_id={kb_id}")
        return True


    async def get_knowledge_base(self, kb_id, user_id):
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.id==int(kb_id),
                    KnowledgeBase.user_id==user_id
                )
            )
            row=result.scalar_one_or_none()
            return self.to_dict(row) if row else None