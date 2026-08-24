#文档处理服务
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from sqlalchemy import select, delete

from db.models import Document, Chunk, KnowledgeBase
from config import config

class DocumentService:
    def __init__(self,parse_service,vector_service,db_session_factory):
        self.parse_service=parse_service
        self.vector_service=vector_service
        self.db_session_factory=db_session_factory

    @staticmethod
    def to_dict(row:Document)->Dict[str,Any]:
        return {
            "id":str(row.id),
            "kb_id":str(row.kb_id),
            "file_name":row.file_name,
            "file_type":row.file_type,
            "file_size":row.file_size,
            "chunk_count":row.chunk_count or 0,
            "status":row.status,
            "error_msg":row.error_msg,
            "created_at":row.created_at
        }

    @staticmethod
    def chunk_split(paragraphs):
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""]
        )
        chunks=splitter.split_documents(paragraphs)
        return chunks

    async def upload_document(self, kb_id, file, user_id):
        #1 保存文件到磁盘
        kb_dir=config.knowledge_base_dir/str(kb_id)
        kb_dir.mkdir(parents=True,exist_ok=True)
        file_path=kb_dir/file.filename
        content=await file.read()
        with open(file_path,"wb") as f:
            f.write(content)
        logger.info(f"保存文件: {file_path} size={len(content)}")
        file_type="."+file.filename.rsplit(".",1)[-1].lower() if "." in file.filename else ""


        async with self.db_session_factory() as db_session:
            #2 插入documents行
            doc=Document(
                kb_id=int(kb_id),
                file_name=file.filename,
                file_path=str(file_path),
                file_type=file_type,
                file_size=len(content),
                status="processing"
            )
            db_session.add(doc)
            await db_session.flush()
            doc_id=doc.id


            #3 解析+分块
            paragraphs=await self.parse_service.parse_document(str(file_path),file_type)
            chunks=self.chunk_split(paragraphs)
            if not chunks:
                doc.status="failed"
                doc.error_msg="文档解析为空，暂不支持该类型"
                await db_session.commit()
                return {"doc_id": str(doc_id), "file_name": file.filename,
                        "status": "failed", "message": doc.error_msg}


            #4 BGE_M3编码 +写入Qdrant （返回chunk的qdrant point id）
            try:
                chunk_texts=[c.page_content for c in chunks]
                point_ids=await self.vector_service.add_vector(
                    kb_id=str(kb_id),
                    doc_id=str(doc_id),
                    texts=chunk_texts,
                    file_name=file.filename
                )
            except Exception as e:
                logger.info(f"向量化失败:kb_id={kb_id} doc_id={doc_id} error={e}")
                doc.status="failed"
                doc.error_msg=f"向量化失败: {e}"
                await db_session.commit()
                return {"doc_id": str(doc_id), "file_name": file.filename,
                        "status": "failed", "message": doc.error_msg}


            #5 逐chunk插入chunks映射
            for idx,chunk_doc in enumerate(chunks):
                chunk=Chunk(doc_id=doc_id,kb_id=int(kb_id),content=chunk_doc.page_content,chunk_index=idx,
                            file_name=file.filename,vector_id=point_ids[idx])
                db_session.add(chunk)
                await db_session.flush()


            #6 更新文档状态+知识库计数
            doc.status="completed"
            doc.chunk_count=len(chunks)
            kb=await db_session.get(KnowledgeBase,int(kb_id))
            if kb:
                kb.doc_count=(kb.doc_count or 0)+1
                kb.chunk_count=(kb.chunk_count or 0)+len(chunks)


            await db_session.commit()
            logger.info(f"上传完成，kb_id={kb_id} doc_id={doc_id} chunks={len(chunks)}")

            return {
                "doc_id":str(doc_id),
                "file_name":file.filename,
                "status":"completed",
                "chunk_count":len(chunks),
                "message":"文档已上传并完成向量化"
            }


    async def list_documents(self, kb_id):
        kb_id=int(kb_id)
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(Document)
                .where(Document.kb_id==kb_id)
                .order_by(Document.created_at.desc())
            )
            return [self.to_dict(r) for r in result.scalars().all()]

    async def get_document(self, doc_id, kb_id):
        kb_id,doc_id=int(kb_id),int(doc_id)
        async with self.db_session_factory() as db_session:
            result=await db_session.execute(
                select(Document)
                .where(Document.id==doc_id,Document.kb_id==kb_id)
            )
            row=result.scalar_one_or_none()
            return self.to_dict(row) if row else None

    async def delete_document(self, kb_id, doc_id):
        doc_id,kb_id=int(doc_id),int(kb_id)
        await self.vector_service.delete_by_doc_id(str(kb_id),str(doc_id))
        async with self.db_session_factory() as db_session:
            await db_session.execute(
                delete(Chunk)
                .where(Chunk.doc_id==doc_id)
            )
            await db_session.execute(
                delete(Document)
                .where(Document.id==doc_id,Document.kb_id==kb_id)
            )
            await db_session.commit()
        logger.info(f"删除文档:kb_id={kb_id} doc_id={doc_id} ")
        return True
