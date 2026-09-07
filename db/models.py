
from sqlalchemy import Column, Integer, ForeignKey, String, Text, DateTime, JSON, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id=Column(Integer,primary_key=True,index=True)
    username=Column(String(50),unique=True,index=True)
    password_hash=Column(String(128))
    role=Column(String(20),default="user")
    created_at=Column(DateTime(timezone=True),server_default=func.now())

class KnowledgeBase(Base):
    __tablename__="knowledge_bases"
    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey("users.id"))
    name=Column(String(100))
    description=Column(Text,nullable=True)
    chunk_count=Column(Integer,default=0)
    doc_count=Column(Integer,default=0)
    created_at=Column(DateTime(timezone=True),server_default=func.now())

class Document(Base):
    __tablename__ = "documents"
    id=Column(Integer,primary_key=True,index=True)
    kb_id=Column(Integer,ForeignKey("knowledge_bases.id"))
    file_name=Column(String(255))
    file_path=Column(String(500))
    file_type=Column(String(20))
    file_size=Column(Integer)
    chunk_count=Column(Integer,default=0)
    status=Column(String(20),default="pending")
    error_msg=Column(Text,nullable=True)
    created_at=Column(DateTime(timezone=True),server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"
    id=Column(Integer,primary_key=True)
    doc_id=Column(Integer,ForeignKey("documents.id"))
    kb_id=Column(Integer,ForeignKey("knowledge_bases.id"))
    content=Column(Text)
    chunk_index=Column(Integer)
    file_name=Column(String(255))
    vector_id=Column(String(64),unique=True)
    metadata_json=Column(JSON,nullable=True,default=dict)
    created_at=Column(DateTime(timezone=True),server_default=func.now())

class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey("users.id"))
    kb_id=Column(Integer,ForeignKey("knowledge_bases.id"),nullable=True)
    title=Column(String(200))
    created_at=Column(DateTime(timezone=True),server_default=func.now())
    updated_at=Column(DateTime(timezone=True),onupdate=func.now())

class AgentMessage(Base):
    __tablename__ = "agent_messages"
    id=Column(Integer,primary_key=True,index=True)
    session_id=Column(Integer,ForeignKey("agent_sessions.id"))
    role=Column(String(20))
    content=Column(Text)
    tool_calls=Column(Text,nullable=True)
    tool_results=Column(Text,nullable=True)
    created_at=Column(DateTime(timezone=True),server_default=func.now())



