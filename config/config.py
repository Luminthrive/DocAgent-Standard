import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)

DB_URL=os.getenv("DB_URL")
REDIS_URL=os.getenv("REDIS_URL")
QDRANT_URL=os.getenv("QDRANT_URL")
backend_host="127.0.0.1"
backend_port="8765"
auth_api_key=os.getenv("API_KEY")
# 模型服务器配置 (Embedding + Rerank + BM25)
model_server_url=os.getenv("MODEL_SERVER_URL","http://localhost:8001")
embedding_model=os.getenv("EMBEDDING_MODEL","BAAI/bge-small-zh-v1.5")
rerank_model=os.getenv("RERANK_MODEL","BAAI/bge-reranker-v2-m3")
sparse_embedding_model=os.getenv("SPARSE_EMBEDDING_MODEL","Qdrant/bm25")
qdrant_collection="docagent"
qdrant_collection_size=1024

knowledge_base_dir=Path(__file__).parent.parent/"data"/"kb"

max_self_rag_retries=3
max_agent_iterations=8
max_history_turns=10

# Self-RAG 混合评分配置
rag_score_low_threshold=float(os.getenv("RAG_SCORE_LOW","0.25"))   # 低于此分数→人工干预
rag_score_mid_threshold=float(os.getenv("RAG_SCORE_MID","0.45"))   # 低于此分数→改写查询
# 高于此分数→直接回答

# 多路召回配置
enable_multi_rewrite=os.getenv("ENABLE_MULTI_REWRITE","true").lower()=="true"

llm_model=os.getenv("LLM_MODEL")
llm_base_url=os.getenv("OPENAI_BASE_URL")
llm_api_key=os.getenv("OPENAI_API_KEY")
llm_retries=3

context_token_budget=4000
recent_message_count=20
history_item_max_chars=500

langgraph_checkpoint_ttl=604800