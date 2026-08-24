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
embedding_base_url=os.getenv("EMBEDDING_BASE_URL")
embedding_api_key=os.getenv("EMBEDDING_API_KEY")
embedding_model=os.getenv("EMBEDDING_MODEL")
qdrant_collection="docagent"
qdrant_collection_size=1024

knowledge_base_dir=Path(__path__).parent.parent/"data"/"kb"

max_self_rag_retries=3
max_agent_iterations=8
max_history_turns=10

llm_model=os.getenv("EMBEDDING_MODEL")
llm_base_url=os.getenv("OPENAI_BASE_URL")
llm_api_key=os.getenv("OPENAI_API_KEY")
llm_retries=3

context_token_budget=4000
recent_message_count=20
history_item_max_chars=500

langgraph_checkpoint_ttl=604800