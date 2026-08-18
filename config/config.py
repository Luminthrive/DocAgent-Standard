import os
from dotenv import load_dotenv


load_dotenv(override=True)

DB_URL=os.getenv("DB_URL")
REDIS_URL=os.getenv("REDIS_URL")
QDRANT_URL=os.getenv("QDRANT_URL")
backend_host="127.0.0.1"
backend_port="8765"
auth_api_key=os.getenv("API_KEY")
