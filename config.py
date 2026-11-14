# Импорт библиотек
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Найстройки БД и токена бота в тг
settings = {
    "token": TG_BOT_TOKEN,
    "database": "alfa_course_hackathon",
    "user": "postgres",
    "password": DB_PASSWORD,
    "host": "127.0.0.1",
    "port": "5432"
}


# Директории
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"


# Параметры RAG
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
MAX_DISTANCE = 0.6
N_MSG_HISTORY = 5

# ChromaDB
def _client():
    import chromadb
    from chromadb.config import Settings

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    """Единственная точка доступа к коллекции ChromaDB. Создаёт её при первом вызове."""
    from chromadb.utils import embedding_functions

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

def reset_collection():
    client = _client()
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if COLLECTION_NAME in names:
        client.delete_collection(COLLECTION_NAME)
    return get_collection()

