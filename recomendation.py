from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict

# Модель сравнивает векторы последних сообщений с векторами описаний курсов
class RecomendEmbedModel:
    def __init__(self, model_name: str="sergeyzh/LaBSE-ru-turbo"):
        self.model = SentenceTransformer(model_name)

    def get_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_docs(self, docs: list[str]) -> np.ndarray:
        return self.model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)

class FormatCourses(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    title: str = Field(description='Название курса')
    description: str = Field(description='Описание курса')
    category: Literal["courses", "articles"] = Field(description='Тип образовательных материалов')
    url: str = Field(description='Ссылка на ресурс')
    embedding: List = Field(description='Эмбеддинг текста')