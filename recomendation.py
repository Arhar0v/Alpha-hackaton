from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Literal, Any
from pydantic import BaseModel, Field, ConfigDict
import psycopg2
from psycopg2.extras import RealDictCursor
import config
from config import settings
from main import create_connection, db_connection


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


description_courses = [] # описание курсов в формате FormatCourses
history = 'Как аннулировать товарный знак?' # история сообщений (str), по которой идет поиск
model = RecomendEmbedModel()


# Подключение к БД за таблицами с курсами (Articles, Courses, Lessons)
with db_connection() as connection:
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        for course in ['Articles', 'Courses']:
            cursor.execute(f'SELECT * FROM "{course}" LIMIT {config.TOP_K};')
            rows = cursor.fetchall()
print(rows)
for row in rows:
    desc_tag = '' if 'DescriptionTag' not in row or row['DescriptionTag'] is None else row['DescriptionTag']
    tit_navig = '' if 'TitleNavigation' not in row or row['TitleNavigation'] is None else row['TitleNavigation']
    print(desc_tag, tit_navig)
    description_courses.append(FormatCourses(category=course.lower(),
                                             title=row['Title'],
                                             description=desc_tag + tit_navig,
                                             url=row['Url'],
                                             embedding=model.embed_query(f'{row["Title"]}. {desc_tag + tit_navig}')))


def get_courses(query: str, items: List[FormatCourses], model: SentenceTransformer, k: int=3, category: str = None) -> List[FormatCourses]:
    if category:
        items = [i for i in items if i.category == category]
    if not items:
        return []

    matrix = np.vstack([i.embedding for i in items])
    q_emb = model.embed_query(query)
    scores = util.cos_sim(q_emb, matrix)[0]
    top_idx = np.argsort(scores.numpy())[::-1][:k]
    return [items[i].title for i in top_idx]

print(get_courses(history, description_courses, model))