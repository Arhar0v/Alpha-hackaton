from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Literal, Any
from pydantic import BaseModel, Field, ConfigDict
import psycopg
from psycopg.rows import dict_row

# модель сравнивает векторы последних сообщений с векторами описаний курсов

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
    embedding: np.ndarray = Field(description='Эмбеддинг текста')


description_courses = [] # описание курсов в формате FormatCourses
history = 'Как аннулировать товарный знак?' # история сообщений (str), по которой идет поиск
model = RecomendEmbedModel()


# подключение к БД за таблицами с курсами (Articles, Courses, Lessons)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "alfa_course_hackathon"
DB_USER = "postgres"
DB_PASS = "1234"

try:
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        row_factory=dict_row
    )
    cur = conn.cursor()
    for course in ['Articles', 'Courses']:
        cur.execute(f'SELECT * FROM "{course}";')
        rows = cur.fetchall()
        for row in rows:
            desc_tag = '' if 'DescriptionTag' not in row or row['DescriptionTag'] is None else row['DescriptionTag']
            tit_navig = '' if 'TitleNavigation' not in row or row['TitleNavigation'] is None else row['TitleNavigation']
            description_courses.append(FormatCourses(category=course.lower(),
                                                     title=row['Title'],
                                                     description=desc_tag + tit_navig,
                                                     url=row['Url'],
                                                     embedding=model.embed_query(f'{row["Title"]}. {desc_tag + tit_navig}')))
    cur.close()
    conn.close()

except Exception as e:
    print(f"Ошибка подключения: {e}")

def get_courses(query: str, items: List[FormatCourses],
                model: SentenceTransformer, k: int=3, category: str = None) -> List[FormatCourses]:
    if category:
        items = [i for i in items if i.category == category]
    if not items:
        return []

    matrix = np.vstack([i.embedding for i in items])
    q_emb = model.embed_query(query)
    scores = util.cos_sim(q_emb, matrix)[0]
    top_idx = np.argsort(scores.numpy())[::-1][:k]
    return [(items[i].title, items[i].category) for i in top_idx]

print(get_courses(history, description_courses, model))