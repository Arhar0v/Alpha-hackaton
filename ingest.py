import re
import hashlib
from bs4 import BeautifulSoup
import config
from main import create_connection, db_connection

mode = "eval"


def clean_text(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    clean = soup.get_text(strip=False)
    return clean

def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError(f"Overlap ({overlap}) должен быть меньше size ({size})")
    if len(text) <= size:
        return [text]
    step = size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        chunks.append(chunk)
        if i + size >= len(text):
            break
    return chunks

# Сборка индекса
def ingest():
    collection = config.reset_collection()
    total = 0
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute('SELECT a."Id", a."Title", a."Description", a."DescriptionTag", asub."Title" AS rubric, aei.htmlcontent '
                           'FROM "Articles" AS a LEFT JOIN "ArticleSubjects" AS asub ON a."SubjectId" = asub."Id" LEFT JOIN '
                           '(SELECT "ArticleId", STRING_AGG("HtmlContent", \'. \') AS htmlcontent FROM "ArticleEmbedItems" '
                           'GROUP BY "ArticleId") AS aei ON a."Id" = aei."ArticleId";')
            articles_tuple = cursor.fetchall()
    for i, row in enumerate(articles_tuple):
        article_id, title, description, descriptiontag, rubric, htmlcontent = row
        cleaned = clean_text(htmlcontent)
        res_text = " ".join(filter(None, [title, description, descriptiontag, cleaned]))
        chunked_text = chunk_text(res_text)
        for chunk in chunked_text:
            chunk_id = hashlib.sha1(f"{title}:{chunk}".encode("utf-8")).hexdigest()
            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{
                    "source": "Articles",
                    "id": article_id,
                    "title": title,
                    "category": rubric
                }],
            )
            total += 1
        if i % 50 == 0:
            print(i + 1, "/", len(articles_tuple))
    return total


if __name__ == "__main__":
    if mode == "eval":
        count = ingest()
        print(f"Индекс пересобран заново. Проиндексировано чанков: {count}")
        print(f"Всего в коллекции: {config.get_collection().count()}")

