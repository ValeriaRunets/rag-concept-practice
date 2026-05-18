import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
import bs4

load_dotenv()
client = OpenAI()

start = time.time()

url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=(url,),
    bs_kwargs={"parse_only": bs4_strainer},
    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)


resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=[doc.page_content for doc in all_splits]
)
embeddings = [item.embedding for item in resp.data]

query = "What is Reflexion?"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# PostgreSQL with pgvector
conn = psycopg.connect("postgresql://postgres:pass@localhost:5432/postgres")
register_vector(conn)  

with conn.cursor() as cur:
    cur.execute("DELETE FROM documents WHERE url = %s", (url,))
    for i, (chunk, emb) in enumerate(zip(all_splits, embeddings)):
        cur.execute(
            "INSERT INTO documents (url, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (url, i, chunk.page_content, emb),
        )
conn.commit()

with conn.cursor() as cur:
    cur.execute(
        "SELECT content, 1 - (embedding <=> %s::vector) AS score "
        "FROM documents ORDER BY embedding <=> %s::vector LIMIT 5",
        (query_embedding, query_embedding),
    )
    rows = cur.fetchall()

conn.close()

for content, score in rows:
    print(f"Score: {score:.4f} | {content[:100]}")
# pgvector finished


# Qdrant option
"""qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

if not qdrant.collection_exists("documents"):
    qdrant.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

qdrant.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=i,
            vector=emb,
            payload={"url": url, "chunk_index": i, "content": chunk.page_content},
        )
        for i, (chunk, emb) in enumerate(zip(all_splits, embeddings))
    ],
)

print(f"Loaded {len(all_splits)} documents")

results = qdrant.query_points(
    collection_name="documents",
    query=query_embedding,
    limit=5,
).points

for point in results:
    print(point.score, point.payload)"""
#Qdrant finished


print(f"Total time: {time.time() - start:.2f}s")