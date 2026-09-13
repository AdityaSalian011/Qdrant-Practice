import uuid
import shutil

from openai import OpenAI
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

shutil.rmtree("./vector_store")

client = QdrantClient(path="./vector_store")

openai_client = OpenAI(base_url="https://api.aicredits.in/v1")

interpreted_languages = [
    "Python",
    "JavaScript",
    "Ruby",
    "PHP",
    "Perl",
    "Lua",
    "R",
    "Shell"
]

compiled_languages = [
    "C",
    "C++",
    "Rust",
    "Go",
    "Swift",
    "Fortran",
    "COBOL",
    "D"
]

interpreted_embeddings = [openai_client.embeddings.create(input=lang, model="text-embedding-3-small").data[0].embedding for lang in interpreted_languages]

compiled_embeddings = [openai_client.embeddings.create(input=lang, model="text-embedding-3-small").data[0].embedding for lang in compiled_languages]


client.create_collection(
    collection_name="languages",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

client.upsert(
    collection_name="languages",
    wait=True,
    points=[
        PointStruct(
            id=uuid.uuid4(),
            vector=interpreted_embeddings[i],
            payload={"language": interpreted_languages[i], "type": "interpreted"}
        )
        for i in range(len(interpreted_languages))
    ]
)

client.upsert(
    collection_name="languages",
    wait=True,
    points=[
        PointStruct(
            id=uuid.uuid4(),
            vector=compiled_embeddings[i],
            payload={"language": compiled_languages[i], "type": "compiled"}
        )
        for i in range(len(compiled_languages))
    ]
)


query = "C#"
query_embedding = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding

results = client.query_points(
    collection_name="languages",
    query=query_embedding,    
    with_payload=True,
    limit=5
).points

print(results)

client.close()
