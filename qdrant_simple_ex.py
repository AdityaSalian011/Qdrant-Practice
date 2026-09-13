import shutil

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

shutil.rmtree("./vector_store")

client = QdrantClient(path="./vector_store")

client.create_collection(
    collection_name="first_collection",
    vectors_config=VectorParams(size=2, distance=Distance.EUCLID)
)

client.upsert(
    collection_name="first_collection",
    wait=True,
    points=[
        PointStruct(id=1, vector=[0.1, 0.1], payload={"role": "Boss"}),
        PointStruct(id=2, vector=[-0.1, 0.1], payload={"role": "Worker"}),
        PointStruct(id=3, vector=[5, 5], payload={"role": "King"}),
        PointStruct(id=4, vector=[20, 20], payload={"role": "God"}),
    ]
)

results = client.query_points(
    collection_name="first_collection",
    query=[-0.08, 0.09],    
    with_payload=True,
    limit=5
).points

print(results)

client.close()