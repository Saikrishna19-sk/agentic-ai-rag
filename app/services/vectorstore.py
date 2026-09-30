from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

DIMENSION = 1536


def get_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_needed():
    pc = get_pinecone()

    existing_indexes = [x["name"] for x in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    return pc.Index(PINECONE_INDEX_NAME)


def get_index():
    return create_index_if_needed()