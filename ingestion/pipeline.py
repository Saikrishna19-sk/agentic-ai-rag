from pathlib import Path

from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.services.embeddings import create_embedding
from app.services.vectorstore import get_index


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")


def chunk_text(text: str) -> list[str]:
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - CHUNK_OVERLAP

    return chunks


def ingest_pdf():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    reader = PdfReader(str(PDF_PATH))

    vectors = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):
            vector_id = f"page-{page_number}-chunk-{chunk_number}"

            embedding = create_embedding(chunk)

            vectors.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "page": page_number,
                        "chunk": chunk_number,
                        "source": PDF_PATH.name,
                    },
                }
            )

    index = get_index()

    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Ingested {len(vectors)} chunks.")
    return len(vectors)


if __name__ == "__main__":
    ingest_pdf()