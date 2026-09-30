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

    if PDF_PATH.stat().st_size == 0:
        print("ERROR: Ebook-Agentic-AI.pdf is empty.")
        print(
            "Please place the actual assessment PDF in "
            "data\\Ebook-Agentic-AI.pdf"
        )
        return 0

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

    if not vectors:
        raise ValueError(
            "No text chunks could be extracted from the PDF."
        )

    index = get_index()

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

    print(f"Ingested {len(vectors)} chunks.")

    return len(vectors)


if __name__ == "__main__":
    ingest_pdf()