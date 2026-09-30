from typing import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from app.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    TOP_K,
    RETRIEVAL_THRESHOLD,
)

from app.services.embeddings import create_embedding
from app.services.vectorstore import get_index


client = OpenAI(api_key=OPENAI_API_KEY)


class RAGState(TypedDict):
    query: str
    context: list[str]
    scores: list[float]
    answer: str
    confidence: float


def retrieve_node(state: RAGState):
    query = state["query"]

    embedding = create_embedding(query)

    index = get_index()

    result = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    context = []
    scores = []

    for match in result.matches:
        score = float(match.score)

        if score >= RETRIEVAL_THRESHOLD:
            metadata = match.metadata or {}
            text = metadata.get("text", "")

            if text:
                context.append(text)
                scores.append(score)

    return {
        "context": context,
        "scores": scores,
    }


def generate_node(state: RAGState):
    context = state["context"]

    if not context:
        return {
            "answer": (
                "I don't have enough information in the "
                "provided eBook to answer this question."
            ),
            "confidence": 0.0,
        }

    joined_context = "\n\n---\n\n".join(context)

    prompt = f"""
You are a strict document-grounded question answering system.

Answer the user's question ONLY using the supplied context.

Do not use outside knowledge.

If the context does not contain enough information to answer,
say exactly:

"I don't have enough information in the provided eBook to answer this question."

Context:
{joined_context}

Question:
{state["query"]}

Answer:
"""

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions strictly from supplied "
                    "document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content.strip()

    average_score = sum(state["scores"]) / len(state["scores"])

    confidence = max(
        0.0,
        min(1.0, average_score)
    )

    return {
        "answer": answer,
        "confidence": round(confidence, 3),
    }


graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


def ask_question(query: str):
    result = graph.invoke(
        {
            "query": query,
            "context": [],
            "scores": [],
            "answer": "",
            "confidence": 0.0,
        }
    )

    return {
        "query": query,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["confidence"],
    }