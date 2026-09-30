from unittest.mock import patch

from app.rag.graph import ask_question


@patch("app.rag.graph.client")
@patch("app.rag.graph.get_index")
@patch("app.rag.graph.create_embedding")
def test_rag_response_structure(mock_embedding, mock_index, mock_client):
    mock_embedding.return_value = [0.1, 0.2, 0.3]

    mock_index.return_value.query.return_value.matches = [
        type(
            "Match",
            (),
            {
                "score": 0.92,
                "metadata": {
                    "text": "Agentic AI systems can autonomously plan and execute tasks."
                },
            },
        )()
    ]

    mock_client.chat.completions.create.return_value.choices = [
        type(
            "Choice",
            (),
            {
                "message": type(
                    "Message",
                    (),
                    {
                        "content": "Agentic AI systems can autonomously plan and execute tasks."
                    },
                )()
            },
        )()
    ]

    result = ask_question("What is Agentic AI?")

    assert result["query"] == "What is Agentic AI?"
    assert "final_answer" in result
    assert "retrieved_context_chunks" in result
    assert "confidence_score" in result
    assert isinstance(result["retrieved_context_chunks"], list)
    assert 0 <= result["confidence_score"] <= 1