# Pokémon RAG Space

This Space runs a FastAPI backend that serves a RAG model for answering Pokémon queries.

Send POST requests to `/query` endpoint:

```json
{
  "query": "What is Pikachu?"
}