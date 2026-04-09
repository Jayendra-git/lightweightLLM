# Embeddings

Embeddings are vector representations of text. The goal is that semantically similar text ends up close together in vector space. This allows a retrieval system to compare a user question with document chunks numerically rather than with exact keyword matching.

Sentence-transformer models are a common choice for small local projects because they are easy to run and often give good semantic search quality out of the box. In a mini RAG system, each markdown chunk is embedded once during ingestion and stored with its source metadata.

At query time, the same embedding model converts the user question into a vector. The system then searches for the nearest stored vectors and returns the corresponding chunks. Those chunks become context for the language model.

