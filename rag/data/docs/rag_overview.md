# Retrieval-Augmented Generation Overview

Retrieval-Augmented Generation, or RAG, combines search with text generation. Before the model answers, the system looks up relevant passages from a local document collection and places those passages into the prompt. The language model then uses that retrieved context as grounding material.

This is especially helpful when the model should reference project-specific notes, internal documentation, or fresh content that was not part of pretraining. In a small local demo, the document collection can be tiny. Even a handful of markdown files is enough to show that retrieval changes the answer and makes source attribution possible.

RAG does not guarantee correctness. Retrieval can miss the best passage, and the model can still overstate weak evidence. A solid demo therefore shows both the answer and the retrieved sources so users can inspect what information the system actually had available.

