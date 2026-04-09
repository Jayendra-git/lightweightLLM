# Architecture Notes

The project is intentionally split into three narrow concerns:

- `train/` handles style-only LoRA fine-tuning.
- `rag/` handles local markdown ingestion and semantic retrieval.
- `app/` handles serving and presenting the chat experience.

At inference time, the request path is:

1. Streamlit sends a question to FastAPI.
2. FastAPI embeds the question and retrieves top chunks from Chroma.
3. The prompt builder injects the chunks into a grounded tutor-style prompt.
4. The base instruct model generates a response with the LoRA adapter loaded when present.
5. The API returns the answer plus retrieved sources to the UI.

This separation makes the portfolio value legible: model adaptation, retrieval, and product surface are all visible without adding unnecessary infrastructure.
