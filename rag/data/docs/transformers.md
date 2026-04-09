# Transformers

Transformer models process text with attention rather than relying on recurrence. Attention lets each token weigh the relevance of other tokens in the sequence, which helps the model track dependencies that may be far apart. This is one reason transformers became dominant for language tasks.

In practical terms, a transformer-based chatbot turns user text into tokens, maps those tokens through many stacked layers, and predicts likely next tokens. Instruction-tuned versions are optimized to follow task-oriented prompts more reliably than raw base models.

When developers fine-tune with LoRA, they usually attach the adapter to transformer layers rather than replacing the overall architecture. That makes transformers a natural fit for parameter-efficient adaptation.

