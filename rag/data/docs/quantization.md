# Quantization

Quantization reduces model memory usage by storing weights with fewer bits, such as 8-bit or 4-bit values instead of 16-bit or 32-bit values. This can make local inference or fine-tuning more accessible on limited hardware.

The tradeoff is that lower precision can introduce some quality loss or compatibility issues depending on the model, tooling, and hardware. For a smallest-credible demo, it is reasonable to keep quantization optional rather than making it part of the core setup.

If a project later needs to scale down memory further, quantization can be added as an implementation detail around model loading. It is not required for proving that LoRA and RAG both work in the same application.

