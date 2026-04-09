# Fine-Tuning Workflow

A clean fine-tuning workflow starts by defining the narrow behavior you want to change. In this project, the target is style rather than factual knowledge. That matters because it keeps the training task honest: the adapter should change tone and response structure, while the RAG layer supplies factual grounding from local documents.

After choosing the style goal, you prepare a synthetic instruction-response dataset that repeatedly demonstrates the target voice. The dataset does not need to be huge for a demo. A few hundred examples can be enough to create a visible shift in style when the base model is already instruction tuned.

Training then produces a lightweight adapter directory. During inference, the application loads the base model, applies the adapter, retrieves supporting context from the local corpus, and generates an answer that reflects both the chosen persona and the retrieved evidence.

