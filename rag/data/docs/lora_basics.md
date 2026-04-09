# LoRA Basics

Low-Rank Adaptation, usually shortened to LoRA, is a parameter-efficient fine-tuning method. Instead of updating every weight in a large language model, LoRA freezes the original model and learns a small set of low-rank matrices that are attached to specific layers. This keeps the number of trainable parameters small and makes experiments cheaper to run.

For a style-tuning demo, LoRA is a practical choice because the base model keeps its general language ability while the adapter nudges how it answers. That means the project can change tone, structure, and helpfulness without pretending to inject a large amount of new factual knowledge into the model weights.

LoRA is often applied to attention projections such as query, key, value, and output layers. When training finishes, the adapter can be loaded on top of the base model for inference. This separation is useful in demos because you can compare base-model answers to adapted answers without duplicating the full model checkpoint.

