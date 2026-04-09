# Evaluation Notes

Even a tiny local assistant benefits from a minimal evaluation mindset. The first question is whether retrieval is pulling the right chunks for common queries. The second is whether the LoRA adapter changes tone consistently without damaging clarity.

For a one-day portfolio project, lightweight evaluation can be manual. Ask a few repeated questions with and without the adapter. Check whether the adapted model sounds more like the intended persona. Then inspect the retrieved snippets and confirm that the answer references the local markdown files when appropriate.

This kind of small evaluation is not a benchmark, but it is honest and useful. It helps demonstrate that the project includes distinct responsibilities: style adaptation, retrieval, and an end-to-end application layer.
