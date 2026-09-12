---
title: Telecom AI Knowledge Orchestration Demo
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Telecom AI Knowledge Orchestration Demo

A Module 4 demonstration comparing three answer paths for telecom questions:

1. **Gemma LLM-Only** — pretrained model knowledge without retrieval grounding.
2. **Strict Knowledge Grounding** — answers constrained by connected telecom evidence.
3. **Augmented Reasoning** — combines RAG, MCP, live web evidence when available, and Gemma pretrained knowledge with explicit provenance.

## Demo focus

The experiment demonstrates that retrieval grounding can materially reduce unsupported claims, while strict grounding can also reduce relevance or completeness when retrieved evidence is incomplete. The Augmented path is designed to preserve evidence provenance while retaining broader reasoning capability.

## Reference questions

- About Xn Interface
- What is the role of the AMF in a 5G Standalone Network
- Function of the SMO in ORAN Architecture

## Runtime architecture

- Generator: Gemma via OpenRouter
- Evaluation/router: IBM Granite via OpenRouter
- RAG: BGE-M3 + FAISS telecom corpus
- MCP: controlled telecom knowledge retrieval with provenance
- Live external evidence: Open-WebSearch
- UI: Streamlit

## Required secrets

Configure these in the hosting platform's secret/environment settings. Do not commit real credentials.

```text
OPENROUTER_API_KEY
KAGGLE_API_TOKEN
HF_TOKEN
```

Optional environment variables are documented in `.env.example`.

## Local launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This repository is configured for a Hugging Face **Docker Space** on port `7860`.

The full RAG runtime is memory-intensive because it loads a large BGE-M3/FAISS index. Hugging Face Spaces is therefore the primary public deployment target for the full Module 4 demo.

## Module status

**Module 4 — Knowledge Orchestration: Release Candidate**

Frozen runtime behavior is documented in `FREEZE_MANIFEST.md`.
