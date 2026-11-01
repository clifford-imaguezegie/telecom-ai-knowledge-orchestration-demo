# Telecom AI Knowledge Orchestration — RAG & Evidence-Grounded Q&A

**Live, cloud-deployed reference implementation** of retrieval-augmented question answering, adaptive knowledge routing and evidence-based LLM evaluation over a pre-indexed telecom knowledge corpus.

The application compares three approaches to technical question answering:

1. **LLM-Only Reasoning** — uses pretrained knowledge without retrieval grounding.
2. **Strict Knowledge Grounding** — constrains answers to connected telecom evidence.
3. **Augmented Reasoning** — combines retrieved telecom evidence, controlled external information and model knowledge with explicit provenance.

The project investigates a practical engineering question:

> **What happens when adding retrieval-augmented generation makes a strong LLM worse?**

Rather than assuming RAG always improves an answer, the demonstration evaluates how retrieval quality, corpus coverage, grounding rules and broader knowledge orchestration affect accuracy, completeness and usefulness.

**Stack:** Python · Streamlit · Gemma · IBM Granite · BGE-M3 · FAISS · MCP · OpenRouter · Docker · Google Cloud Run

## Live Demo

**Google Cloud Run:** https://telecom-ai-orchestration-693254716479.europe-west2.run.app

> **Cold-start notice:** The public service scales to zero when idle to control infrastructure cost. A cold start requires the RAG runtime, FAISS index, metadata corpus and BGE-M3 embedding model to initialize. Subsequent queries are faster once the instance is warm.

### Explore the demonstration

1. Open the live application and allow the RAG runtime to initialize.
2. Select one of the reference telecom questions or enter a technical question within the connected knowledge domain.
3. Compare **LLM-Only**, **Strict Knowledge Grounding** and **Augmented Reasoning**.
4. Review available source/provenance labels, execution paths and comparative evaluation.

**Scope:** This public application uses a connected, pre-indexed telecom corpus. It is not a generic upload-and-query interface for private business PDFs.

## Relevance to Document Q&A and Enterprise Knowledge Assistants

The engineering patterns demonstrated here are applicable to internal knowledge assistants built around company policies, manuals, procedures and technical documentation. Client-specific implementations require additional ingestion, document/page citation validation and security controls.

| Capability | Demonstrated here | Additional work for a client document Q&A system |
|---|---|---|
| Semantic retrieval | BGE-M3 embeddings and FAISS telecom vector corpus | Process and index the client's approved documents |
| Evidence-grounded answers | Strict grounding and augmented reasoning | Enforce the agreed evidence boundary and answer policy |
| Source traceability | Source/provenance-aware answers and retrieval metadata | Reliable filename, page, chunk and document-version citations |
| Evaluation | Three validated reference scenarios and comparative Granite evaluation | Representative retrieval, citation, faithfulness and abstention tests |
| Web application | Streamlit on Google Cloud Run | Employee-facing interface and appropriate authentication |
| Document ingestion | Prebuilt index is loaded at runtime | PDF/manual/policy parsing, cleaning, chunking and update/reindex pipeline |
| Unanswerable questions | Grounding modes surface knowledge-coverage limitations | Explicit tested refusal/abstention for unsupported questions |
| Security | Deterministic input/security processing in the demonstration | Private storage, permissions, audit, privacy and retention requirements |

### Typical adaptation workflow

```text
Company PDFs / Policies / Manuals
              ↓
Text Extraction + Layout / OCR Handling (when needed)
              ↓
Cleaning + Chunking + Document / Page Metadata
              ↓
Embedding Generation + Vector Index
              ↓
Employee Question → Retrieval → Optional Reranking
              ↓
Evidence-Grounded LLM Answer
              ↓
Verifiable Source Citations or Explicit Abstention
              ↓
Evaluation + Monitoring + Document Updates
```

**Important:** This workflow describes how the architecture *could be adapted*. The current public demo does not claim to include arbitrary PDF uploads, validated page-level citations, employee identity management or permissions for confidential company documents.

## What the Demo Compares

### LLM-Only Reasoning

The base Gemma model answers from pretrained knowledge. This is a reference baseline for checking whether retrieval actually improves the answer.

### Strict Knowledge Grounding

The answer is constrained by connected telecom evidence. This prioritizes grounding but can reduce completeness when the source corpus lacks relevant information.

### Augmented Reasoning

Combines semantic RAG retrieval, MCP-based telecom knowledge retrieval, controlled external evidence when appropriate and pretrained model knowledge, with provenance labels to distinguish evidence sources.

## Reference Questions

The public interface includes three validated telecom scenarios:

- **About Xn Interface**
- **What is the role of the AMF in a 5G Standalone Network?**
- **Function of the SMO in O-RAN Architecture**

The scenarios demonstrate different retrieval, grounding and reasoning behaviours.

## Architecture

```text
User Question
      ↓
Deterministic Security Layer
      ↓
Gemma Baseline + Granite Knowledge-Scope Router
      ↓
 ┌───────────────────────────────────────────────┐
 │                                               │
 ▼                                               ▼
Connected Telecom Knowledge              External / General Scope
      ↓                                          ↓
Gemma Retrieval Planner                 ├─ Local runtime tools
      ↓                                 ├─ Open-WebSearch
 ├─ RAG_ONLY                            └─ Gemma knowledge fallback
 ├─ MCP_ONLY
 └─ HYBRID
      ↓
Adaptive Evidence Loop
      ↓
Grounded Gemma Answer
      ↓
Granite Comparative Evaluation
```

## Technology Stack

### Generation and Evaluation

- **Generator:** Gemma via OpenRouter
- **Knowledge routing / evaluation:** IBM Granite via OpenRouter

### Retrieval

- **Embedding model:** `BAAI/bge-m3`
- **Vector search:** FAISS
- **Vector corpus:** Approximately 1.5 million telecom vectors
- **Embedding dimension:** 1,024
- **Metadata:** 151 validated shards
- **Retrieval Top-K:** 5

### Knowledge Orchestration

- Semantic RAG retrieval
- Telecom-focused MCP retrieval and controlled fallback
- Open-WebSearch for changing public information
- Provenance-aware answer generation

### Application / Deployment

- Python, Streamlit, Docker
- Google Cloud Run, Google Secret Manager, Google Artifact Registry, Cloud Build

## Key Engineering Findings

### RAG is not automatically better than a strong LLM

When the retrieved corpus is incomplete, a strong base model can produce a more complete answer than a strictly grounded retrieval system.

### Retrieval relevance and answer quality are different

Relevant evidence may still be insufficient for a complete, useful answer. Retrieval and generation should therefore be evaluated separately.

### Strict grounding reduces risk but may limit completeness

Confining claims to the available evidence helps manage unsupported statements, but coverage gaps remain important.

### Augmented reasoning can recover completeness

Combining retrieved and controlled external evidence with model reasoning can improve coverage, provided provenance and unsupported claims remain visible.

### Deployment engineering matters

The deployment exposed practical concerns involving FAISS memory, embedding-model memory, startup, asset caching, Streamlit concurrency, memory limits, scaling and infrastructure cost.

## Validated Cloud Deployment

```text
Platform:              Google Cloud Run
Region:                europe-west2
Execution environment: Gen2
CPU:                   6 vCPU
Memory:                24 GiB
Concurrency:           20
Minimum instances:     0
Maximum instances:     1
Request timeout:       900 seconds
Container image:       module4-rc2
```

Minimum instances is set to zero to control cost. During validation, 4 vCPU / 16 GiB exceeded the available memory during RAG initialization; 6 vCPU / 24 GiB completed the tested startup successfully. Once warm, representative RAG and Augmented queries completed in approximately **12–13 seconds** in those validation runs. These are observed results, not latency guarantees.

## Demo-Day Operation

For a planned demonstration, provision a warm instance ahead of the session:

```text
Set minimum instances = 1
        ↓
Open the application
        ↓
Allow RAG initialization to complete
        ↓
Run a reference question
        ↓
Share the live URL
```

Afterward, return minimum instances to `0` if appropriate to limit infrastructure cost.

## Required Secrets

```text
OPENROUTER_API_KEY
KAGGLE_API_TOKEN
HF_TOKEN
```

Supply secrets via the deployment environment or secret manager. **Never commit credentials to the repository.**

## Local Development

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

Initial RAG startup can take longer because the embedding model, FAISS index and metadata assets must load. The local environment must have access to the required assets and credentials.

## Repository Purpose and Portfolio Relevance

This repository is a domain-specific reference implementation for **RAG, evidence-grounded technical Q&A, knowledge orchestration and comparative evaluation**. It provides reusable engineering concepts for knowledge-assistant projects while remaining transparent about the additional work needed to handle client documents, citations, user access and ongoing updates.

The central design tradeoffs are grounding, hallucination risk, relevance, completeness, provenance, latency and operational cost.

## Status

**Validated public telecom demonstration.** The application has been validated locally and on Google Cloud Run with the three reference telecom scenarios. The deployment is intentionally preserved as a stable public baseline rather than being described as a turnkey employee-document Q&A product.
