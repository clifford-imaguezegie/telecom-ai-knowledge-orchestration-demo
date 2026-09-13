\# Telecom AI Knowledge Orchestration Demo



A production-deployed telecom AI demonstration that compares three approaches to answering technical telecom questions:



1\. \*\*LLM-Only Reasoning\*\* — answers using pretrained model knowledge without retrieval grounding.

2\. \*\*Strict Knowledge Grounding\*\* — answers constrained by connected telecom evidence.

3\. \*\*Augmented Reasoning\*\* — combines retrieved telecom evidence, controlled external information, and pretrained model knowledge with explicit provenance.



The project explores a practical engineering question:



> \*\*What happens when adding retrieval-augmented generation makes a strong LLM worse?\*\*



Rather than assuming that RAG automatically improves an LLM, this implementation evaluates when retrieval helps, when strict grounding limits answer quality, and when broader knowledge orchestration provides a better result.



\## Live Demo



\*\*Google Cloud Run\*\*



https://telecom-ai-orchestration-693254716479.europe-west2.run.app



> \*\*Cold-start notice:\*\* The public service scales to zero when idle to control infrastructure cost. A true cold start requires the RAG runtime, FAISS index, metadata corpus, and BGE-M3 embedding model to initialize before the full interface becomes available. Once the instance is warm, subsequent queries are substantially faster.



\## What the Demo Compares



\### LLM-Only



The base Gemma model answers directly from pretrained knowledge.



This provides a useful benchmark for determining whether retrieval actually improves the final response.



\### Strict Knowledge Grounding



The answer is constrained by evidence retrieved from connected telecom knowledge sources.



This path prioritizes grounding and reduced hallucination risk, but can lose completeness when the available corpus does not contain sufficient evidence.



\### Augmented Reasoning



The system combines:



\* semantic RAG retrieval

\* MCP-based telecom knowledge retrieval

\* live external evidence when appropriate

\* pretrained Gemma knowledge

\* explicit provenance



This allows the system to recover useful technical completeness without hiding where the information came from.



\## Reference Questions



The public interface includes three validated telecom scenarios:



\* \*\*About Xn Interface\*\*

\* \*\*What is the role of the AMF in a 5G Standalone Network?\*\*

\* \*\*Function of the SMO in O-RAN Architecture\*\*



These questions demonstrate different retrieval and reasoning characteristics across the three answer paths.



\## Architecture



```text

User Question

&#x20;     ↓

Deterministic Security Layer

&#x20;     ↓

Gemma Baseline + Granite Knowledge-Scope Router

&#x20;     ↓

&#x20;┌───────────────────────────────────────────────┐

&#x20;│                                               │

&#x20;▼                                               ▼

Connected Telecom Knowledge              External / General Scope

&#x20;     ↓                                          ↓

Gemma Retrieval Planner                 ├─ Local runtime tools

&#x20;     ↓                                 ├─ Open-WebSearch

&#x20;├─ RAG\_ONLY                            └─ Gemma knowledge fallback

&#x20;├─ MCP\_ONLY

&#x20;└─ HYBRID

&#x20;     ↓

Adaptive Evidence Loop

&#x20;     ↓

Grounded Gemma Answer

&#x20;     ↓

Granite Comparative Evaluation

```



\## Technology Stack



\### Generation and Evaluation



\* \*\*Generator:\*\* Gemma via OpenRouter

\* \*\*Knowledge routing / evaluation:\*\* IBM Granite via OpenRouter



\### Retrieval



\* \*\*Embedding model:\*\* `BAAI/bge-m3`

\* \*\*Vector search:\*\* FAISS

\* \*\*Vector corpus:\*\* approximately 1.5 million telecom vectors

\* \*\*Embedding dimension:\*\* 1,024

\* \*\*Metadata:\*\* 151 validated shards

\* \*\*Retrieval Top-K:\*\* 5



\### Knowledge Orchestration



\* Semantic RAG retrieval

\* Telecom-focused MCP retrieval

\* Controlled MCP fallback

\* Open-WebSearch for changing public information

\* Provenance-aware answer generation



\### Application



\* Python

\* Streamlit

\* Docker

\* Google Cloud Run

\* Google Secret Manager

\* Google Artifact Registry

\* Cloud Build



\## Key Engineering Findings



\### RAG is not automatically better than a strong LLM



A strong base model can sometimes produce a more complete answer than a strictly grounded RAG system when the retrieval corpus is incomplete.



\### Retrieval quality and answer quality are different



A retrieval system can return technically relevant evidence while still failing to provide enough information for a high-quality final answer.



\### Strict grounding reduces risk but can reduce completeness



Grounding helps reduce unsupported claims, but a model constrained to incomplete evidence can produce an answer that is less useful than the LLM-only baseline.



\### Augmented reasoning can recover completeness



Combining retrieved evidence, controlled external evidence, and pretrained model knowledge can restore missing information while retaining provenance.



\### Deployment architecture matters



A working RAG notebook is not automatically a deployable RAG application.



The production deployment exposed practical issues around:



\* FAISS memory consumption

\* embedding-model memory

\* cold-start initialization

\* asset caching

\* Streamlit concurrency

\* container memory limits

\* autoscaling

\* deployment cost



These became part of the engineering evaluation rather than being treated as separate infrastructure concerns.



\## Validated Cloud Deployment



The live deployment currently uses:



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



The service intentionally uses \*\*minimum instances = 0\*\* so it can scale to zero when inactive.



During validation, the original 4 vCPU / 16 GiB configuration exceeded its memory limit while initializing the full RAG runtime. The deployment was subsequently validated successfully at 6 vCPU / 24 GiB.



Once warm, representative RAG and Augmented queries completed in approximately \*\*12–13 seconds\*\* during live testing.



\## Demo-Day Operation



For an interview, technical demonstration, or customer discussion:



```text

Set minimum instances = 1

&#x20;       ↓

Open the application

&#x20;       ↓

Allow RAG initialization to complete

&#x20;       ↓

Run one reference question

&#x20;       ↓

Share the live URL

```



After the demonstration, minimum instances can be returned to `0` so the service scales to zero again.



\## Required Secrets



The deployment requires:



```text

OPENROUTER\_API\_KEY

KAGGLE\_API\_TOKEN

HF\_TOKEN

```



Secrets must be supplied through the deployment environment or secret manager.



\*\*Never commit credentials to the repository.\*\*



\## Local Development



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run app.py

```



The first RAG initialization can take significantly longer than subsequent queries because the embedding model, FAISS index, and metadata assets must be loaded.



\## Repository Purpose



This repository is a standalone implementation of a \*\*Telecom AI Knowledge Orchestration architecture\*\*.



Its purpose is not simply to demonstrate RAG, but to explore how multiple knowledge and reasoning mechanisms can be orchestrated to produce telecom answers with a more deliberate balance between:



\* grounding

\* hallucination risk

\* relevance

\* completeness

\* provenance

\* latency

\* operational cost



\## Status



\*\*Validated Public Deployment\*\*



The application has been validated locally and on Google Cloud Run using the three reference telecom scenarios.



The current deployment is intentionally preserved as the stable public demonstration



