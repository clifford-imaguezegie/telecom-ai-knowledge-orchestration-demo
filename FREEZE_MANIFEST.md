\# Authoritative Deployment Baseline



\*\*Project:\*\* Telecom AI Knowledge Orchestration Demo  

\*\*Baseline date:\*\* 13 September 2026  

\*\*Status:\*\* Validated public deployment baseline



\## Purpose



This document records the authoritative deployed state of the Telecom AI Knowledge Orchestration Demo.



The repository demonstrates three telecom answer paths:



1\. Gemma LLM-Only

2\. Strict Knowledge Grounding

3\. Augmented Reasoning



The implementation is preserved as a standalone public demonstration baseline.



\## Validated Runtime Behaviour



The deployed application includes:



\- Gemma LLM-only baseline

\- Granite knowledge-scope routing and comparative evaluation

\- BGE-M3 semantic retrieval

\- FAISS telecom vector search

\- controlled MCP telecom retrieval

\- generic MCP fallback for demo continuity

\- Open-WebSearch integration

\- provenance-aware answer generation

\- clickable telecom reference questions

\- blank default question input

\- prewarmed RAG runtime initialization

\- serialized heavy-resource initialization to prevent duplicate concurrent loads



\## Reference Questions



Validated public reference scenarios:



1\. About Xn Interface

2\. What is the role of the AMF in a 5G Standalone Network?

3\. Function of the SMO in O-RAN Architecture



\## RAG Runtime



\- Embedding model: `BAAI/bge-m3`

\- Vector store: FAISS

\- Vector count: 1,506,367

\- Vector dimension: 1024

\- Metadata shards: 151

\- Retrieval Top-K: 5



RAG assets are resolved through the validated deployment runtime and loaded locally into the running container before telecom queries are exposed to users.



\## Cloud Deployment



\*\*Platform:\*\* Google Cloud Run  

\*\*Region:\*\* `europe-west2`  

\*\*Execution environment:\*\* Gen2  

\*\*Container image:\*\* `module4-rc2`



Validated resource configuration:



\- CPU: 6 vCPU

\- Memory: 24 GiB

\- Concurrency: 20

\- Minimum instances: 0

\- Maximum instances: 1

\- Request timeout: 900 seconds



The service intentionally scales to zero when inactive.



A true cold start requires the RAG assets, FAISS index, metadata corpus and BGE-M3 embedding model to initialize before the interface becomes fully available.



Warm RAG and Augmented requests were observed completing in approximately 12–13 seconds during deployment validation.



\## Deployment Lessons



The original 4 vCPU / 16 GiB deployment exceeded the Cloud Run memory limit during full RAG initialization.



The validated deployment therefore uses 6 vCPU / 24 GiB.



This demonstrated that production RAG deployment requirements include not only model inference, but also:



\- embedding-model memory

\- FAISS index memory

\- corpus asset handling

\- application runtime overhead

\- concurrency behaviour

\- cold-start management

\- infrastructure cost



\## Security



Required runtime secrets:



\- `OPENROUTER\_API\_KEY`

\- `KAGGLE\_API\_TOKEN`

\- `HF\_TOKEN`



Secrets are provided through Google Secret Manager and must never be committed to the repository.



The repository contains placeholder values only.



\## Known Limitations



\- Cold starts can take several minutes because the service scales to zero.

\- MCP fallback can occasionally return weaker evidence when no authoritative source target is identified.

\- Streamlit may emit non-fatal source-watcher errors related to optional `torchvision` imports inside the Transformers package. These do not affect the validated text-only telecom workflows.



\## Release State



The current implementation is the validated public baseline for the Telecom AI Knowledge Orchestration Demo.



Future experimentation should be performed without altering this baseline except for genuine bug fixes or security updates.

