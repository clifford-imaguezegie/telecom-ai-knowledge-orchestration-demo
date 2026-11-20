# Tebunor AI — Existing Cloud Run RAG API integration

**Status: code staged; not deployed; not live.**
**Scope: reuse the original Telecom Knowledge Orchestration engine without duplicating inference assets or OpenRouter credentials.**

## What this change does

- Adds \`api_server.py\`: a small authenticated JSON API that calls the existing \`runtime.comparison.run_three_path_experiment\` and, when selected, the original \`runtime.evaluation_threeway\` judge. It does not implement a second RAG algorithm or call a different model.
- Adds \`api_adapter.py\`: a bounded response schema for Tebunor AI. It strips raw indexed passages, private filesystem paths, corpus text and prompts.
- Adds \`tests/test_native_api.py\` and CI assertions covering bearer-token authentication, closed-by-default behaviour, readiness, input validation, 429 concurrency and original-answer projection.
- **Does not change \`app.py\`, \`runtime/\`, \`requirements.txt\`, the Dockerfile's existing Streamlit CMD, any secret, traffic or Google Cloud resource.** The existing deployment continues to serve Streamlit until an operator explicitly creates and shifts traffic to a new revision.

## Architecture after approved cutover

\`\`\`
Tebunor first-party /demos/telecom-rag/
           ↓
Cloudflare Pages Functions
  [Turnstile + session + quotas + backend bearer token]
           ↓ HTTPS /api/v1/answer
Existing Cloud Run service: telecom-ai-orchestration (API-serving revision)
           ↓
Original runtime/comparison.py  ←  Original prewarm / FAISS / BGE-M3 / MCP
           ↓
Original Gemma generator + Granite router and independent judge
           ↓
Bounded three-path JSON response → Cloudflare → Tebunor UI
\`\`\`

The original Streamlit-serving Cloud Run revision remains available for rollback. Same service ≠ simultaneous Streamlit and API; an API revision replaces the Streamlit-serving revision when traffic is switched. Avoid running two separate Python processes with separate FAISS/BGE-M3 caches in one 24 GiB container.

## Available API routes

- \`GET /healthz\`: process liveness, not RAG readiness.
- \`GET /api/v1/ready\`: 503 until original index/model initialization finishes and the API is explicitly enabled; 200 when available.
- \`POST /api/v1/answer\`: requires \`Authorization: Bearer <private-backend-token>\` and JSON: \`{"question":"Role of AMF?","mode":"three_way","include_evaluation":false}\`. Sends to original 3-path experiment; optional judge uses the original logic.

No visitor should see, store or send the backend bearer token from client-side JavaScript.

## Before any deployment — operator-only checks

1. Verify the Cloud Run **project**, the service's current revision and its already-configured OpenRouter/HF/Kaggle secret references (do not expose secret values).
2. Preserve the existing Streamlit revision name, traffic allocation, CPU/memory settings and the source commit. Confirm billing status and per-query model budget.
3. Review and merge this branch after CI passes. Verify compilation and offline tests:

   \`\`\`bash
   python -m compileall -q api_adapter.py api_server.py runtime app.py app
   python -m unittest discover -s tests -p 'test_native_api.py' -v
   \`\`\`

4. Prepare an independently scoped \`TELECOM_RAG_BACKEND_TOKEN\` secret (random 32+ characters), separate from \`OPENROUTER_API_KEY\`. The **OpenRouter credential already bound to the existing service is reused unchanged**. Cloudflare also needs the bearer token as a server-side secret.
5. Build a **no-traffic tagged revision** from this original source repository, overriding the normal command to \`python api_server.py\` **only on the tagged revision**. Retain the previous revision at its original traffic percentage. The account owner approves the build and inference spend.
6. **Cold-start blocker to verify:** \`api_server.py\` prewarms the original FAISS index/BGE-M3 on a background thread. With Cloud Run request-based CPU allocation, a background thread can be throttled after the server responds. Inspect revision CPU-allocation and startup behaviour; do not assume readiness from \`/healthz\` alone. Check readiness and cold starts before any traffic switch. Avoid silently enabling always-on CPU if it would increase charges.
7. Confirm tagged API revision can complete Xn, AMF and SMO reference tests, including full three-path JSON, optional independent Granite evaluation, normal answer latency and a 95-second Cloudflare gateway deadline. If original inference takes longer than 95 seconds, **do not enable the public gateway yet**; agree on an asynchronous job/poll solution or a validated latency adjustment.
8. Verify 401 for unauthorized traffic, 503 while warming, 429 for parallel experiments, no leaked raw passages, correct session/rate limiting and predictable Cloud Run costs.
9. After acceptance only: switch Cloud Run service traffic to the verified API revision, configure Cloudflare's \`TELECOM_RAG_BACKEND_URL\` to the original service hostname plus \`/api/v1/answer\`, configure \`TELECOM_RAG_BACKEND_TOKEN\` server-side and enable \`TELECOM_RAG_EXISTING_HOST_API_VERIFIED=true\` and \`TELECOM_RAG_DEMO_ENABLED=true\`.
10. Then change the Tebunor gallery CTA to the native \`/demos/telecom-rag/\` route, not the Cloud Run API URL.

## Rollback

Set \`TELECOM_RAG_DEMO_ENABLED=false\` in Cloudflare (fail closed), then return traffic to the original, saved Streamlit Cloud Run revision. The original Python files, Dockerfile Streamlit command and service history are untouched by this PR.

See the matching website release runbook at:
https://github.com/tebunor-ai/tebunor-website/blob/main/docs/V2.1_TELECOM_RAG_EXISTING_SERVICE_CUTOVER.md

**No live deployment is performed by GitHub CI or by this change.**
