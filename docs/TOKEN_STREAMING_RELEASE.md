# Native Telecom RAG: real Gemma token streaming acceptance

**Scope:** A new **0%-traffic tagged** test revision of the *existing* Cloud Run service. The current original Streamlit production revision `telecom-ai-orchestration-00006-ww4` must keep 100% normal traffic. The already-validated response-complete SSE tagged revision `telecom-ai-orchestration-sse1121` must remain reachable as `rag-api-test` until real-token acceptance passes.

## What changed

- Original Gemma final-answer calls use OpenRouter Chat Completions `stream=True` only when the native SSE experiment explicitly subscribes to output tokens. No extra requests and no model/prompts/provider changes.
- Each verified model `delta.content` is forwarded as SSE `answer_delta` with `{key,text}` for `llm_only`, `strict`, `augmented`; original `path_complete` remains the canonical answer/evidence correction.
- The original routing/planner, RAG/MCP/web retrieval, evidence validation and Granite judge never stream raw prompts or private retrieved material.
- Native normal JSON mode and Streamlit remain on their original nonstreaming call path.
- Strict non-telecom general-knowledge fallback does not emit partial deltas because a required disclaimer is added after generation.
- New model bytes must be rendered as plain text, then reconciled with final markdown on `path_complete`; never treat SSE bytes as HTML.

## Pre-build validation

```bash
git switch main
git pull --ff-only
git log -1 --oneline
python3 -m unittest discover -s tests -p 'test_native_api.py' -v
python3 -m unittest discover -s tests -p 'test_token_stream.py' -v
```

Use a **new image tag** with the merged commit short SHA. A Cloud Build and Artifact Registry push cost money, but no hosted model calls are made during build:

```bash
IMAGE="europe-west2-docker.pkg.dev/project-7140365a-b5b0-4dbe-972/telecom-ai-demo/telecom-ai-orchestration:token-<MERGED_SHA>-20261121"
gcloud builds submit . --project=project-7140365a-b5b0-4dbe-972 --region=europe-west2 --tag="$IMAGE"
```

Do not literally include angle brackets in an image tag.

## Deployment: separate token-test tag, no default traffic

Deploy to `rag-token-test` **not** `rag-api-test`, preserving the currently tested SSE revision and 100% Streamlit allocation. Reuse original secrets from the existing Cloud Run service template; never override/clear secret references.

```bash
gcloud run deploy telecom-ai-orchestration \
  --project=project-7140365a-b5b0-4dbe-972 \
  --region=europe-west2 \
  --image="$IMAGE" \
  --command=python --args=api_server.py --port=7860 \
  --cpu=6 --memory=24Gi --concurrency=1 --timeout=900 \
  --min-instances=0 --max-instances=1 \
  --service-account=telecom-ai-runner@project-7140365a-b5b0-4dbe-972.iam.gserviceaccount.com \
  --update-env-vars=TELECOM_RAG_API_ENABLED=true \
  --no-traffic --tag=rag-token-test --revision-suffix=tok1121

gcloud run services describe telecom-ai-orchestration \
  --project=project-7140365a-b5b0-4dbe-972 \
  --region=europe-west2 \
  --format='json(status.traffic)'
```

**Expected:** Streamlit 100%; `rag-api-test` still points to `sse1121` at 0%; `rag-token-test` points to the new token revision at 0%.

## Acceptance gates before production enablement

1. Readiness: tagged `rag-token-test` URL `/api/v1/ready` returns HTTP 200 and `ready:true`. Cold start may take over a minute.
2. Authentication: unauthenticated `POST /api/v1/stream` returns HTTP 401; no inference.
3. **One billable, authenticated experiment** through `rag-token-test`: requires `accepted`, real non-empty `answer_delta` from **all three keys** before each corresponding `path_complete`, followed by `evaluation_started`, `evaluation_complete`, `complete`. Assert final three answers and Granite scores, no raw private context, full streamed-text reconciliation and observed first-token latency. Do not log model text, question, token or secrets.
4. With feature flag still off, confirm original `rag-api-test` and Streamlit URL continue to work. Do not redirect normal service traffic.
5. Only after passing: move the **`rag-api-test` tag** to the verified token revision using `gcloud run services update-traffic telecom-ai-orchestration --region=europe-west2 --update-tags rag-api-test=telecom-ai-orchestration-tok1121`. This alters a test tag, **not** 100% Streamlit traffic. Confirm the full traffic JSON immediately.
6. In Cloudflare Pages production settings, `TELECOM_RAG_STREAMING_ENABLED=true` activates the first-party SSE gateway once the verified tagged backend is in place. Test the actual browser (user bubble, token-by-token three panels, Granite default and latency-first metrics, mobile). Retain 15/session, 80/IP/day, 250/site/day limits.
7. **Rollback** if token generation fails: change tag back to `rag-api-test=telecom-ai-orchestration-sse1121`. Setting `TELECOM_RAG_STREAMING_ENABLED=false` also restores the old synchronous gateway. Recheck Streamlit traffic remains 100%.

Do not remove the old tagged revision, alter the three model IDs, expose service credentials in logs or run repeated paid tests.
