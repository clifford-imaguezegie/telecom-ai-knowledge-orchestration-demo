# Tebunor AI — Durable Independent Granite Evaluation Jobs (Implementation Candidate)

Status: source-only, feature OFF by default. Requires cloud provisioning, a verified tagged API revision and a separately merged website queue/poll adapter.

## Why the change

The original API completed an AMF three-path + Granite quality/risk evaluation at HTTP 200 in 106.11 seconds, leaving only ~9 seconds of margin against the Tebunor Pages Function's 115-second synchronous deadline. The existing synchronous API is proven to work but cannot guarantee a timeout-safe website experience under varying provider latency. The original comparison's 13.71s timer excludes the optional four Granite judge calls.

Cloud Run min-instances=0 is NOT safe for a fire-and-forget in-memory job; Cloud Run may scale down the instance after HTTP 202. Use durable Cloud Tasks invocation and Firestore state instead. No separate Cloud Run service is required in the proposed design.

## New API (flag-gated)

Only available when `TELECOM_RAG_ASYNC_ENABLED=true` and every queue/identity/project binding is present. Default Docker CMD still runs Streamlit and does not expose this Python HTTP server.

- `POST /api/v1/evaluation-jobs` with backend Bearer secret and JSON `{"question":"...","mode":"three_way"}` -> 202 `{job_id,capability,status:"queued"}`, after writing Firestore and queueing Cloud Task. It performs **no model inference** in the HTTP request.
- `GET /api/v1/evaluation-jobs/<32 lowercase hex>` with backend Bearer and `X-Job-Capability: <64 hex>` -> state `queued|running|completed|failed` and sanitized result on completion. Capabilities are stored only as SHA-256 hashes. An expired job returns 404 regardless of Firestore TTL deletion lag.
- `POST /api/v1/internal/evaluation-run` is only callable with a Google-signed OIDC service-account token whose **signature, audience, issuer, email and verification flag** are checked. The Cloud Tasks payload contains only `job_id`, never a user question or document text.
- The worker atomically claims a queued Firestore job, waits for the original BGE-M3/FAISS runtime to be ready (Cloud Tasks can retry 503), then calls unchanged `run_three_systems(question,True)`. It stores only the bounded original public result, removes the question and acknowledges completion. After a task has been claimed, failures are marked and **not retried** to avoid duplicate LLM billing. Unclaimed failures may retry using the queue policy.

This at-most-once-after-claim policy deliberately favours cost safety over transparent retry of an interrupted paid evaluation. A failed or evicted claimed job may remain `running` until expiry; a future hardening pass should set an explicit timed-out/expired status for visitor support.

## External setup (NOT performed by this code change)

1. Explicitly approve Google Cloud Tasks and Firestore billing/API enablement for project `project-7140365a-b5b0-4dbe-972` and choose the Firestore database/region. Verify whether a default Firestore database already exists before creating one.
2. Create one Cloud Tasks queue in `europe-west2` with conservative dispatch (e.g. one concurrent task), a task attempt deadline of 600 seconds and a carefully capped retry policy. The queue endpoint must be the verified **tagged** Cloud Run API revision, not the existing Streamlit service host.
3. Create a dedicated task-invoker service account. Grant the existing API runtime service account only the narrow Cloud Tasks enqueuer, Firestore document read/write and `iam.serviceAccountUser` privileges it needs. Cloud Tasks issues the OIDC ID token; the task endpoint validates the expected account even though the wider existing Cloud Run service URL is public.
4. Deploy a **new tagged API revision** using the updated image, preserving 100% original Streamlit traffic and 0% normal traffic to the tagged API revision. Set `--concurrency` to at least 4 so read-only status polls are not blocked by a long evaluation task; keep `--max-instances=1` and the model's semaphore to restrict concurrent paid inference.
5. Set only the API revision environment variables:
   - `TELECOM_RAG_ASYNC_ENABLED=true`
   - `GOOGLE_CLOUD_PROJECT=project-7140365a-b5b0-4dbe-972`
   - `TELECOM_RAG_TASKS_QUEUE=projects/project-7140365a-b5b0-4dbe-972/locations/europe-west2/queues/<approved-queue-name>`
   - `TELECOM_RAG_TASK_TARGET=https://rag-api-test---telecom-ai-orchestration-swxzeguboa-nw.a.run.app/api/v1/internal/evaluation-run`
   - `TELECOM_RAG_TASK_AUDIENCE=https://rag-api-test---telecom-ai-orchestration-swxzeguboa-nw.a.run.app`
   - `TELECOM_RAG_TASK_IDENTITY=<dedicated-task-invoker>@project-7140365a-b5b0-4dbe-972.iam.gserviceaccount.com`
6. Configure Firestore TTL for `tebunor_rag_evaluation_jobs.expires_at`, max 60-minute retention. Firestore deletes expired documents asynchronously; authorization refuses an expired job immediately.
7. Only enable the matching Cloudflare D1 signed-session and job/poll adapter after it is implemented and offline/live tested. Poll reads must not use the provider charging quota. Keep `TELECOM_RAG_EVALUATION_ENABLED=false` until cutover.

## Test matrix

- Missing backend auth, missing/invalid capability, expired job, invalid question and excessive size
- Spoofed/unsigned OIDC task requests denied before inference
- At-most-once claim under task duplicate delivery
- Cloud Run cold start: Cloud Tasks retries **before claim**, never charges the model during prewarm
- Firestore unavailable: fail closed with no fabricated scores
- Real >125-second Granite evaluation: client receives 202 and polls; no Cloudflare origin 524
- Instance interruption and provider failure: bounded job expiry/failed state, no duplicate-cost retry
- Turnstile signed-session, D1 quota once per created job, owner/capability isolation, cost limits and audit logging
- Normal Streamlit 100% traffic unchanged

Official: https://docs.cloud.google.com/run/docs/triggering/using-tasks ;
https://docs.cloud.google.com/tasks/docs/create-tasks ;
https://docs.cloud.google.com/run/docs/about-instance-autoscaling ;
https://firebase.google.com/docs/firestore/ttl

The source addition alone does **not** enable Cloud Tasks, Firestore, paid LLM calls, deployment or user-facing evaluation.
