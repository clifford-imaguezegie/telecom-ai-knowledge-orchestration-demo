# Module 4 Authoritative Runtime Freeze

**Project:** Telecom AI Knowledge Orchestration Demo  
**Freeze date:** 12 September 2026  
**Status:** Release candidate for deployment-readiness audit

## Authoritative source

This freeze is based on the user's latest uploaded local repository snapshot, with only the already-validated final hardening changes applied.

## Final runtime behavior included

- Three-path experiment: Gemma LLM-only, Strict Knowledge Grounding, Augmented Reasoning.
- MCP authoritative routing with generic fallback retained for demo continuity.
- Generic fallback uses Wikipedia-Telecom / Wikidata-Telecom when no stronger MCP target is selected.
- Augmented reasoning tolerates an individual evidence source returning zero evidence.
- Open-WebSearch timeout/failure is isolated and does not crash the Augmented path.
- Human-readable provenance/source classification in the UI.
- Reference/topic display built from available source metadata.
- Clickable reference questions populate the input and immediately run the experiment.
- Default question input is blank.
- Reference questions:
  1. About Xn Interface
  2. What is the role of the AMF in a 5G Standalone Network
  3. Function of the SMO in ORAN Architecture

## Security sanitization

The uploaded `.env.example` contained live-looking credentials. They are NOT included in this freeze. The file has been replaced with placeholders only.

**Required action before public deployment:** revoke/rotate the exposed OpenRouter, Kaggle and Hugging Face credentials from the uploaded snapshot, even if you believe they were never committed publicly.

## Validation completed

- All Python source files pass `py_compile`.
- Secret-pattern scan of the frozen bundle found no OpenRouter/Kaggle/Hugging Face token patterns.
- `.git`, `__pycache__`, and `.pyc` artifacts are excluded from the release bundle.

## Known limitation retained intentionally

MCP's lexical/generic fallback can return weakly relevant evidence. This is retained for demo continuity and should not be interpreted as the final production retrieval architecture.

## Deployment-readiness items still to audit

- `Dockerfile` in the uploaded repo is currently empty.
- Dependency/version pinning in `requirements.txt` should be reviewed for reproducible cloud deployment.
- RAG artifact paths/download strategy must be verified for Streamlit Community Cloud and Hugging Face Spaces.
- Open-WebSearch runtime/CLI availability must be verified in hosted environments.
- Secrets must be supplied through the hosting platform's secret manager, never committed.
