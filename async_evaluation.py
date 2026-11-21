"""Durable, opt-in Granite evaluation jobs for the original Telecom RAG API.

Never spawn a Python thread after returning HTTP 202. Cloud Tasks performs the
long request, while Firestore retains a bounded status/result across scale-to-zero.
All Google clients are imported lazily; the default Streamlit app remains inert.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone


COLLECTION = "tebunor_rag_evaluation_jobs"
JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
CAP_RE = re.compile(r"^[a-f0-9]{64}$")
MAX_RETENTION = timedelta(minutes=60)


class JobUnavailable(Exception):
    """The durable job backend has not been provisioned or is disabled."""


class JobNotFound(Exception):
    """The opaque job ID/capability pair is invalid or expired."""


def enabled():
    return (
        os.getenv("TELECOM_RAG_ASYNC_ENABLED") == "true"
        and bool(os.getenv("TELECOM_RAG_TASKS_QUEUE"))
        and bool(os.getenv("TELECOM_RAG_TASK_TARGET"))
        and bool(os.getenv("TELECOM_RAG_TASK_AUDIENCE"))
        and bool(os.getenv("TELECOM_RAG_TASK_IDENTITY"))
        and bool(os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT"))
    )


def _clients():
    if not enabled():
        raise JobUnavailable("Durable evaluation backend is disabled.")
    from google.cloud import firestore, tasks_v2
    db = firestore.Client()
    tasks = tasks_v2.CloudTasksClient()
    return db, tasks


def _ref(db, job_id):
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise JobNotFound()
    return db.collection(COLLECTION).document(job_id)


def _expired(data):
    when = data.get("expires_at")
    return not isinstance(when, datetime) or when <= datetime.now(timezone.utc)


def start_job(question):
    """Persist a job and enqueue a task; no LLM call occurs on this request."""
    db, tasks = _clients()
    from google.cloud import firestore
    from google.protobuf import duration_pb2

    job_id = secrets.token_hex(16)
    capability = secrets.token_hex(32)
    now = datetime.now(timezone.utc)
    ref = _ref(db, job_id)
    ref.create({
        "status": "queued",
        "question": question,
        "capability_hash": hashlib.sha256(capability.encode()).hexdigest(),
        "created_at": now,
        "expires_at": now + MAX_RETENTION,
    })

    queue = os.environ["TELECOM_RAG_TASKS_QUEUE"]
    task = {
        "name": queue + "/tasks/" + job_id,
        "http_request": {
            "http_method": "POST",
            "url": os.environ["TELECOM_RAG_TASK_TARGET"],
            "headers": {"Content-Type": "application/json"},
            "body": ('{"job_id":"' + job_id + '"}').encode(),
            "oidc_token": {
                "service_account_email": os.environ["TELECOM_RAG_TASK_IDENTITY"],
                "audience": os.environ["TELECOM_RAG_TASK_AUDIENCE"],
            },
        },
        "dispatch_deadline": duration_pb2.Duration(seconds=600),
    }
    try:
        tasks.create_task(request={"parent": queue, "task": task})
    except Exception:
        ref.update({"status": "failed", "question": firestore.DELETE_FIELD})
        raise JobUnavailable("Could not queue the independent evaluation.")
    return {"ok": True, "job_id": job_id, "capability": capability, "status": "queued"}


def job_status(job_id, capability):
    if not isinstance(capability, str) or not CAP_RE.fullmatch(capability):
        raise JobNotFound()
    db, _ = _clients()
    snap = _ref(db, job_id).get()
    if not snap.exists:
        raise JobNotFound()
    data = snap.to_dict()
    actual = hashlib.sha256(capability.encode()).hexdigest()
    if not hmac.compare_digest(actual, data.get("capability_hash", "")) or _expired(data):
        raise JobNotFound()
    status = data.get("status")
    result = data.get("result") if status == "completed" else None
    return {
        "ok": True,
        "status": status if status in ("queued", "running", "completed", "failed") else "failed",
        "result": result if isinstance(result, dict) else None,
    }


def verify_task_oidc(header):
    """Verify Cloud Tasks' Google-signed bearer token before reading any job."""
    if not enabled() or not isinstance(header, str) or not header.startswith("Bearer "):
        return False
    from google.oauth2 import id_token
    from google.auth.transport import requests
    try:
        claims = id_token.verify_oauth2_token(
            header[7:], requests.Request(),
            audience=os.environ["TELECOM_RAG_TASK_AUDIENCE"],
        )
        return (
            claims.get("iss") in ("https://accounts.google.com", "accounts.google.com")
            and claims.get("email") == os.environ["TELECOM_RAG_TASK_IDENTITY"]
            and claims.get("email_verified") is True
        )
    except Exception:
        return False


def run_job(job_id, runner):
    """At-most-one active claim; never retry LLM spend after a job was claimed."""
    db, _ = _clients()
    from google.cloud import firestore
    ref = _ref(db, job_id)
    transaction = db.transaction()

    @firestore.transactional
    def claim(tx):
        snap = ref.get(transaction=tx)
        if not snap.exists:
            return None
        data = snap.to_dict()
        if data.get("status") != "queued" or _expired(data):
            return None
        question = data.get("question")
        if not isinstance(question, str) or len(question) > 500:
            return None
        tx.update(ref, {"status": "running", "started_at": datetime.now(timezone.utc)})
        return question

    question = claim(transaction)
    if question is None:
        return {"ok": True, "status": "already_claimed"}

    try:
        result = runner(question, True)
        # This is the already sanitized original API projection, never the
        # original private FAISS contexts, prompts or provider credentials.
        if not isinstance(result, dict) or result.get("schema_version") != "tebunor-rag-1":
            raise ValueError("Missing original evaluation result")
        ref.update({
            "status": "completed",
            "result": result,
            "question": firestore.DELETE_FIELD,
            "completed_at": datetime.now(timezone.utc),
        })
        return {"ok": True, "status": "completed"}
    except Exception:
        ref.update({
            "status": "failed",
            "question": firestore.DELETE_FIELD,
            "completed_at": datetime.now(timezone.utc),
        })
        # Do not raise to Cloud Tasks after a paid run: retries may double-bill.
        return {"ok": False, "status": "failed"}
