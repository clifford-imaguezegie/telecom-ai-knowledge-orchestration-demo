"""Optional authenticated native API entry point for the ORIGINAL Telecom Knowledge Orchestration.
It imports original runtime modules; no alternative LLM or synthetic RAG is used.
This file is inert in the default Dockerfile's Streamlit CMD. A tagged, no-traffic
Cloud Run revision can override the container command to `python api_server.py`.
No live deployment, billing change, traffic switch or frontend enablement here.
"""
from __future__ import annotations
import asyncio
import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

from api_adapter import public_result,validate_question

PORT=int(os.environ.get("PORT","7860"))
TOKEN=os.environ.get("TELECOM_RAG_BACKEND_TOKEN","")
ENABLED=os.environ.get("TELECOM_RAG_API_ENABLED","false")=="true"
ready=False
single=threading.BoundedSemaphore(1)


def is_ready():
    return bool(ENABLED and len(TOKEN)>=32 and ready)


def prewarm():
    global ready
    if not ENABLED or len(TOKEN)<32:
        logging.warning("Telecom RAG API disabled; missing explicit opt-in or secret")
        return
    try:
        from runtime.prewarm import prewarm_rag_runtime
        prewarm_rag_runtime()  # original BGE-M3 + FAISS + metadata initialisation
        ready=True
        logging.info("Original Telecom RAG engine is ready")
    except Exception:
        logging.exception("RAG prewarming failed; API remains unavailable")


def run_three_systems(question:str,include_evaluation:bool):
    # The original comparison timing does NOT include the optional Granite judges.
    # Measure the entire API experiment separately without altering the engine.
    started=time.perf_counter()
    from runtime.comparison import run_three_path_experiment
    r=asyncio.run(run_three_path_experiment(question))
    three_path_s=round(time.perf_counter()-started,3)
    evaluation=None
    evaluation_s=None
    if include_evaluation:
        evaluation_started=time.perf_counter()
        from runtime.evaluation_threeway import (
            evaluate_three_way_telecom,run_three_way_quality_judge)
        paths=r["paths"]
        result=lambda key:paths[key]["result"] or {}
        answer=lambda key:str(result(key).get("answer") or "")
        q=r.get("sanitized_question") or question
        if r.get("query_type")=="TELECOM_GROUNDED":
            strict=result("strict");aug=result("augmented")
            evaluation=evaluate_three_way_telecom(
                question=q,
                llm_only_answer=answer("llm_only"),
                strict_answer=answer("strict"),
                augmented_answer=answer("augmented"),
                strict_evidence_context=strict.get("final_context") or "",
                strict_evidence_items=strict.get("final_evidence") or [],
                augmented_evidence_context=aug.get("final_context") or "",
                augmented_evidence_items=aug.get("final_evidence") or [],
            )
        else:
            # No corpus hallucination-risk assessment outside telecom knowledge scope.
            v=run_three_way_quality_judge(
                q,answer("llm_only"),answer("strict"),answer("augmented"))["payload"]
            evaluation={"applicable":False,
                "systems":{key:{
                    "relevance_score":v[answer_key].get("relevance_score"),
                    "completeness_score":v[answer_key].get("completeness_score"),
                    "hallucination_risk_pct":None,
                    "hallucination_risk_label":"N/A"
                } for key,answer_key in (
                    ("llm_only","answer_a"),("strict","answer_b"),("augmented","answer_c"))},
                "comparison_summary":v.get("comparison_summary"),
                "question_intent":v.get("question_intent")}
    if include_evaluation:
        evaluation_s=round(time.perf_counter()-evaluation_started,3)
    output=public_result(r,evaluation)
    output["timing"].update({
        "api_total_wall_s":round(time.perf_counter()-started,3),
        "three_path_api_wall_s":three_path_s,
        "evaluation_wall_s":evaluation_s,
    })
    return output


class Handler(BaseHTTPRequestHandler):
    server_version="TebunorTelecomRAG/1.0"
    sys_version=""
    def log_message(self,fmt,*args):
        # Never log question text, tokens, documents or PII.
        logging.info("Backend HTTP route %s",self.path.split("?")[0][:90])
    def reply(self,status,payload,headers=None):
        body=json.dumps(payload,ensure_ascii=False,separators=(",",":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Content-Length",str(len(body)))
        for k,v in (headers or {}).items():self.send_header(k,v)
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if self.path.startswith("/api/v1/evaluation-jobs/"):
            if len(TOKEN)<32 or not hmac.compare_digest(
                self.headers.get("Authorization",""),"Bearer "+TOKEN):
                return self.reply(401,{"ok":False,"error":"Unauthorised"})
            from async_evaluation import JobNotFound,JobUnavailable,job_status
            job_id=self.path[len("/api/v1/evaluation-jobs/"):]
            try:
                data=job_status(job_id,self.headers.get("X-Job-Capability",""))
                return self.reply(200,data)
            except JobNotFound:
                return self.reply(404,{"ok":False,"error":"Evaluation job not found"})
            except JobUnavailable:
                return self.reply(503,{"ok":False,"error":"Asynchronous evaluation is not configured"})
            except Exception:
                logging.exception("Evaluation job status unavailable")
                return self.reply(503,{"ok":False,"error":"Evaluation job status unavailable"})
        if self.path=="/healthz":
            return self.reply(200,{"ok":True,"service":"telecom-rag-api"})
        if self.path=="/api/v1/ready":
            return self.reply(200 if is_ready() else 503,
                              {"ok":is_ready(),"ready":is_ready()},
                              {"Retry-After":"20"} if not is_ready() else None)
        return self.reply(404,{"ok":False,"error":"Not found"})
    def do_POST(self):
        if self.path=="/api/v1/internal/evaluation-run":
            from async_evaluation import enabled,verify_task_oidc,run_job
            if not enabled():
                return self.reply(503,{"ok":False,"error":"Evaluation jobs disabled"})
            if not verify_task_oidc(self.headers.get("Authorization","")):
                return self.reply(401,{"ok":False,"error":"Cloud Tasks identity rejected"})
            if not is_ready():
                return self.reply(503,{"ok":False,"error":"Original RAG engine warming"},
                                  {"Retry-After":"20"})
            try:
                length=int(self.headers.get("Content-Length") or "0")
                if length<=0 or length>200:
                    raise ValueError("invalid task body")
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict) or set(body)!={"job_id"}:
                    raise ValueError("invalid job")
                job_id=body["job_id"]
            except (ValueError,TypeError,UnicodeDecodeError):
                return self.reply(400,{"ok":False,"error":"Invalid job"})
            if not single.acquire(blocking=False):
                return self.reply(429,{"ok":False,"error":"Original experiment busy"},
                                  {"Retry-After":"15"})
            try:
                return self.reply(200,run_job(job_id,run_three_systems))
            except Exception:
                logging.exception("Evaluation job runner unavailable")
                return self.reply(503,{"ok":False,"error":"Evaluation job runner unavailable"})
            finally:
                single.release()
        if self.path not in ("/api/v1/answer","/api/v1/evaluation-jobs"):
            return self.reply(404,{"ok":False,"error":"Not found"})
        if len(TOKEN)<32 or not hmac.compare_digest(
            self.headers.get("Authorization",""),"Bearer "+TOKEN):
            return self.reply(401,{"ok":False,"error":"Unauthorised"})
        if self.path=="/api/v1/evaluation-jobs":
            from async_evaluation import enabled,start_job,JobUnavailable
            if not enabled():
                return self.reply(503,{"ok":False,"error":"Durable evaluation jobs disabled"})
            try:
                length=int(self.headers.get("Content-Length") or "0")
                if length<=0 or length>1800:
                    return self.reply(413,{"ok":False,"error":"Request too large"})
                if self.headers.get("Content-Type","").split(";")[0].lower().strip()!="application/json":
                    return self.reply(415,{"ok":False,"error":"JSON required"})
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict) or set(body)!={"question","mode"} or body["mode"]!="three_way":
                    raise ValueError("Invalid evaluation input")
                question=validate_question(body["question"])
                return self.reply(202,start_job(question))
            except (ValueError,TypeError,UnicodeDecodeError,KeyError):
                return self.reply(400,{"ok":False,"error":"Invalid evaluation question"})
            except JobUnavailable:
                return self.reply(503,{"ok":False,"error":"Durable evaluation is unavailable"})
            except Exception:
                logging.exception("Could not submit evaluation job")
                return self.reply(503,{"ok":False,"error":"Evaluation job could not be queued"})
        if not is_ready():
            return self.reply(503,{"ok":False,"error":"Original RAG engine is warming or unavailable"},
                              {"Retry-After":"20"})
        if self.headers.get("Content-Type","").split(";")[0].lower().strip()!="application/json":
            return self.reply(415,{"ok":False,"error":"JSON required"})
        try:
            size=int(self.headers.get("Content-Length") or "0")
            if size<=0 or size>1800:return self.reply(413,{"ok":False,"error":"Request too large"})
            d=json.loads(self.rfile.read(size))
            if not isinstance(d,dict) or set(d)-{"question","mode","include_evaluation"}:
                raise ValueError("Unknown request fields")
            if d.get("mode")!="three_way" or not isinstance(d.get("include_evaluation"),bool):
                raise ValueError("Unsupported experiment")
            question=validate_question(d.get("question"))
        except (ValueError,UnicodeDecodeError,TypeError):
            return self.reply(400,{"ok":False,"error":"Invalid question or experiment mode"})
        if not single.acquire(blocking=False):
            return self.reply(429,{"ok":False,"error":"The original engine is processing another question"},
                              {"Retry-After":"15"})
        try:
            return self.reply(200,run_three_systems(question,d["include_evaluation"]))
        except Exception:
            logging.exception("Three-system experiment execution failed")
            return self.reply(502,{"ok":False,"error":"Original telecom experiment could not complete"})
        finally:single.release()


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    # Production requires a deliberate cold-start CPU/readiness strategy.
    threading.Thread(target=prewarm,name="rag-prewarm",daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0",PORT),Handler).serve_forever()
