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
    from runtime.comparison import run_three_path_experiment
    r=asyncio.run(run_three_path_experiment(question))
    evaluation=None
    if include_evaluation:
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
    return public_result(r,evaluation)


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
        if self.path=="/healthz":
            return self.reply(200,{"ok":True,"service":"telecom-rag-api"})
        if self.path=="/api/v1/ready":
            return self.reply(200 if is_ready() else 503,
                              {"ok":is_ready(),"ready":is_ready()},
                              {"Retry-After":"20"} if not is_ready() else None)
        return self.reply(404,{"ok":False,"error":"Not found"})
    def do_POST(self):
        if self.path!="/api/v1/answer":
            return self.reply(404,{"ok":False,"error":"Not found"})
        if len(TOKEN)<32 or not hmac.compare_digest(
            self.headers.get("Authorization",""),"Bearer "+TOKEN):
            return self.reply(401,{"ok":False,"error":"Unauthorised"})
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
