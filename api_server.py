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


def evaluate_completed_original(r:dict,question:str):
    """Original Granite judge, invoked only after all three paths finish."""
    from runtime.evaluation_threeway import (
        evaluate_three_way_telecom,run_three_way_quality_judge)
    paths=r["paths"]
    result=lambda key:paths[key]["result"] or {}
    answer=lambda key:str(result(key).get("answer") or "")
    q=r.get("sanitized_question") or question
    if r.get("query_type")=="TELECOM_GROUNDED":
        strict=result("strict");aug=result("augmented")
        return evaluate_three_way_telecom(
            question=q,
            llm_only_answer=answer("llm_only"),
            strict_answer=answer("strict"),
            augmented_answer=answer("augmented"),
            strict_evidence_context=strict.get("final_context") or "",
            strict_evidence_items=strict.get("final_evidence") or [],
            augmented_evidence_context=aug.get("final_context") or "",
            augmented_evidence_items=aug.get("final_evidence") or [],
        )
    v=run_three_way_quality_judge(
        q,answer("llm_only"),answer("strict"),answer("augmented"))["payload"]
    return {"applicable":False,
        "systems":{key:{
            "relevance_score":v[answer_key].get("relevance_score"),
            "completeness_score":v[answer_key].get("completeness_score"),
            "hallucination_risk_pct":None,
            "hallucination_risk_label":"N/A"
        } for key,answer_key in (
            ("llm_only","answer_a"),("strict","answer_b"),("augmented","answer_c"))},
        "comparison_summary":v.get("comparison_summary"),
        "question_intent":v.get("question_intent")}


def run_three_systems(question:str,include_evaluation:bool):
    # Backward-compatible original synchronous JSON route.
    started=time.perf_counter()
    from runtime.comparison import run_three_path_experiment
    r=asyncio.run(run_three_path_experiment(question))
    three_path_s=round(time.perf_counter()-started,3)
    evaluation=None
    evaluation_s=None
    if include_evaluation:
        evaluation_started=time.perf_counter()
        evaluation=evaluate_completed_original(r,question)
        evaluation_s=round(time.perf_counter()-evaluation_started,3)
    output=public_result(r,evaluation)
    output["timing"].update({
        "api_total_wall_s":round(time.perf_counter()-started,3),
        "three_path_api_wall_s":three_path_s,
        "evaluation_wall_s":evaluation_s,
    })
    return output


async def stream_three_systems(question:str,include_evaluation:bool,send):
    """Send real path-completion and Granite-stage SSE events; no synthetic answers."""
    from runtime.comparison import run_three_path_experiment
    from api_adapter import public_path,safe_evaluation
    started=time.perf_counter()
    pending=asyncio.Queue()
    event_loop=asyncio.get_running_loop()
    def observe_path(key,raw):
        # Full-answer callback runs in the original comparison event loop.
        pending.put_nowait(("path_complete",key,raw))
    def observe_token(key,delta):
        # SDK token callbacks run in existing model worker threads.
        # Schedule into this one asyncio event loop; never write to HTTP
        # response streams from the provider threads.
        if not delta or key not in ("llm_only","strict","augmented"):
            return
        for offset in range(0,len(delta),512):
            part=delta[offset:offset+512]
            try:
                current_loop=asyncio.get_running_loop()
            except RuntimeError:
                current_loop=None
            if current_loop is event_loop:
                # Tests or future native async providers already on this loop:
                # preserve token-before-path order synchronously.
                pending.put_nowait(("answer_delta",key,part))
            else:
                # Original OpenAI client executes inside asyncio.to_thread.
                event_loop.call_soon_threadsafe(
                    pending.put_nowait,("answer_delta",key,part))
    send("accepted",{"ok":True,"evaluation_requested":include_evaluation,
                     "token_streaming":True})
    run_task=asyncio.create_task(
        run_three_path_experiment(question,on_path=observe_path,
                                  on_token=observe_token))
    while not run_task.done() or not pending.empty():
        try:
            event,key,payload=await asyncio.wait_for(pending.get(),timeout=3.0)
            if event=="answer_delta":
                send("answer_delta",{"key":key,"text":payload})
            else:
                send("path_complete",{"key":key,"path":public_path(payload,key)})
        except asyncio.TimeoutError:
            send("heartbeat",{"phase":"reasoning"})
    r=await run_task
    three_path_s=round(time.perf_counter()-started,3)
    evaluation=None
    evaluation_s=None
    if include_evaluation:
        send("evaluation_started",{"ok":True})
        judge_started=time.perf_counter()
        judge_task=asyncio.create_task(
            asyncio.to_thread(evaluate_completed_original,r,question))
        while not judge_task.done():
            await asyncio.sleep(3)
            if not judge_task.done():
                send("heartbeat",{"phase":"evaluation"})
        try:
            evaluation=await judge_task
        except Exception:
            logging.exception("Original Granite judge failed; keeping completed answers")
            send("evaluation_failed",{"ok":False,"error":"Granite assessment unavailable; the three original answers remain available."})
        evaluation_s=round(time.perf_counter()-judge_started,3)
        if evaluation is not None:
            send("evaluation_complete",{"ok":True,"evaluation":safe_evaluation(evaluation),
                                        "elapsed_s":evaluation_s})
    output=public_result(r,evaluation)
    output["timing"].update({
        "api_total_wall_s":round(time.perf_counter()-started,3),
        "three_path_api_wall_s":three_path_s,
        "evaluation_wall_s":evaluation_s,
    })
    send("complete",output)


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
        streaming=self.path=="/api/v1/stream"
        if self.path!="/api/v1/answer" and not streaming:
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
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream; charset=utf-8")
                self.send_header("Cache-Control","no-cache, no-store")
                self.send_header("X-Content-Type-Options","nosniff")
                self.send_header("X-Accel-Buffering","no")
                self.end_headers()
                def send_event(name,data):
                    payload=json.dumps(data,ensure_ascii=False,separators=(",",":"))
                    wire=("event: "+name+"\n"+"data: "+payload+"\n\n").encode("utf-8")
                    self.wfile.write(wire)
                    self.wfile.flush()
                try:
                    asyncio.run(stream_three_systems(question,d["include_evaluation"],send_event))
                except (BrokenPipeError,ConnectionResetError):
                    logging.info("RAG streaming client disconnected")
                except Exception:
                    logging.exception("RAG streaming experiment execution failed")
                    try:send_event("error",{"ok":False,"error":"Original telecom experiment could not complete"})
                    except (BrokenPipeError,ConnectionResetError):pass
                return
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
