"""Offline native API contract tests; never load FAISS, BGE-M3 or call OpenRouter."""
import http.client
import asyncio
import json
import threading
import sys
import types
import time
import unittest
from unittest.mock import patch

import api_adapter
import api_server


class NativeAPIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = api_server.ThreadingHTTPServer(("127.0.0.1", 0), api_server.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=3)

    def call(self, method, path, payload=None, token=None, content_type="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        headers = {"Content-Type": content_type}
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        body = None if payload is None else json.dumps(payload)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        result = resp.status, json.loads(resp.read().decode("utf-8"))
        conn.close()
        return result

    def test_health_check_and_unknown_route(self):
        self.assertEqual(self.call("GET", "/healthz")[0], 200)
        self.assertEqual(self.call("GET", "/not-found")[0], 404)

    def test_missing_opt_in_stays_closed(self):
        with patch.object(api_server, "ENABLED", False), \
             patch.object(api_server, "TOKEN", "X" * 40), \
             patch.object(api_server, "ready", True):
            status, data = self.call("GET", "/api/v1/ready")
            self.assertEqual(status, 503)
            self.assertFalse(data["ready"])

    def test_ready_when_original_engine_prepared(self):
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", "X" * 40), \
             patch.object(api_server, "ready", True):
            status, data = self.call("GET", "/api/v1/ready")
            self.assertEqual(status, 200)
            self.assertTrue(data["ready"])

    def test_post_requires_backend_secret(self):
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", "X" * 40), \
             patch.object(api_server, "ready", True):
            status, _ = self.call("POST", "/api/v1/answer", {
                "question": "Role of AMF?", "mode": "three_way",
                "include_evaluation": False,
            })
            self.assertEqual(status, 401)

    def test_warming_post_returns_503(self):
        token = "X" * 40
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", token), \
             patch.object(api_server, "ready", False):
            status, _ = self.call("POST", "/api/v1/answer", {
                "question": "Role of AMF?", "mode": "three_way",
                "include_evaluation": False,
            }, token=token)
            self.assertEqual(status, 503)

    def test_malformed_request_rejected_before_inference(self):
        token = "X" * 40
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", token), \
             patch.object(api_server, "ready", True), \
             patch.object(api_server, "run_three_systems") as engine:
            cases = [
                {"question": "x", "mode": "three_way", "include_evaluation": False},
                {"question": "Role of AMF?", "mode": "one_way", "include_evaluation": False},
                {"question": "Role of AMF?", "mode": "three_way", "include_evaluation": "yes"},
                {"question": "Role of AMF?", "mode": "three_way",
                 "include_evaluation": False, "private_input": "not allowed"},
            ]
            for invalid in cases:
                status, _ = self.call("POST", "/api/v1/answer", invalid, token=token)
                self.assertEqual(status, 400)
            engine.assert_not_called()

    def test_valid_post_runs_original_engine_handler(self):
        token = "X" * 40
        expected = {"ok": True, "schema_version": "tebunor-rag-1", "paths": {}}
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", token), \
             patch.object(api_server, "ready", True), \
             patch.object(api_server, "run_three_systems", return_value=expected) as engine:
            status, payload = self.call("POST", "/api/v1/answer", {
                "question": "Role of AMF?", "mode": "three_way",
                "include_evaluation": True,
            }, token=token)
            self.assertEqual(status, 200)
            self.assertEqual(payload, expected)
            engine.assert_called_once_with("Role of AMF?", True)

    def test_concurrent_request_is_bounded(self):
        token = "X" * 40
        with patch.object(api_server, "ENABLED", True), \
             patch.object(api_server, "TOKEN", token), \
             patch.object(api_server, "ready", True):
            self.assertTrue(api_server.single.acquire(blocking=False))
            try:
                status, _ = self.call("POST", "/api/v1/answer", {
                    "question": "Role of AMF?", "mode": "three_way",
                    "include_evaluation": False,
                }, token=token)
                self.assertEqual(status, 429)
            finally:
                api_server.single.release()

    def test_projection_keeps_only_public_evidence_metadata(self):
        paths = {
            key: {
                "metadata": {"name": key, "mode": key, "objective": "original",
                             "policy": "tested", "sources": ["Gemma", "FAISS"]},
                "result": {"answer": "Original result for " + key,
                           "final_context": "PRIVATE_CORPUS_CONTENT",
                           "final_evidence": [{
                               "evidence_id": "RAG:1", "title": "3GPP",
                               "source_family": "3GPP", "text": "PRIVATE_PASSAGE",
                               "file_path": "/secret/path", "chunk_id": "abc",
                           }]},
                "timing": {"total_wall_s": 4.2},
            } for key in api_adapter.PATHS
        }
        data = api_adapter.public_result({
            "question": "Role of AMF?",
            "query_type": "TELECOM_GROUNDED",
            "paths": paths,
            "timing": {"total_wall_s": 10.1},
        })
        self.assertEqual(set(data["paths"]), set(api_adapter.PATHS))
        self.assertEqual(data["paths"]["strict"]["answer"], "Original result for strict")
        self.assertNotIn("PRIVATE_CORPUS_CONTENT", str(data))
        self.assertNotIn("PRIVATE_PASSAGE", str(data))
        self.assertNotIn("/secret/", str(data))


    def test_full_api_timing_includes_optional_granite_evaluation(self):
        # Replaces the heavy original model modules with isolated stubs.
        compare = types.ModuleType("runtime.comparison")
        judges = types.ModuleType("runtime.evaluation_threeway")

        async def fake_three_paths(question):
            return {
                "sanitized_question": question,
                "query_type": "TELECOM_GROUNDED",
                "timing": {"total_wall_s": 2.0},
                "paths": {
                    key: {"result": {"answer": "Answer " + key},
                          "timing": {"total_wall_s": 1.0}}
                    for key in ("llm_only", "strict", "augmented")
                },
            }

        def fake_granite_evaluation(**kwargs):
            time.sleep(0.01)
            return {"applicable": True, "systems": {}}

        compare.run_three_path_experiment = fake_three_paths
        judges.evaluate_three_way_telecom = fake_granite_evaluation
        judges.run_three_way_quality_judge = lambda *a, **k: {}

        with patch.dict(sys.modules, {
            "runtime.comparison": compare,
            "runtime.evaluation_threeway": judges,
        }):
            result = api_server.run_three_systems("Role of AMF?", True)
            no_eval = api_server.run_three_systems("Role of AMF?", False)

        timing = result["timing"]
        self.assertEqual(timing["total_wall_s"], 2.0)  # unchanged engine contract
        self.assertGreater(timing["evaluation_wall_s"], 0)
        self.assertGreaterEqual(
            timing["api_total_wall_s"], timing["evaluation_wall_s"]
        )
        self.assertGreaterEqual(
            timing["api_total_wall_s"], timing["three_path_api_wall_s"]
        )
        self.assertIsNone(no_eval["timing"]["evaluation_wall_s"])
        self.assertIsNone(no_eval["evaluation"])

    def test_stream_sse_requires_auth_and_remains_bounded(self):
        token="X"*40
        payload={"question":"Role of AMF?","mode":"three_way","include_evaluation":True}
        async def fake_stream(question,wants_eval,emit):
            self.assertEqual(question,"Role of AMF?")
            self.assertTrue(wants_eval)
            emit("accepted",{"ok":True})
            emit("evaluation_started",{"ok":True})
            emit("complete",{"ok":True,"paths":{"strict":{"answer":"real stub"}}})
        with patch.object(api_server,"ENABLED",True), \
             patch.object(api_server,"TOKEN",token), \
             patch.object(api_server,"ready",True), \
             patch.object(api_server,"stream_three_systems",side_effect=fake_stream):
            status,_=self.call("POST","/api/v1/stream",payload)
            self.assertEqual(status,401)
            conn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=3)
            conn.request("POST","/api/v1/stream",body=json.dumps(payload),
                         headers={"Content-Type":"application/json","Authorization":"Bearer "+token})
            response=conn.getresponse()
            wire=response.read().decode("utf-8")
            self.assertEqual(response.status,200)
            self.assertIn("text/event-stream",response.getheader("Content-Type"))
            self.assertIn("event: accepted\n",wire)
            self.assertIn("event: evaluation_started\n",wire)
            self.assertIn("event: complete\n",wire)
            self.assertLess(wire.index("event: accepted"),wire.index("event: evaluation_started"))
            self.assertLess(wire.index("event: evaluation_started"),wire.index("event: complete"))
            conn.close()

    def test_progressive_events_are_real_bounded_results_not_placeholders(self):
        compare=types.ModuleType("runtime.comparison")
        judges=types.ModuleType("runtime.evaluation_threeway")
        async def fake_paths(question,on_path=None):
            paths={}
            for key in ("strict","llm_only","augmented"):
                raw={"metadata":{"name":key,"sources":["3GPP"]},
                     "result":{"answer":"Answered "+key,
                               "final_context":"PRIVATE_PASSAGE",
                               "final_evidence":[{"title":"3GPP","source_family":"3GPP",
                                                  "text":"PRIVATE_CORPUS_CONTENT"}]},
                     "timing":{"total_wall_s":1.5}}
                paths[key]=raw
                await asyncio.sleep(0.01)
                on_path(key,raw)
            return {"question":question,"sanitized_question":question,
                    "query_type":"TELECOM_GROUNDED","paths":paths,
                    "timing":{"total_wall_s":0.03}}
        def fake_granite_evaluation(**kwargs):
            time.sleep(0.01)
            return {"applicable":True,"systems":{"llm_only":{"relevance_score":88}}}
        compare.run_three_path_experiment=fake_paths
        judges.evaluate_three_way_telecom=fake_granite_evaluation
        judges.run_three_way_quality_judge=lambda *args,**kwargs: {}
        events=[]
        with patch.dict(sys.modules,{"runtime.comparison":compare,
                                      "runtime.evaluation_threeway":judges}):
            asyncio.run(api_server.stream_three_systems(
                "Role of AMF?",True,lambda name,data:events.append((name,data))))
        names=[e[0] for e in events]
        self.assertEqual([name for name in names if name=="path_complete"],
                         ["path_complete"]*3)
        self.assertEqual([data["key"] for name,data in events if name=="path_complete"],
                         ["strict","llm_only","augmented"])
        self.assertLess(names.index("path_complete"),names.index("evaluation_started"))
        self.assertLess(names.index("evaluation_started"),names.index("evaluation_complete"))
        self.assertLess(names.index("evaluation_complete"),names.index("complete"))
        self.assertEqual(events[-1][1]["evaluation"]["systems"]["llm_only"]["relevance_score"],88.0)
        self.assertNotIn("PRIVATE_PASSAGE",str(events))
        self.assertNotIn("PRIVATE_CORPUS_CONTENT",str(events))

    def test_question_length_guard(self):
        for invalid in ("", "a", "x" * 501, None):
            with self.assertRaises(ValueError):
                api_adapter.validate_question(invalid)


if __name__ == "__main__":
    unittest.main()
