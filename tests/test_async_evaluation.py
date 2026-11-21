"""Pure tests for optional async Granite job security; no Google SDK or models."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import async_evaluation


class AsyncGraniteJobContract(unittest.TestCase):
    def test_requires_all_explicit_feature_and_cloud_bindings(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(async_evaluation.enabled())
            with self.assertRaises(async_evaluation.JobUnavailable):
                async_evaluation.start_job("Role of AMF?")

    def test_enabled_with_complete_settings(self):
        env = {
            "TELECOM_RAG_ASYNC_ENABLED": "true",
            "TELECOM_RAG_TASKS_QUEUE": "projects/test/locations/europe-west2/queues/rag",
            "TELECOM_RAG_TASK_TARGET": "https://example.run.app/api/v1/internal/evaluation-run",
            "TELECOM_RAG_TASK_AUDIENCE": "https://example.run.app",
            "TELECOM_RAG_TASK_IDENTITY": "worker@example.iam.gserviceaccount.com",
            "GOOGLE_CLOUD_PROJECT": "test",
        }
        with patch.dict("os.environ", env, clear=True):
            self.assertTrue(async_evaluation.enabled())

    def test_job_ids_are_strict_hex(self):
        for bad in ("../../etc/passwd", "not-a-job", "a"*31, "A"*32, "", None):
            with self.assertRaises(async_evaluation.JobNotFound):
                async_evaluation._ref(None, bad)

    def test_expired_result_refused_without_waiting_for_firestore_ttl(self):
        now = datetime.now(timezone.utc)
        self.assertTrue(async_evaluation._expired({"expires_at":now-timedelta(seconds=1)}))
        self.assertFalse(async_evaluation._expired({"expires_at":now+timedelta(minutes=10)}))
        self.assertTrue(async_evaluation._expired({}))

    def test_unsigned_internal_task_is_never_accepted_when_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(async_evaluation.verify_task_oidc("Bearer "+("a"*5000)))
            self.assertFalse(async_evaluation.verify_task_oidc("Bearer untrusted"))
            self.assertFalse(async_evaluation.verify_task_oidc(None))


if __name__ == "__main__":
    unittest.main()
