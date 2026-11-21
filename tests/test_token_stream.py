"""Pure offline tests of the original Gemma completion streaming adapter."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.token_stream import final_answer_completion, output_tokens_for


def obj(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeCompletions:
    def __init__(self, output=None, fail=False):
        self.calls=[]
        self.output=output or [
            obj(model="google/gemma-4-26b-a4b-it",
                choices=[obj(delta=obj(content="Hello "),finish_reason=None)],
                usage=None),
            obj(model="google/gemma-4-26b-a4b-it",
                choices=[obj(delta=obj(content="world."),finish_reason="stop")],
                usage=None),
            obj(model="google/gemma-4-26b-a4b-it",
                choices=[],usage=obj(prompt_tokens=30,completion_tokens=4)),
        ]
        self.fail=fail
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider unavailable")
        if kwargs.get("stream"):
            return iter(self.output)
        return obj(model=kwargs.get("model"),usage=obj(prompt_tokens=30,completion_tokens=4),
                   choices=[obj(message=obj(content="Original nonstreamed result"),
                                finish_reason="stop")])


class StreamContractTests(unittest.TestCase):
    def test_no_observer_keeps_original_nonstream_api_unchanged(self):
        completions=FakeCompletions()
        client=obj(chat=obj(completions=completions))
        result=final_answer_completion(client,model="gemma",messages=[],max_tokens=900)
        self.assertEqual(result.choices[0].message.content,"Original nonstreamed result")
        self.assertEqual(len(completions.calls),1)
        self.assertEqual(completions.calls[0],
                         {"model":"gemma","messages":[],"max_tokens":900})

    def test_streams_real_visible_tokens_and_retains_usage(self):
        completions=FakeCompletions()
        client=obj(chat=obj(completions=completions))
        events=[]
        with output_tokens_for("llm_only",lambda path,text:events.append((path,text))):
            response=final_answer_completion(client,model="gemma",messages=[],
                                             max_tokens=900)
        self.assertEqual(events,[("llm_only","Hello "),("llm_only","world.")])
        self.assertEqual(response.choices[0].message.content,"Hello world.")
        self.assertEqual(response.choices[0].finish_reason,"stop")
        self.assertEqual(response.usage.completion_tokens,4)
        self.assertEqual(len(completions.calls),1)  # No duplicate inference
        self.assertIs(completions.calls[0]["stream"],True)
        self.assertTrue(completions.calls[0]["stream_options"]["include_usage"])

    def test_observer_is_path_local_and_propagates_through_to_thread(self):
        async def workload():
            completions=FakeCompletions()
            client=obj(chat=obj(completions=completions))
            output=[]
            with output_tokens_for("strict",lambda path,text:output.append((path,text))):
                result=await asyncio.to_thread(
                    final_answer_completion,client,model="gemma",messages=[])
            return output,result.choices[0].message.content
        output,result=asyncio.run(workload())
        self.assertEqual(output,[("strict","Hello "),("strict","world.")])
        self.assertEqual(result,"Hello world.")

    def test_stream_error_is_not_a_fabricated_success(self):
        chunks=[
            obj(model="gemma",choices=[obj(delta=obj(content="Partial"),
                                           finish_reason=None)],usage=None),
            obj(model="gemma",choices=[obj(delta=obj(content=""),
                                           finish_reason="error")],usage=None)
        ]
        completions=FakeCompletions(chunks)
        client=obj(chat=obj(completions=completions))
        with output_tokens_for("augmented",lambda *_:None):
            with self.assertRaisesRegex(RuntimeError,"terminated with error"):
                final_answer_completion(client,model="gemma",messages=[])

    def test_observer_failure_does_not_interrupt_original_model(self):
        def failed_observer(path,text):
            raise RuntimeError("Browser closed")
        client=obj(chat=obj(completions=FakeCompletions()))
        with output_tokens_for("strict",failed_observer):
            result=final_answer_completion(client,model="gemma",messages=[])
        self.assertEqual(result.choices[0].message.content,"Hello world.")

    def test_path_observer_does_not_leak_across_requests(self):
        events=[]
        client=obj(chat=obj(completions=FakeCompletions()))
        with output_tokens_for("llm_only",lambda path,text:events.append(text)):
            final_answer_completion(client,model="gemma",messages=[])
        count=len(events)
        final_answer_completion(client,model="gemma",messages=[])
        self.assertEqual(len(events),count)
        self.assertFalse(client.chat.completions.calls[-1].get("stream",False))

    def test_rejects_oversized_partial_output(self):
        chunks=[obj(model="gemma",
                    choices=[obj(delta=obj(content="x"*12001),
                                 finish_reason="stop")],usage=None)]
        client=obj(chat=obj(completions=FakeCompletions(chunks)))
        with output_tokens_for("strict",lambda *_:None):
            with self.assertRaisesRegex(RuntimeError,"length limit"):
                final_answer_completion(client,model="gemma",messages=[])


if __name__=="__main__":
    unittest.main()
