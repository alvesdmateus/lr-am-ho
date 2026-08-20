import json
import unittest
from unittest.mock import MagicMock, patch
from llm.client import LLMClient, LLMResponse, ToolCall
from agent.manager import ManagerAgent
from agent.evaluator import EvaluatorAgent
from harness.state import Goal, Task, ExecutionResult


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(base_url="http://mock:11434/v1", api_key="test", model="mock-model")

    def tearDown(self):
        self.client.close()

    @patch("httpx.Client.post")
    def test_chat_text_response_parsing(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        res = self.client.chat(messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(res.content, "Hello world")
        self.assertEqual(res.finish_reason, "stop")
        self.assertEqual(res.usage["total_tokens"], 15)
        self.assertEqual(len(res.tool_calls), 0)

    @patch("httpx.Client.post")
    def test_chat_tool_call_parsing(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "echo 'test'"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        res = self.client.chat(messages=[{"role": "user", "content": "Run echo"}])
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].name, "bash")
        self.assertEqual(res.tool_calls[0].arguments["command"], "echo 'test'")
        self.assertEqual(res.tool_calls[0].id, "call_123")


class TestAgentLLMIntegration(unittest.TestCase):
    @patch("llm.client.LLMClient.chat")
    def test_manager_llm_decomposition(self, mock_chat):
        mock_chat.return_value = LLMResponse(
            content=json.dumps({
                "tasks": [
                    {
                        "id": "task_1",
                        "title": "Install dependencies",
                        "description": "Add pyjwt",
                        "definition_of_done": ["pip install succeeds"],
                        "dependencies": [],
                    },
                    {
                        "id": "task_2",
                        "title": "Create auth module",
                        "description": "Add auth/jwt.py",
                        "definition_of_done": ["auth.jwt importable"],
                        "dependencies": ["task_1"],
                    },
                ]
            }),
            finish_reason="stop",
        )

        client = LLMClient(base_url="http://mock:11434/v1")
        manager = ManagerAgent(llm_client=client)
        goal = Goal(id="g1", objective="Add auth")

        tasks = manager.decompose_goal(goal)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].id, "task_1")
        self.assertEqual(tasks[1].dependencies, ["task_1"])
        client.close()

    @patch("llm.client.LLMClient.chat")
    def test_evaluator_llm_audit(self, mock_chat):
        mock_chat.return_value = LLMResponse(
            content=json.dumps({
                "passed": True,
                "score": 0.95,
                "rubric_results": {"criterion_1": True},
                "feedback": "All criteria met.",
                "suggested_actions": [],
            }),
            finish_reason="stop",
        )

        client = LLMClient(base_url="http://mock:11434/v1")
        evaluator = EvaluatorAgent(llm_client=client)
        task = Task(id="t1", title="T1", definition_of_done=["criterion_1"])
        exec_res = ExecutionResult(task_id="t1", success=True, summary_of_changes="Done")

        verdict = evaluator.evaluate_task(task, exec_res)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.score, 0.95)
        self.assertEqual(verdict.feedback, "All criteria met.")
        client.close()


if __name__ == "__main__":
    unittest.main()
