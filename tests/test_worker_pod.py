import os
import shutil
import unittest
from harness.state import Task, TaskStatus
from tools.protocol import ToolType
from agent.worker import Worker, MicroPlanner, MicroAction


class TestWorkerPod(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_worker_pod_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.worker = Worker(workspace_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_micro_planner_generates_steps(self):
        task = Task(
            id="t_api",
            title="Create login API route",
            description="Add /api/login endpoint with JWT response",
            definition_of_done=["Endpoint returns 200 on valid credentials"],
        )
        planner = MicroPlanner()
        steps = planner.plan_steps(task, memory_briefing="Follow REST conventions")
        self.assertGreaterEqual(len(steps), 2)
        self.assertEqual(steps[0].step_number, 1)

    def test_worker_executes_micro_loop(self):
        task = Task(
            id="t_calc",
            title="Implement calculate function",
            description="Write add function in calc.py",
            definition_of_done=["Function returns sum"],
        )
        
        custom_actions = [
            MicroAction(
                step_number=1,
                intent="Write calculation module",
                tool_type=ToolType.FILE_WRITE,
                parameters={"file_path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
            ),
            MicroAction(
                step_number=2,
                intent="Verify module with python assertion",
                tool_type=ToolType.BASH,
                parameters={"command": "python -c \"import calc; assert calc.add(2, 3) == 5\""},
            ),
        ]

        result = self.worker.execute_task(
            task=task,
            memory_briefing="Clean code",
            custom_micro_actions=custom_actions,
            simulated_discoveries=["calc.py can be imported as top-level module"],
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_outputs), 2)
        self.assertIn("calc.py", result.discovered_learnings[0])
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "calc.py")))

    def test_worker_halts_on_failing_micro_step(self):
        task = Task(
            id="t_fail",
            title="Task with failing step",
            description="Testing error handling in micro-loop",
        )
        custom_actions = [
            MicroAction(
                step_number=1,
                intent="Run failing assertion",
                tool_type=ToolType.BASH,
                parameters={"command": "python -c \"assert 1 == 2\""},
            ),
            MicroAction(
                step_number=2,
                intent="Should not be reached",
                tool_type=ToolType.BASH,
                parameters={"command": "echo 'Should not run'"},
            ),
        ]

        result = self.worker.execute_task(task=task, memory_briefing="", custom_micro_actions=custom_actions)
        self.assertFalse(result.success)
        # Should have stopped after step 1
        self.assertEqual(len(result.tool_outputs), 1)
        self.assertFalse(result.tool_outputs[0]["success"])


if __name__ == "__main__":
    unittest.main()
