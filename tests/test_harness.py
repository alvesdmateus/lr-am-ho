import os
import shutil
import unittest
from harness.state import Goal, TaskStatus, EvaluationVerdict
from harness.engine import HarnessEngine


class TestHarnessEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_harness_core"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_goal_execution_loop(self):
        goal = Goal(
            id="goal_1",
            objective="Build and verify authentication system",
            definition_of_done=[
                "Implement JWT token generation",
                "Implement login endpoint",
                "Add test suite for auth module",
            ],
        )

        engine = HarnessEngine(goal=goal, storage_dir=self.test_dir)
        final_state = engine.run_until_completion()

        # Check that all tasks are completed and goal is marked done
        self.assertTrue(final_state.goal.is_completed)
        self.assertEqual(len(final_state.tasks), 3)
        for task in final_state.tasks.values():
            self.assertEqual(task.status, TaskStatus.COMPLETED)

        # Verify state persistence file exists
        self.assertTrue(os.path.exists(f"{self.test_dir}/state/harness_state.json"))
        self.assertTrue(os.path.exists(f"{self.test_dir}/state/events.jsonl"))

    def test_checkpoint_resume_capability(self):
        goal = Goal(
            id="goal_resume",
            objective="Multi-step refactoring",
            definition_of_done=["Step 1", "Step 2", "Step 3"],
        )

        # Run 1 step
        engine1 = HarnessEngine(goal=goal, storage_dir=self.test_dir)
        engine1.run_step()
        self.assertEqual(engine1.state.iteration_count, 1)

        # Simulate restart by instantiating new engine pointing to same storage
        engine2 = HarnessEngine(goal=goal, storage_dir=self.test_dir)
        self.assertEqual(engine2.state.iteration_count, 1)
        self.assertEqual(len(engine2.state.tasks), 3)

        # Continue to completion
        final_state = engine2.run_until_completion()
        self.assertTrue(final_state.goal.is_completed)

    def test_failure_halt_guard(self):
        goal = Goal(
            id="goal_fail",
            objective="Failing task goal",
            definition_of_done=["Impossible task"],
        )

        engine = HarnessEngine(goal=goal, storage_dir=self.test_dir, max_consecutive_failures=2)

        # Force evaluator to fail
        engine.evaluator.evaluate_task = lambda task, execution_result: EvaluationVerdict(
            task_id=task.id,
            passed=False,
            score=0.0,
            feedback="Intentional simulated failure",
            invalidated_memory_ids=task.used_memory_ids,
        )

        final_state = engine.run_until_completion()
        self.assertTrue(final_state.is_halted)
        self.assertIn("consecutive failure", final_state.halt_reason.lower())
        self.assertFalse(final_state.goal.is_completed)


if __name__ == "__main__":
    unittest.main()
