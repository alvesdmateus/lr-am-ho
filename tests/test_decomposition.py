import unittest
from harness.goal_parser import GoalParser
from harness.decomposition import DecompositionValidator
from harness.state import Task, TaskStatus


class TestGoalParser(unittest.TestCase):
    def setUp(self):
        self.parser = GoalParser()

    def test_parse_plain_text(self):
        goal = self.parser.parse("Build user registration system")
        self.assertEqual(goal.objective, "Build user registration system")
        self.assertTrue(goal.id.startswith("goal_"))
        self.assertEqual(len(goal.definition_of_done), 0)

    def test_parse_with_bulleted_hints(self):
        text = (
            "Build authentication\n"
            "- Implement login endpoint\n"
            "- Add JWT generation\n"
            "- Write unit tests\n"
        )
        goal = self.parser.parse(text)
        self.assertEqual(len(goal.definition_of_done), 3)
        self.assertIn("Implement login endpoint", goal.definition_of_done)
        self.assertIn("Add JWT generation", goal.definition_of_done)

    def test_parse_with_numbered_hints(self):
        text = (
            "Deploy to production\n"
            "1. Run migrations\n"
            "2. Build docker image\n"
            "3. Deploy container\n"
        )
        goal = self.parser.parse(text)
        self.assertEqual(len(goal.definition_of_done), 3)
        self.assertIn("Run migrations", goal.definition_of_done)

    def test_parse_empty_raises_error(self):
        with self.assertRaises(ValueError):
            self.parser.parse("   ")


class TestDecompositionValidator(unittest.TestCase):
    def setUp(self):
        self.validator = DecompositionValidator(max_tasks=5)

    def test_valid_dag_passes(self):
        tasks = [
            Task(id="task_1", title="Setup DB", definition_of_done=["DB running"], dependencies=[]),
            Task(id="task_2", title="Add Models", definition_of_done=["Models imported"], dependencies=["task_1"]),
            Task(id="task_3", title="Add Tests", definition_of_done=["Tests pass"], dependencies=["task_2"]),
        ]
        errors = self.validator.validate(tasks)
        self.assertEqual(errors, [])

    def test_missing_definition_of_done_caught(self):
        tasks = [
            Task(id="task_1", title="No DoD Task", definition_of_done=[], dependencies=[]),
        ]
        errors = self.validator.validate(tasks)
        self.assertTrue(any("Missing definition_of_done" in e for e in errors))

    def test_unknown_dependency_caught(self):
        tasks = [
            Task(id="task_1", title="T1", definition_of_done=["Done"], dependencies=["non_existent_task"]),
        ]
        errors = self.validator.validate(tasks)
        self.assertTrue(any("unknown dependency" in e for e in errors))

    def test_forward_dependency_caught(self):
        tasks = [
            Task(id="task_1", title="T1", definition_of_done=["Done"], dependencies=["task_2"]),
            Task(id="task_2", title="T2", definition_of_done=["Done"], dependencies=[]),
        ]
        errors = self.validator.validate(tasks)
        self.assertTrue(any("Forward dependency" in e for e in errors))

    def test_dependency_cycle_caught(self):
        tasks = [
            Task(id="task_1", title="T1", definition_of_done=["Done"], dependencies=["task_3"]),
            Task(id="task_2", title="T2", definition_of_done=["Done"], dependencies=["task_1"]),
            Task(id="task_3", title="T3", definition_of_done=["Done"], dependencies=["task_2"]),
        ]
        errors = self.validator.validate(tasks)
        self.assertTrue(any("cycle" in e.lower() for e in errors))

    def test_too_many_tasks_caught(self):
        tasks = [
            Task(id=f"task_{i}", title=f"T{i}", definition_of_done=["Done"], dependencies=[f"task_{i-1}"] if i > 0 else [])
            for i in range(7)
        ]
        errors = self.validator.validate(tasks)
        self.assertTrue(any("Too many tasks" in e for e in errors))

    def test_zero_tasks_caught(self):
        errors = self.validator.validate([])
        self.assertTrue(any("zero tasks" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
