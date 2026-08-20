import time
from typing import List, Dict, Optional
from harness.state import Goal, Task, TaskStatus, EvaluationVerdict


class ManagerAgent:
    """
    Manages task decomposition, dependencies, scheduling, and feedback integration.
    """
    def __init__(self, name: str = "Manager"):
        self.name = name

    def decompose_goal(self, goal: Goal) -> List[Task]:
        """
        Decomposes a top-level goal into a sequence or DAG of actionable subtasks.
        """
        tasks: List[Task] = []
        for i, dod_item in enumerate(goal.definition_of_done):
            task_id = f"task_{i+1}"
            dependencies = [f"task_{i}"] if i > 0 else []
            
            task = Task(
                id=task_id,
                title=f"Implement: {dod_item}",
                description=f"Fulfill acceptance criteria for: {dod_item}",
                definition_of_done=[dod_item],
                dependencies=dependencies,
                status=TaskStatus.PENDING,
            )
            tasks.append(task)
        return tasks

    def select_next_task(self, tasks: Dict[str, Task]) -> Optional[Task]:
        """
        Finds the highest priority task ready to execute (all dependencies completed).
        """
        completed_ids = {t.id for t in tasks.values() if t.status == TaskStatus.COMPLETED}
        
        for task in tasks.values():
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    return task
        return None

    def handle_task_evaluation(
        self,
        task: Task,
        verdict: EvaluationVerdict,
    ) -> None:
        """
        Updates task state based on independent Evaluator verdict.
        """
        if verdict.passed:
            task.mark_completed()
        else:
            task.mark_failed(verdict.feedback)
