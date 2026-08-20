from typing import List, Dict, Set
from harness.state import Task


class DecompositionError(Exception):
    """Raised when goal decomposition fails after all retry attempts."""
    pass


class DecompositionValidator:
    """
    Harness-owned: Validates the structural integrity of a task decomposition.
    Purely deterministic — no LLM involvement.

    Checks:
    1. Every task has a non-empty definition_of_done.
    2. All dependency references point to existing task IDs.
    3. No forward dependencies (task_3 cannot depend on task_5).
    4. No dependency cycles (Kahn's topological sort).
    5. Max task count enforcement.
    """

    def __init__(self, max_tasks: int = 7):
        self.max_tasks = max_tasks

    def validate(self, tasks: List[Task]) -> List[str]:
        """
        Validates the task list and returns a list of error strings.
        An empty list means the decomposition is valid.
        """
        errors: List[str] = []

        if not tasks:
            errors.append("Decomposition produced zero tasks.")
            return errors

        if len(tasks) > self.max_tasks:
            errors.append(
                f"Too many tasks: {len(tasks)} (max {self.max_tasks}). "
                "Break the goal into phases or reduce granularity."
            )

        task_ids: List[str] = [t.id for t in tasks]
        task_id_set: Set[str] = set(task_ids)

        # Check for duplicate IDs
        if len(task_ids) != len(task_id_set):
            errors.append("Duplicate task IDs detected.")

        task_index: Dict[str, int] = {tid: i for i, tid in enumerate(task_ids)}

        for task in tasks:
            # 1. Every task must have a definition_of_done
            if not task.definition_of_done:
                errors.append(f"{task.id}: Missing definition_of_done (no acceptance criteria).")

            # 2. Dependencies must reference existing task IDs
            for dep in task.dependencies:
                if dep not in task_id_set:
                    errors.append(f"{task.id}: References unknown dependency '{dep}'.")

            # 3. No forward dependencies
            for dep in task.dependencies:
                if dep in task_index and task.id in task_index:
                    if task_index[dep] >= task_index[task.id]:
                        errors.append(
                            f"{task.id}: Forward dependency on '{dep}' "
                            f"(task at index {task_index[task.id]} depends on task at index {task_index[dep]})."
                        )

        # 4. Cycle detection
        if self._has_cycle(tasks, task_id_set):
            errors.append("Task graph contains a dependency cycle.")

        return errors

    def _has_cycle(self, tasks: List[Task], task_id_set: Set[str]) -> bool:
        """
        Detects cycles using Kahn's algorithm (topological sort).
        Returns True if a cycle exists.
        """
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
        adj: Dict[str, List[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in task_id_set:
                    adj[dep].append(task.id)
                    in_degree[task.id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return visited != len(tasks)
