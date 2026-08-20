import json
import time
from typing import List, Dict, Optional, Any
from harness.state import Goal, Task, TaskStatus, EvaluationVerdict
from llm.client import LLMClient
from llm.prompts import MANAGER_DECOMPOSITION_PROMPT, MANAGER_RETRY_PROMPT


class ManagerAgent:
    """
    Manages task decomposition, dependencies, scheduling, and feedback integration.
    Supports dual-mode: LLM-backed (autonomous) and deterministic (test/offline).
    """

    def __init__(self, name: str = "Manager", llm_client: Optional[LLMClient] = None):
        self.name = name
        self.llm_client = llm_client

    def decompose_goal(
        self,
        goal: Goal,
        feedback: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Decomposes a goal into a validated Task DAG.
        Uses LLM if configured; otherwise falls back to deterministic rule-based decomposition.
        """
        if self.llm_client:
            return self.decompose_goal_with_llm(goal, feedback=feedback)
        return self._decompose_deterministic(goal)

    def decompose_goal_with_llm(
        self,
        goal: Goal,
        feedback: Optional[List[str]] = None,
    ) -> List[Task]:
        """
        Calls the LLM with structured decomposition prompt and parses the JSON response into Tasks.
        """
        if not self.llm_client:
            raise ValueError("LLM client required for decompose_goal_with_llm.")

        if feedback:
            user_content = MANAGER_RETRY_PROMPT.format(
                errors="\n".join(f"- {e}" for e in feedback),
                objective=goal.objective,
            )
        else:
            hints = "\n".join(f"- {d}" for d in goal.definition_of_done) if goal.definition_of_done else "None provided."
            user_content = f"Objective: {goal.objective}\nPreliminary criteria hints:\n{hints}"

        messages = [
            {"role": "system", "content": MANAGER_DECOMPOSITION_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.llm_client.chat(messages=messages, temperature=0.2)
        content = response.content or ""
        
        # Clean any accidental markdown code fencing from output
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            task_list = data.get("tasks", [])
            tasks: List[Task] = []
            for item in task_list:
                task = Task(
                    id=item["id"],
                    title=item["title"],
                    description=item.get("description", ""),
                    definition_of_done=item.get("definition_of_done", []),
                    dependencies=item.get("dependencies", []),
                    status=TaskStatus.PENDING,
                )
                tasks.append(task)
            return tasks
        except (json.JSONDecodeError, KeyError) as e:
            # If JSON decoding fails, return empty list (validator will trigger retry)
            return []

    def _decompose_deterministic(self, goal: Goal) -> List[Task]:
        """Deterministic rule-based decomposition for offline runs and testing."""
        tasks: List[Task] = []
        dods = goal.definition_of_done or [goal.objective]
        for i, dod_item in enumerate(dods):
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
