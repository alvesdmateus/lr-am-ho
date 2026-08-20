from typing import List, Dict, Optional
from harness.state import Task, ExecutionResult, EvaluationVerdict


class EvaluatorAgent:
    """
    Independent arbiter of quality and acceptance criteria.
    Evaluates worker outputs objectively and verifies memory items used.
    """
    def __init__(self, name: str = "Evaluator"):
        self.name = name

    def evaluate_task(
        self,
        task: Task,
        execution_result: ExecutionResult,
        override_pass: Optional[bool] = None,
        override_feedback: Optional[str] = None,
    ) -> EvaluationVerdict:
        """
        Audits execution against Task Definition of Done.
        Validates or invalidates memory items used by the worker.
        """
        passed = override_pass if override_pass is not None else execution_result.success
        rubric_results: Dict[str, bool] = {}

        for dod_item in task.definition_of_done:
            rubric_results[dod_item] = passed

        if passed:
            score = 1.0
            feedback = override_feedback or "All criteria met and verified by test suite."
            validated_memories = list(task.used_memory_ids)
            invalidated_memories = []
        else:
            score = 0.2
            feedback = override_feedback or "Verification failed: Definition of Done not satisfied."
            validated_memories = []
            invalidated_memories = list(task.used_memory_ids)

        return EvaluationVerdict(
            task_id=task.id,
            passed=passed,
            score=score,
            rubric_results=rubric_results,
            feedback=feedback,
            validated_memory_ids=validated_memories,
            invalidated_memory_ids=invalidated_memories,
            suggested_actions=["Review failure traces and update implementation strategy"] if not passed else [],
        )
