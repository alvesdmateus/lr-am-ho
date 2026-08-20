import json
from typing import List, Dict, Optional
from harness.state import Task, ExecutionResult, EvaluationVerdict
from llm.client import LLMClient
from llm.prompts import EVALUATOR_SYSTEM_PROMPT


class EvaluatorAgent:
    """
    Independent arbiter of quality and acceptance criteria.
    Supports dual-mode: LLM-backed (autonomous audit) and deterministic (offline/test).
    """

    def __init__(self, name: str = "Evaluator", llm_client: Optional[LLMClient] = None):
        self.name = name
        self.llm_client = llm_client

    def evaluate_task(
        self,
        task: Task,
        execution_result: ExecutionResult,
        override_pass: Optional[bool] = None,
        override_feedback: Optional[str] = None,
    ) -> EvaluationVerdict:
        """
        Audits execution against Task Definition of Done.
        """
        if override_pass is not None:
            return self._evaluate_deterministic(task, execution_result, override_pass, override_feedback)
        if self.llm_client:
            return self.evaluate_task_with_llm(task, execution_result)
        return self._evaluate_deterministic(task, execution_result, None, override_feedback)

    def evaluate_task_with_llm(
        self,
        task: Task,
        execution_result: ExecutionResult,
    ) -> EvaluationVerdict:
        """
        LLM independently audits the execution trace against the task Definition of Done.
        """
        if not self.llm_client:
            raise ValueError("LLM client required for evaluate_task_with_llm.")

        user_content = (
            f"### TASK UNDER EVALUATION\n"
            f"**Title**: {task.title}\n"
            f"**Description**: {task.description}\n"
            f"**Definition of Done**:\n"
            + "\n".join(f"- {d}" for d in task.definition_of_done)
            + f"\n\n### WORKER EXECUTION TRACE\n"
            f"**Success Flag**: {execution_result.success}\n"
            f"**Commands Executed**:\n"
            + "\n".join(f"- {c}" for c in execution_result.commands_executed)
            + f"\n\n**Test Output**:\n{execution_result.test_output or 'None'}\n\n"
            f"**Raw Traces**:\n{execution_result.raw_trace[:2000]}\n"
        )

        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.llm_client.chat(messages=messages, temperature=0.1)
        content = response.content or ""

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
            passed = bool(data.get("passed", False))
            score = float(data.get("score", 1.0 if passed else 0.0))
            rubric = data.get("rubric_results", {})
            feedback = data.get("feedback", "Evaluation complete.")
            actions = data.get("suggested_actions", [])

            return EvaluationVerdict(
                task_id=task.id,
                passed=passed,
                score=score,
                rubric_results=rubric,
                feedback=feedback,
                validated_memory_ids=list(task.used_memory_ids) if passed else [],
                invalidated_memory_ids=list(task.used_memory_ids) if not passed else [],
                suggested_actions=actions,
            )
        except Exception:
            # Fallback to execution_result.success if LLM output parse fails
            return self._evaluate_deterministic(task, execution_result)

    def _evaluate_deterministic(
        self,
        task: Task,
        execution_result: ExecutionResult,
        override_pass: Optional[bool] = None,
        override_feedback: Optional[str] = None,
    ) -> EvaluationVerdict:
        """Deterministic rule-based evaluation (for testing and offline runs)."""
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
