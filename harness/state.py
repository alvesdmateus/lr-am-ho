import time
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    id: str
    title: str
    description: str = ""
    definition_of_done: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    feedback_history: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    used_memory_ids: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS
        self.attempt_count += 1
        self.updated_at = time.time()

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.updated_at = time.time()

    def mark_failed(self, feedback: str) -> None:
        self.feedback_history.append(feedback)
        self.updated_at = time.time()
        if self.attempt_count >= self.max_attempts:
            self.status = TaskStatus.FAILED
        else:
            self.status = TaskStatus.PENDING  # Reset for retry with feedback


class Goal(BaseModel):
    id: str
    objective: str
    definition_of_done: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    is_completed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    task_id: str
    success: bool
    summary_of_changes: str
    commands_executed: List[str] = Field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    test_output: Optional[str] = None
    git_commit_hash: Optional[str] = None
    raw_trace: str = ""
    discovered_learnings: List[str] = Field(default_factory=list)


class EvaluationVerdict(BaseModel):
    task_id: str
    passed: bool
    score: float = 0.0  # 0.0 to 1.0
    rubric_results: Dict[str, bool] = Field(default_factory=dict)
    feedback: str = ""
    validated_memory_ids: List[str] = Field(default_factory=list)
    invalidated_memory_ids: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)


class HarnessState(BaseModel):
    goal: Goal
    tasks: Dict[str, Task] = Field(default_factory=dict)
    iteration_count: int = 0
    consecutive_failures: int = 0
    max_iterations: int = 50
    max_consecutive_failures: int = 5
    is_halted: bool = False
    halt_reason: Optional[str] = None

    def get_pending_ready_tasks(self) -> List[Task]:
        """Returns pending tasks whose dependencies are all COMPLETED."""
        completed_ids = {t_id for t_id, t in self.tasks.items() if t.status == TaskStatus.COMPLETED}
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
        return ready

    def all_tasks_completed(self) -> bool:
        if not self.tasks:
            return False
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())
