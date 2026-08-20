import time
from typing import Optional, Dict, Any
from harness.state import Goal, Task, TaskStatus, HarnessState, ExecutionResult, EvaluationVerdict
from storage.state_store import StateStore
from agent.manager import ManagerAgent
from agent.worker import Worker
from agent.evaluator import EvaluatorAgent
from agent.memory import TieredMemoryManager


class HarnessEngine:
    """
    Long-Running Harness Engine.
    Owns the master loop, enforces Definition of Done, manages crash-resilient checkpoints,
    and coordinates Manager, Worker, Evaluator, and Tiered Memory.
    """
    def __init__(
        self,
        goal: Goal,
        storage_dir: str = ".harness",
        max_iterations: int = 50,
        max_consecutive_failures: int = 5,
    ):
        self.state_store = StateStore(storage_dir=f"{storage_dir}/state")
        self.memory_agent = TieredMemoryManager(storage_dir=f"{storage_dir}/storage")
        self.manager = ManagerAgent()
        self.worker = Worker()
        self.evaluator = EvaluatorAgent()

        # Initialize or restore state
        saved_state = self.state_store.load_state()
        if saved_state:
            self.state = HarnessState(**saved_state)
        else:
            self.state = HarnessState(
                goal=goal,
                max_iterations=max_iterations,
                max_consecutive_failures=max_consecutive_failures,
            )

    def run_step(self) -> bool:
        """
        Executes a single cycle of the harness loop.
        Returns True if loop should continue, False if completed or halted.
        """
        # Guard 1: Check if already completed
        if self.state.goal.is_completed:
            return False

        # Guard 2: Iteration limits & runaway protection
        if self.state.iteration_count >= self.state.max_iterations:
            self.state.is_halted = True
            self.state.halt_reason = "Max loop iterations reached."
            self._save_checkpoint("max_iterations_halt")
            return False

        if self.state.consecutive_failures >= self.state.max_consecutive_failures:
            self.state.is_halted = True
            self.state.halt_reason = "Exceeded maximum consecutive failure threshold."
            self._save_checkpoint("failure_halt")
            return False

        self.state.iteration_count += 1
        self.state_store.log_event("step_start", {"iteration": self.state.iteration_count})

        # Phase 1: Goal Decomposition (if no tasks exist yet)
        if not self.state.tasks:
            tasks = self.manager.decompose_goal(self.state.goal)
            self.state.tasks = {t.id: t for t in tasks}
            self.state_store.log_event("goal_decomposed", {"task_count": len(tasks)})

        # Phase 2: Task Selection
        ready_task = self.manager.select_next_task(self.state.tasks)
        if not ready_task:
            if self.state.all_tasks_completed():
                self.state.goal.is_completed = True
                self.state.goal.completed_at = time.time()
                self.state_store.log_event("goal_completed", {"goal_id": self.state.goal.id})
                self._save_checkpoint("completed")
                return False
            else:
                # Deadlock or blocked tasks
                self.state.is_halted = True
                self.state.halt_reason = "No ready tasks available and goal is not complete."
                self._save_checkpoint("deadlock_halt")
                return False

        ready_task.mark_in_progress()

        # Phase 3: Epistemic Memory Briefing (Principles 1, 4 & 7)
        memory_briefing = self.memory_agent.format_prompt_briefing(ready_task)

        # Phase 4: Worker Execution in Clean Context
        execution_result = self.worker.execute_task(
            task=ready_task,
            memory_briefing=memory_briefing,
        )

        # Phase 5: Independent Evaluation
        verdict = self.evaluator.evaluate_task(
            task=ready_task,
            execution_result=execution_result,
        )

        # Phase 6: Active Memory Consolidation & Feedback Loop (Principles 2, 5 & 6)
        self.memory_agent.consolidate_episode(
            task=ready_task,
            execution_result=execution_result,
            verdict=verdict,
        )

        # Phase 7: Manager State Transition
        self.manager.handle_task_evaluation(ready_task, verdict)

        if verdict.passed:
            self.state.consecutive_failures = 0
        else:
            self.state.consecutive_failures += 1

        # Phase 8: Persistence & Checkpoint
        self._save_checkpoint(f"iter_{self.state.iteration_count}")
        return True

    def run_until_completion(self) -> HarnessState:
        """Runs the loop until the Goal Definition of Done is fully satisfied or halted."""
        while self.run_step():
            pass
        return self.state

    def _save_checkpoint(self, tag: str) -> None:
        state_dict = self.state.model_dump()
        self.state_store.save_state(state_dict)
        self.state_store.create_checkpoint(tag, state_dict)
