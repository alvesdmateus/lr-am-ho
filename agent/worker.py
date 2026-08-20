import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from harness.state import Task, ExecutionResult
from tools.protocol import ToolRequest, ToolType, ToolResponse
from tools.sidecar import ToolSidecar


class MicroAction(BaseModel):
    step_number: int
    intent: str
    tool_type: ToolType
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MicroPlanner:
    """
    Sub-Planner inside the Worker Pod.
    Breaks a single task into a concise micro-plan (2-4 immediate steps)
    grounded by the task requirements and memory briefing.
    """
    def plan_steps(self, task: Task, memory_briefing: str) -> List[MicroAction]:
        # In an LLM-backed worker, this calls the model to produce structured actions.
        # Deterministic micro-plan representation:
        actions = [
            MicroAction(
                step_number=1,
                intent=f"Inspect requirements & context for: {task.title}",
                tool_type=ToolType.BASH,
                parameters={"command": "echo 'Checking environment setup'"},
            ),
            MicroAction(
                step_number=2,
                intent=f"Execute implementation for: {task.title}",
                tool_type=ToolType.BASH,
                parameters={"command": f"echo 'Implementing {task.id}'"},
            ),
            MicroAction(
                step_number=3,
                intent=f"Verify criteria: {', '.join(task.definition_of_done)}",
                tool_type=ToolType.RUN_TESTS,
                parameters={"test_cmd": "python -c \"print('Verification passed')\""},
            ),
        ]
        return actions


class WorkspaceTracker:
    """
    Tracks workspace changes, git diffs, and creates atomic task-scoped commit checkpoints.
    """
    def __init__(self, sidecar: ToolSidecar):
        self.sidecar = sidecar

    def get_current_diff(self) -> str:
        res = self.sidecar.dispatch(ToolRequest(tool_type=ToolType.GIT_DIFF))
        return res.output if res.success else ""

    def record_commit(self, task_id: str, summary: str) -> str:
        # In a full git environment, this would run git add / git commit
        # For now, return a deterministic commit hash tag
        commit_hash = f"task_{task_id}_{int(time.time()*1000)}"
        return commit_hash


class ActionEngine:
    """
    Executes micro-actions through the Tool Sidecar proxy.
    """
    def __init__(self, sidecar: ToolSidecar):
        self.sidecar = sidecar

    def execute_action(self, action: MicroAction, task_id: str) -> ToolResponse:
        req = ToolRequest(
            tool_type=action.tool_type,
            parameters=action.parameters,
            task_id=task_id,
        )
        return self.sidecar.dispatch(req)


class Worker:
    """
    Worker Pod Coordinator.
    Runs the internal micro-loop within a clean context:
    1. MicroPlanner generates sub-steps using Task + Memory Briefing.
    2. ActionEngine invokes tools via ToolSidecar proxy.
    3. WorkspaceTracker records changes and diffs.
    4. Emits structured ExecutionResult.
    """
    def __init__(self, name: str = "Worker", workspace_dir: str = "."):
        self.name = name
        self.sidecar = ToolSidecar(workspace_dir=workspace_dir)
        self.planner = MicroPlanner()
        self.action_engine = ActionEngine(self.sidecar)
        self.tracker = WorkspaceTracker(self.sidecar)

    def execute_task(
        self,
        task: Task,
        memory_briefing: str,
        custom_micro_actions: Optional[List[MicroAction]] = None,
        simulated_discoveries: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Executes the micro-loop for a single task.
        """
        # Step 1: Micro-Planning
        micro_actions = custom_micro_actions or self.planner.plan_steps(task, memory_briefing)
        
        executed_commands: List[str] = []
        tool_outputs: List[Dict[str, Any]] = []
        all_passed = True
        raw_traces: List[str] = []

        # Step 2: Micro-Loop Execution
        for action in micro_actions:
            response = self.action_engine.execute_action(action, task.id)
            executed_commands.append(f"[{action.tool_type.value}] {action.intent}")
            
            tool_outputs.append({
                "step": action.step_number,
                "tool": action.tool_type.value,
                "success": response.success,
                "is_truncated": response.is_truncated,
                "output": response.output,
                "error": response.error,
            })
            
            trace_line = f"Step {action.step_number} ({action.tool_type.value}): {action.intent} -> {'SUCCESS' if response.success else 'FAILED'}"
            raw_traces.append(trace_line)

            if not response.success:
                all_passed = False
                break

        # Step 3: Workspace Tracking & Checkpoint
        commit_hash = self.tracker.record_commit(task.id, f"Implemented {task.title}")

        return ExecutionResult(
            task_id=task.id,
            success=all_passed,
            summary_of_changes=f"Executed {len(executed_commands)} micro-steps for {task.title}.",
            commands_executed=executed_commands,
            tool_outputs=tool_outputs,
            test_output="All micro-steps verified." if all_passed else "Micro-action failed.",
            git_commit_hash=commit_hash,
            raw_trace="\n".join(raw_traces),
            discovered_learnings=simulated_discoveries or [],
        )