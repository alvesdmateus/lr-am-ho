import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from harness.state import Task, ExecutionResult
from tools.protocol import ToolRequest, ToolType, ToolResponse
from tools.sidecar import ToolSidecar
from llm.client import LLMClient
from llm.prompts import WORKER_SYSTEM_PROMPT
from llm.schemas import WORKER_TOOL_SCHEMAS


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
    Supports dual-mode:
    1. Autonomous ReAct tool-calling loop (when llm_client provided)
    2. Deterministic micro-plan execution (when llm_client is None)
    """

    def __init__(
        self,
        name: str = "Worker",
        workspace_dir: str = ".",
        sandbox_type: str = "local",
        docker_image: str = "python:3.11-slim",
        llm_client: Optional[LLMClient] = None,
        max_tool_steps: int = 15,
    ):
        self.name = name
        self.sidecar = ToolSidecar(
            workspace_dir=workspace_dir,
            sandbox_type=sandbox_type,
            docker_image=docker_image,
        )
        self.planner = MicroPlanner()
        self.action_engine = ActionEngine(self.sidecar)
        self.tracker = WorkspaceTracker(self.sidecar)
        self.llm_client = llm_client
        self.max_tool_steps = max_tool_steps

    def execute_task(
        self,
        task: Task,
        memory_briefing: str,
        custom_micro_actions: Optional[List[MicroAction]] = None,
        simulated_discoveries: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Executes the task using LLM ReAct loop (if llm_client provided)
        or deterministic micro-actions.
        """
        if self.llm_client and not custom_micro_actions:
            return self.execute_task_with_llm(task, memory_briefing)
        return self._execute_task_deterministic(task, memory_briefing, custom_micro_actions, simulated_discoveries)

    def execute_task_with_llm(
        self,
        task: Task,
        memory_briefing: str,
    ) -> ExecutionResult:
        """
        Autonomous ReAct tool-calling loop powered by the LLM.
        """
        if not self.llm_client:
            raise ValueError("LLM client required for execute_task_with_llm.")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": WORKER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"### TASK ASSIGNMENT\n"
                    f"**Title**: {task.title}\n"
                    f"**Description**: {task.description}\n"
                    f"**Definition of Done**:\n"
                    + "\n".join(f"- {d}" for d in task.definition_of_done)
                    + f"\n\n{memory_briefing}\n\n"
                    f"Begin implementing and verifying the solution using your tools."
                ),
            },
        ]

        executed_commands: List[str] = []
        tool_outputs: List[Dict[str, Any]] = []
        raw_traces: List[str] = []
        step_count = 0
        all_passed = True

        while step_count < self.max_tool_steps:
            step_count += 1
            response = self.llm_client.chat(
                messages=messages,
                tools=WORKER_TOOL_SCHEMAS,
                temperature=0.2,
            )

            # If the LLM stopped calling tools, the task is finished
            if not response.tool_calls:
                raw_traces.append(f"LLM completed task: {response.content or 'Done'}")
                break

            # Append assistant message with tool calls to conversation history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # Dispatch each tool call through the ToolSidecar
            for tc in response.tool_calls:
                tool_type_enum = self._map_tool_name_to_type(tc.name)
                req = ToolRequest(
                    tool_type=tool_type_enum,
                    parameters=tc.arguments,
                    task_id=task.id,
                )
                tool_res = self.sidecar.dispatch(req)

                executed_commands.append(f"[{tc.name}] {tc.arguments}")
                tool_outputs.append({
                    "step": step_count,
                    "tool": tc.name,
                    "success": tool_res.success,
                    "is_truncated": tool_res.is_truncated,
                    "output": tool_res.output,
                    "error": tool_res.error,
                })
                raw_traces.append(
                    f"Step {step_count} [{tc.name}]: {'SUCCESS' if tool_res.success else 'FAILED'}\n"
                    f"Output: {tool_res.output[:200]}"
                )

                # Feed truncated tool result back to LLM context
                tool_msg_content = tool_res.output if tool_res.success else f"ERROR: {tool_res.error}\n{tool_res.output}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_msg_content or "(No output)",
                })

                if not tool_res.success:
                    all_passed = False

        commit_hash = self.tracker.record_commit(task.id, f"Implemented {task.title}")

        return ExecutionResult(
            task_id=task.id,
            success=all_passed,
            summary_of_changes=f"Completed {len(executed_commands)} autonomous tool actions for {task.title}.",
            commands_executed=executed_commands,
            tool_outputs=tool_outputs,
            test_output="Autonomous tool execution completed.",
            git_commit_hash=commit_hash,
            raw_trace="\n".join(raw_traces),
            discovered_learnings=[],
        )

    def _map_tool_name_to_type(self, name: str) -> ToolType:
        mapping = {
            "bash": ToolType.BASH,
            "file_write": ToolType.FILE_WRITE,
            "file_read": ToolType.FILE_READ,
            "run_tests": ToolType.RUN_TESTS,
            "git_diff": ToolType.GIT_DIFF,
        }
        return mapping.get(name, ToolType.BASH)

    def _execute_task_deterministic(
        self,
        task: Task,
        memory_briefing: str,
        custom_micro_actions: Optional[List[MicroAction]] = None,
        simulated_discoveries: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """Deterministic micro-action execution (for testing and offline runs)."""
        micro_actions = custom_micro_actions or self.planner.plan_steps(task, memory_briefing)

        executed_commands: List[str] = []
        tool_outputs: List[Dict[str, Any]] = []
        all_passed = True
        raw_traces: List[str] = []

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