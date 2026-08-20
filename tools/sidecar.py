import os
import time
from typing import Dict, Any, Optional, List, Union
from tools.protocol import ToolRequest, ToolResponse, ToolType, SafetyPolicy
from tools.sandbox import BaseSandbox, LocalSandbox, DockerSandbox


class ToolSidecar:
    """
    Tool Proxy & Safety Sidecar.
    Intercepts tool calls, enforces safety policies, automatically truncates large outputs,
    and delegates execution to a pluggable Sandbox backend (Local or Docker).
    """

    def __init__(
        self,
        workspace_dir: str = ".",
        policy: Optional[SafetyPolicy] = None,
        sandbox: Optional[BaseSandbox] = None,
        sandbox_type: str = "local",
        docker_image: str = "python:3.11-slim",
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.policy = policy or SafetyPolicy()
        self.invocation_history: List[Dict[str, Any]] = []

        if sandbox:
            self.sandbox = sandbox
        elif sandbox_type == "docker":
            self.sandbox = DockerSandbox(workspace_dir=self.workspace_dir, image=docker_image)
        else:
            self.sandbox = LocalSandbox(workspace_dir=self.workspace_dir)

    def dispatch(self, request: ToolRequest) -> ToolResponse:
        """Main entry point for all tool invocations from the Worker."""
        # 1. Safety verification
        safety_error = self._validate_safety(request)
        if safety_error:
            return ToolResponse(
                success=False,
                output="",
                error=f"Security Policy Violation: {safety_error}",
                metadata={"blocked_by_sidecar": True},
            )

        # 2. Dispatch to Sandbox Backend
        start_time = time.time()
        try:
            if request.tool_type == ToolType.BASH:
                response = self.sandbox.execute_command(request.parameters.get("command", ""))
            elif request.tool_type == ToolType.FILE_WRITE:
                response = self.sandbox.write_file(
                    request.parameters.get("file_path", ""),
                    request.parameters.get("content", ""),
                )
            elif request.tool_type == ToolType.FILE_READ:
                response = self.sandbox.read_file(request.parameters.get("file_path", ""))
            elif request.tool_type == ToolType.RUN_TESTS:
                cmd = request.parameters.get("test_cmd", "") or "python -m unittest discover -s tests"
                response = self.sandbox.execute_command(cmd)
            elif request.tool_type == ToolType.GIT_DIFF:
                response = self.sandbox.get_git_diff()
            else:
                response = ToolResponse(
                    success=False,
                    output="",
                    error=f"Unsupported tool type: {request.tool_type}",
                )
        except Exception as e:
            response = ToolResponse(
                success=False,
                output="",
                error=f"Sidecar execution exception: {str(e)}",
            )

        duration = time.time() - start_time
        response.metadata["duration_seconds"] = round(duration, 4)
        response.metadata["tool_type"] = request.tool_type.value

        # 3. Output truncation filter (protect context window)
        response = self._apply_output_truncation(response)

        # 4. Log invocation
        self.invocation_history.append({
            "timestamp": time.time(),
            "request": request.model_dump(),
            "success": response.success,
            "is_truncated": response.is_truncated,
        })

        return response

    # -------------------------------------------------------------------------
    # Safety Guards
    # -------------------------------------------------------------------------
    def _validate_safety(self, request: ToolRequest) -> Optional[str]:
        if request.tool_type in (ToolType.BASH, ToolType.RUN_TESTS):
            cmd = request.parameters.get("command", "") or request.parameters.get("test_cmd", "")
            for blocked in self.policy.blocked_commands:
                if blocked in cmd:
                    return f"Command '{cmd}' contains forbidden sequence '{blocked}'."
        return None

    # -------------------------------------------------------------------------
    # Truncation Engine
    # -------------------------------------------------------------------------
    def _apply_output_truncation(self, response: ToolResponse) -> ToolResponse:
        if not response.output:
            return response

        lines = response.output.splitlines()
        if len(lines) > self.policy.max_output_lines:
            head = lines[: self.policy.head_lines]
            tail = lines[-self.policy.tail_lines :]
            omitted = len(lines) - (len(head) + len(tail))
            truncated_text = (
                "\n".join(head)
                + f"\n\n... [TRUNCATED: {omitted} lines omitted by Tool Sidecar] ...\n\n"
                + "\n".join(tail)
            )
            response.output = truncated_text
            response.is_truncated = True
            response.metadata["original_line_count"] = len(lines)

        return response

    def close(self) -> None:
        """Cleans up the sandbox backend."""
        if self.sandbox:
            self.sandbox.reset()
