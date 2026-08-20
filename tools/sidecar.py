import os
import subprocess
import time
from typing import Dict, Any, Optional, List
from tools.protocol import ToolRequest, ToolResponse, ToolType, SafetyPolicy


class ToolSidecar:
    """
    Tool Proxy & Safety Sidecar.
    Intercepts tool calls, enforces safety policies, automatically truncates large outputs,
    and captures workspace diffs.
    """
    def __init__(
        self,
        workspace_dir: str = ".",
        policy: Optional[SafetyPolicy] = None,
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.policy = policy or SafetyPolicy()
        self.invocation_history: List[Dict[str, Any]] = []

    def dispatch(self, request: ToolRequest) -> ToolResponse:
        """Main entry point for all tool invocations from the Worker."""
        # 1. Safety verification
        safety_error = self._validate_safety(request)
        if safety_error:
            return ToolResponse(
                success=False,
                output="",
                error=f"Security Policy Violation: {safety_error}",
            )

        # 2. Dispatch by tool type
        start_time = time.time()
        try:
            if request.tool_type == ToolType.BASH:
                response = self._handle_bash(request.parameters.get("command", ""))
            elif request.tool_type == ToolType.FILE_WRITE:
                response = self._handle_file_write(
                    request.parameters.get("file_path", ""),
                    request.parameters.get("content", ""),
                )
            elif request.tool_type == ToolType.FILE_READ:
                response = self._handle_file_read(request.parameters.get("file_path", ""))
            elif request.tool_type == ToolType.RUN_TESTS:
                response = self._handle_run_tests(request.parameters.get("test_cmd", ""))
            elif request.tool_type == ToolType.GIT_DIFF:
                response = self._handle_git_diff()
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

    # -------------------------------------------------------------------------
    # Handlers
    # -------------------------------------------------------------------------
    def _handle_bash(self, command: str) -> ToolResponse:
        if not command:
            return ToolResponse(success=False, output="", error="No command provided.")
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            combined_output = res.stdout + (("\n" + res.stderr) if res.stderr else "")
            return ToolResponse(
                success=(res.returncode == 0),
                output=combined_output.strip(),
                error=res.stderr.strip() if res.returncode != 0 else None,
                metadata={"return_code": res.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResponse(
                success=False,
                output="",
                error="Command execution timed out after 30s.",
            )

    def _handle_file_write(self, file_path: str, content: str) -> ToolResponse:
        if not file_path:
            return ToolResponse(success=False, output="", error="File path required.")
        target = os.path.join(self.workspace_dir, file_path) if not os.path.isabs(file_path) else file_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(
            success=True,
            output=f"Successfully wrote {len(content)} bytes to {file_path}",
            metadata={"bytes_written": len(content)},
        )

    def _handle_file_read(self, file_path: str) -> ToolResponse:
        if not file_path:
            return ToolResponse(success=False, output="", error="File path required.")
        target = os.path.join(self.workspace_dir, file_path) if not os.path.isabs(file_path) else file_path
        if not os.path.exists(target):
            return ToolResponse(success=False, output="", error=f"File not found: {file_path}")
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResponse(
            success=True,
            output=content,
            metadata={"bytes_read": len(content)},
        )

    def _handle_run_tests(self, test_cmd: str) -> ToolResponse:
        cmd = test_cmd or "python -m unittest discover -s tests"
        return self._handle_bash(cmd)

    def _handle_git_diff(self) -> ToolResponse:
        return self._handle_bash("git diff")
