import os
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from tools.protocol import ToolResponse


class BaseSandbox(ABC):
    """Abstract Base Class for Task Execution Sandboxes."""

    @abstractmethod
    def execute_command(self, command: str, timeout: int = 30) -> ToolResponse:
        """Executes a command inside the sandbox environment."""
        pass

    @abstractmethod
    def write_file(self, file_path: str, content: str) -> ToolResponse:
        """Writes a file to the sandbox workspace."""
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> ToolResponse:
        """Reads a file from the sandbox workspace."""
        pass

    @abstractmethod
    def get_git_diff(self) -> ToolResponse:
        """Returns the current git diff in the sandbox workspace."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Cleans up or destroys the sandbox state."""
        pass


class LocalSandbox(BaseSandbox):
    """
    Local Directory Sandbox.
    Runs commands on host system scoped to a specific workspace directory.
    Fast, lightweight, ideal for quick iterations and unit tests.
    """

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

    def execute_command(self, command: str, timeout: int = 30) -> ToolResponse:
        if not command:
            return ToolResponse(success=False, output="", error="No command provided.")
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined_output = res.stdout + (("\n" + res.stderr) if res.stderr else "")
            return ToolResponse(
                success=(res.returncode == 0),
                output=combined_output.strip(),
                error=res.stderr.strip() if res.returncode != 0 else None,
                metadata={"return_code": res.returncode, "sandbox_type": "local"},
            )
        except subprocess.TimeoutExpired:
            return ToolResponse(
                success=False,
                output="",
                error=f"Command execution timed out after {timeout}s.",
                metadata={"sandbox_type": "local"},
            )

    def write_file(self, file_path: str, content: str) -> ToolResponse:
        if not file_path:
            return ToolResponse(success=False, output="", error="File path required.")
        target = os.path.join(self.workspace_dir, file_path) if not os.path.isabs(file_path) else file_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(
            success=True,
            output=f"Successfully wrote {len(content)} bytes to {file_path}",
            metadata={"bytes_written": len(content), "sandbox_type": "local"},
        )

    def read_file(self, file_path: str) -> ToolResponse:
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
            metadata={"bytes_read": len(content), "sandbox_type": "local"},
        )

    def get_git_diff(self) -> ToolResponse:
        return self.execute_command("git diff")

    def reset(self) -> None:
        """No-op for local directory; directory can be purged by caller."""
        pass


class DockerSandbox(BaseSandbox):
    """
    Docker Container Sandbox.
    Spawns an ephemeral Docker container mounting the workspace directory.
    Provides complete process, network, and filesystem quarantine.
    """

    def __init__(
        self,
        workspace_dir: str = ".",
        image: str = "python:3.11-slim",
        container_name_prefix: str = "harness_worker",
        network_disabled: bool = False,
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.image = image
        self.container_name = f"{container_name_prefix}_{uuid.uuid4().hex[:8]}"
        self.network_disabled = network_disabled
        self.container_id: Optional[str] = None
        self._start_container()

    def _start_container(self) -> None:
        """Starts an idle container with the workspace volume mounted."""
        # Convert windows backslashes to forward slashes for Docker volume mounting
        mount_path = self.workspace_dir.replace("\\", "/")
        net_flag = "--network none" if self.network_disabled else ""
        
        cmd = (
            f"docker run -d --name {self.container_name} "
            f"-v \"{mount_path}:/workspace\" "
            f"-w /workspace {net_flag} "
            f"{self.image} tail -f /dev/null"
        )
        try:
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                self.container_id = res.stdout.strip()
            else:
                self.container_id = None
        except Exception:
            self.container_id = None

    def execute_command(self, command: str, timeout: int = 30) -> ToolResponse:
        if not self.container_id:
            return ToolResponse(
                success=False,
                output="",
                error="Docker container is not running.",
                metadata={"sandbox_type": "docker"},
            )
        # Escape double quotes in command for docker exec
        escaped_cmd = command.replace('"', '\\"')
        exec_cmd = f'docker exec {self.container_name} bash -c "{escaped_cmd}"'
        try:
            res = subprocess.run(
                exec_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined = res.stdout + (("\n" + res.stderr) if res.stderr else "")
            return ToolResponse(
                success=(res.returncode == 0),
                output=combined.strip(),
                error=res.stderr.strip() if res.returncode != 0 else None,
                metadata={"return_code": res.returncode, "sandbox_type": "docker"},
            )
        except subprocess.TimeoutExpired:
            return ToolResponse(
                success=False,
                output="",
                error=f"Command timed out in container after {timeout}s.",
                metadata={"sandbox_type": "docker"},
            )

    def write_file(self, file_path: str, content: str) -> ToolResponse:
        # Since workspace is volume-mounted, local file write automatically reflects in container
        if not file_path:
            return ToolResponse(success=False, output="", error="File path required.")
        target = os.path.join(self.workspace_dir, file_path) if not os.path.isabs(file_path) else file_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(
            success=True,
            output=f"Successfully wrote {len(content)} bytes to {file_path} (Docker mounted)",
            metadata={"bytes_written": len(content), "sandbox_type": "docker"},
        )

    def read_file(self, file_path: str) -> ToolResponse:
        target = os.path.join(self.workspace_dir, file_path) if not os.path.isabs(file_path) else file_path
        if not os.path.exists(target):
            return ToolResponse(success=False, output="", error=f"File not found: {file_path}")
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResponse(
            success=True,
            output=content,
            metadata={"bytes_read": len(content), "sandbox_type": "docker"},
        )

    def get_git_diff(self) -> ToolResponse:
        return self.execute_command("git diff")

    def reset(self) -> None:
        """Stops and removes the ephemeral Docker container."""
        if self.container_name:
            subprocess.run(
                f"docker rm -f {self.container_name}",
                shell=True,
                capture_output=True,
                text=True,
            )
            self.container_id = None
