from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ToolType(str, Enum):
    BASH = "bash"
    FILE_WRITE = "file_write"
    FILE_READ = "file_read"
    RUN_TESTS = "run_tests"
    GIT_DIFF = "git_diff"


class ToolRequest(BaseModel):
    tool_type: ToolType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    task_id: Optional[str] = None


class ToolResponse(BaseModel):
    success: bool
    output: str
    is_truncated: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SafetyPolicy(BaseModel):
    blocked_commands: List[str] = Field(
        default_factory=lambda: [
            "rm -rf /",
            ":(){ :|:& };:",
            "mkfs",
            "dd if=/dev/zero",
            "shutdown",
            "reboot",
        ]
    )
    max_output_lines: int = 100        # Max lines before sidecar truncates
    head_lines: int = 40              # Lines preserved at top
    tail_lines: int = 40              # Lines preserved at bottom
    max_file_size_bytes: int = 1024 * 1024  # 1MB max file read
