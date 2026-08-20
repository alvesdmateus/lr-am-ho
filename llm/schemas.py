# OpenAI-compatible tool/function schemas for Worker sidecar tools.
# These are passed to the LLM in the `tools` parameter of chat completions.

WORKER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the sandboxed workspace. "
                "Use for running scripts, installing packages, checking file states, "
                "or any shell operation. The command runs in the workspace root directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": (
                "Write content to a file in the workspace. "
                "Creates the file and any parent directories if they don't exist. "
                "Overwrites the file if it already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": (
                "Read the contents of a file in the workspace. "
                "Use to inspect existing code, configuration, or output files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run a test command in the workspace to verify code correctness. "
                "Defaults to 'python -m unittest discover -s tests' if no command specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_cmd": {
                        "type": "string",
                        "description": "The test command to execute. Defaults to unittest discovery.",
                    }
                },
                "required": [],
            },
        },
    },
]
