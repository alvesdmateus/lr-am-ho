import os
import tempfile
from harness.state import Task
from tools.protocol import ToolType
from agent.worker import Worker, MicroAction

def run_atomic_bash_file_creation_test():
    # 1. Create an isolated temporary workspace directory
    with tempfile.TemporaryDirectory() as sandbox_dir:
        print(f"📁 Isolated Sandbox created at: {sandbox_dir}")

        # 2. Instantiate the Worker (you can use sandbox_type="local" or "docker")
        worker = Worker(
            name="AtomicWorker",
            workspace_dir=sandbox_dir,
            sandbox_type="docker",  # <-- Runs inside container
            docker_image="python:3.11-slim",
        )
        # 3. Define the atomic task
        task = Task(
            id="atomic_task_1",
            title="Create nested config file via Bash",
            description="Create folder 'configs' and file 'settings.json' using bash command",
            definition_of_done=["configs/settings.json exists with valid content"],
        )

        # 4. Define the micro-action that invokes a BASH command to create folder + file
        # Using a cross-platform python one-liner inside bash to create folder and file:
        bash_command = (
            'python -c "'
            'import os; '
            'os.makedirs(\'configs\', exist_ok=True); '
            'open(\'configs/settings.json\', \'w\').write(\'{\\\"env\\\": \\\"production\\\"}\')'
            '"'
        )

        custom_actions = [
            MicroAction(
                step_number=1,
                intent="Create configs directory and settings.json file via bash",
                tool_type=ToolType.BASH,
                parameters={"command": bash_command},
            ),
            MicroAction(
                step_number=2,
                intent="Verify file content via bash assertion",
                tool_type=ToolType.BASH,
                parameters={"command": "python -c \"import json; data=json.load(open('configs/settings.json')); assert data['env'] == 'production'; print('VERIFIED')\""},
            ),
        ]

        # 5. Execute the task
        result = worker.execute_task(
            task=task,
            memory_briefing="=== BRIEFING ===\n- [CONVENTION] Create configs in JSON format.",
            custom_micro_actions=custom_actions,
        )

        # 6. Assertions & Verification
        target_file = os.path.join(sandbox_dir, "configs", "settings.json")
        
        print("\n--- ATOMIC TEST RESULTS ---")
        print(f"Task Success: {result.success}")
        print(f"File Exists on Disk: {os.path.exists(target_file)}")
        
        with open(target_file, "r") as f:
            content = f.read()
        print(f"File Content: {content}")
        print(f"Tool Outputs Logged: {len(result.tool_outputs)}")

        assert result.success is True, "Worker task should succeed"
        assert os.path.exists(target_file), "Target file must exist in sandbox"
        assert '{"env": "production"}' in content, "Content must match"
        print("\n✅ Atomic Test Passed Successfully!")

if __name__ == "__main__":
    run_atomic_bash_file_creation_test()
