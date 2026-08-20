import os
import shutil
import subprocess
import unittest
from tools.sandbox import LocalSandbox, DockerSandbox


class TestSandboxes(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_sandbox_backends"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Local Sandbox Tests
    # -------------------------------------------------------------------------
    def test_local_sandbox_file_and_command(self):
        sandbox = LocalSandbox(workspace_dir=self.test_dir)
        
        # Write file
        write_res = sandbox.write_file("greeting.txt", "Hello from Local Sandbox")
        self.assertTrue(write_res.success)
        self.assertEqual(write_res.metadata["sandbox_type"], "local")

        # Read file
        read_res = sandbox.read_file("greeting.txt")
        self.assertTrue(read_res.success)
        self.assertEqual(read_res.output, "Hello from Local Sandbox")

        # Execute command in workspace
        cmd_res = sandbox.execute_command("python -c \"import os; print('EXISTS:', os.path.exists('greeting.txt'))\"")
        self.assertTrue(cmd_res.success)
        self.assertIn("EXISTS: True", cmd_res.output)

    def test_local_sandbox_timeout(self):
        sandbox = LocalSandbox(workspace_dir=self.test_dir)
        # Run command with 1 second timeout
        res = sandbox.execute_command("python -c \"import time; time.sleep(3)\"", timeout=1)
        self.assertFalse(res.success)
        self.assertIn("timed out", res.error.lower())

    # -------------------------------------------------------------------------
    # Docker Sandbox Tests
    # -------------------------------------------------------------------------
    def _is_docker_available(self) -> bool:
        try:
            res = subprocess.run("docker info", shell=True, capture_output=True, timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def test_docker_sandbox_lifecycle(self):
        if not self._is_docker_available():
            self.skipTest("Docker daemon is not running or accessible.")

        sandbox = DockerSandbox(
            workspace_dir=self.test_dir,
            image="python:3.11-slim",
            container_name_prefix="test_harness",
        )

        try:
            self.assertIsNotNone(sandbox.container_id)

            # Write file via sandbox mount
            write_res = sandbox.write_file("docker_test.py", "print('Executed inside Docker container')\n")
            self.assertTrue(write_res.success)

            # Execute command inside container
            exec_res = sandbox.execute_command("python docker_test.py")
            self.assertTrue(exec_res.success)
            self.assertIn("Executed inside Docker container", exec_res.output)
            self.assertEqual(exec_res.metadata["sandbox_type"], "docker")
        finally:
            # Clean destruction
            sandbox.reset()
            self.assertIsNone(sandbox.container_id)


if __name__ == "__main__":
    unittest.main()
