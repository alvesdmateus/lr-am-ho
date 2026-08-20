import os
import shutil
import unittest
from tools.protocol import ToolRequest, ToolType, SafetyPolicy
from tools.sidecar import ToolSidecar


class TestToolSidecar(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_sidecar_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.policy = SafetyPolicy(
            max_output_lines=20,
            head_lines=5,
            tail_lines=5,
        )
        self.sidecar = ToolSidecar(workspace_dir=self.test_dir, policy=self.policy)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_safety_guard_blocks_forbidden_commands(self):
        req = ToolRequest(
            tool_type=ToolType.BASH,
            parameters={"command": "rm -rf / --no-preserve-root"},
        )
        res = self.sidecar.dispatch(req)
        self.assertFalse(res.success)
        self.assertIn("Security Policy Violation", res.error)

    def test_file_write_and_read(self):
        write_req = ToolRequest(
            tool_type=ToolType.FILE_WRITE,
            parameters={"file_path": "sub/hello.txt", "content": "Hello Sidecar"},
        )
        write_res = self.sidecar.dispatch(write_req)
        self.assertTrue(write_res.success)

        read_req = ToolRequest(
            tool_type=ToolType.FILE_READ,
            parameters={"file_path": "sub/hello.txt"},
        )
        read_res = self.sidecar.dispatch(read_req)
        self.assertTrue(read_res.success)
        self.assertEqual(read_res.output, "Hello Sidecar")

    def test_output_truncation_preserves_head_and_tail(self):
        # Generate 100 lines of output via bash
        cmd = 'python -c "for i in range(100): print(f\'line_{i}\')"'
        req = ToolRequest(
            tool_type=ToolType.BASH,
            parameters={"command": cmd},
        )
        res = self.sidecar.dispatch(req)
        self.assertTrue(res.success)
        self.assertTrue(res.is_truncated)
        self.assertIn("TRUNCATED", res.output)
        self.assertIn("line_0", res.output)
        self.assertIn("line_99", res.output)
        self.assertEqual(res.metadata["original_line_count"], 100)


if __name__ == "__main__":
    unittest.main()
