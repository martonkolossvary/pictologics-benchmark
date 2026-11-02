from __future__ import annotations

import os
import subprocess
import sys
import unittest

import psutil

from bench.process_tree import memory_in_use, tree_memory_in_use


class ProcessTreeTests(unittest.TestCase):
    def test_memory_in_use_is_measured_for_this_process(self) -> None:
        process = psutil.Process()
        self.assertGreater(memory_in_use(process), 0)
        self.assertGreaterEqual(tree_memory_in_use(process), memory_in_use(process))

    @unittest.skipUnless(os.name == "nt", "Windows Job Objects")
    def test_closing_the_windows_job_ends_its_processes(self) -> None:
        from bench.process_tree import WindowsJob

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        job = WindowsJob(child.pid)
        job.close()
        self.assertIsNotNone(child.wait(timeout=10))


if __name__ == "__main__":
    unittest.main()
