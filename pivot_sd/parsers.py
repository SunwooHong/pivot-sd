"""Code extraction and execution used by MBPP+ scoring."""
import re
import resource
import subprocess
import sys


class Parser:
    @classmethod
    def extract_answer_code(cls, solution_str):
        matches = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(matches):
            return matches[0]
        return None


def test_solution(code_str, output_dir=None):
    def preexec():
        resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))

    try:
        result = subprocess.run(
            [sys.executable, "-c", code_str],
            preexec_fn=preexec,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            cwd=output_dir,
        )
        return 1 if result.returncode == 0 else 0
    except Exception:
        return 0
