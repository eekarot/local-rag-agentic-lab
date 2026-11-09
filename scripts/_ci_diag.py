"""临时诊断脚本（排查完删除）：单独跑一个测试模块，只打**末尾**的 traceback。"""
import subprocess
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "tests.test_rerank_by_language"
proc = subprocess.run([sys.executable, "-m", "unittest", target, "-v"],
                      capture_output=True, text=True)
text = (proc.stdout or "") + (proc.stderr or "")
lines = [line for line in text.splitlines()
         if not set(line.strip()) <= {".", "s", "F", "E"} and "重试" not in line]
for line in lines[-25:]:
    print(f"::error::{line[:900]}")
