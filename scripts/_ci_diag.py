"""临时诊断脚本（排查完删除）：单独跑一个测试模块，只把失败段打成 annotation。"""
import subprocess
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "tests.test_rerank_by_language"
proc = subprocess.run([sys.executable, "-m", "unittest", target, "-v"],
                      capture_output=True, text=True)
text = (proc.stdout or "") + (proc.stderr or "")
lines = text.splitlines()
start = next((i for i, line in enumerate(lines) if line.startswith("====")), None)
segment = lines[start:start + 30] if start is not None else lines[-30:]
for line in segment:
    print(f"::error::{line[:900]}")
