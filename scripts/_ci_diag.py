"""临时诊断脚本（排查完删除）：单独跑一个测试模块，把输出打成 annotation。"""
import subprocess
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "tests.test_rerank_by_language"
proc = subprocess.run([sys.executable, "-m", "unittest", target, "-v"],
                      capture_output=True, text=True)
text = (proc.stdout or "") + (proc.stderr or "")
for line in text.splitlines()[-40:]:
    print(f"::error::{line[:900]}")
print(f"::error::exit={proc.returncode}")
