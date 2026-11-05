"""O6：环境变量清单 / 依赖清单必须与代码一致 —— **靠用例锁死，不靠人记得改**。

这个用例存在的理由很具体：`RAG_*` 有三十多个，散在七八个文件里用 `os.getenv` 直接读。
任何一次「加了一个开关但忘了写进 `.env.example`」，用户就永远不知道它存在；
反过来 `.env.example` 里留一个已经不读的变量，用户改了没反应 —— 两种都是纯粹的坑。

所以这里做集合相等（两边都必须为空差集），外加两条版本纪律：
- 开发依赖要吃进运行时依赖（`-r requirements.txt`），否则「装完还是缺包」；
- `qdrant-client` 的 minor 必须等于 `docker-compose.yml` 里镜像的 minor，且 ≥1.16.2
  （v1.15.0 在大集合优化期会 panic，见设计文档 D19）。
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"
DEV_REQS = ROOT / "requirements-dev.txt"

#: 代码里真正会被读到的变量名（字面量）
_LITERAL = re.compile(
    r'''(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z_0-9]+)["']'''
    r'''|environ\[\s*["']([A-Z][A-Z_0-9]+)["']\s*\]'''
    r'''|^\s*[A-Z_0-9]*ENV[A-Z_0-9]*\s*=\s*["']([A-Z][A-Z_0-9]+)["']''',
    re.MULTILINE,
)
_SCAN_DIRS = ("src", "scripts", "tests", "experiments")


def env_vars_read_by_code() -> set[str]:
    """扫出代码实际读的环境变量：字面量 + `make_queue_from_env` 拼出来的那 8 个。"""
    found: set[str] = set()
    for directory in _SCAN_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for groups in _LITERAL.findall(text):
                found.add(next(g for g in groups if g))
    found |= generated_queue_vars()
    return found


def generated_queue_vars() -> set[str]:
    """`f"{prefix}_{name}"` 形式的变量：前缀来自两处调用，字段名从 `queue.py` 正文里读。"""
    import inspect

    from src.rag.api import queue as queue_module

    defaults = inspect.signature(queue_module.make_queue_from_env).parameters
    body = inspect.getsource(queue_module.make_queue_from_env)
    fields = set(re.findall(r'_(?:int|float)\(\s*"([A-Z_]+)"', body))
    assert fields == {"MAX_INFLIGHT", "MAX_QUEUE", "QUEUE_TIMEOUT", "RETRY_AFTER"}, (
        "queue.py 里的字段名变了，请同步本用例与 .env.example"
    )
    prefixes = {defaults["prefix"].default, "RAG_RETRIEVE"}  # 生成队列 / 检索队列
    return {f"{prefix}_{field}" for prefix in prefixes for field in fields}


def documented_vars() -> set[str]:
    """`.env.example` 里**未注释**的赋值行。"""
    found: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        found.add(line.split("=", 1)[0].strip())
    return found


class EnvExampleTest(unittest.TestCase):
    def test_example_file_exists(self):
        self.assertTrue(ENV_EXAMPLE.exists(), "缺少 .env.example：用户不知道有哪些开关")

    def test_every_var_read_by_code_is_documented(self):
        missing = env_vars_read_by_code() - documented_vars()
        self.assertEqual(missing, set(), f"代码会读但 .env.example 没写：{sorted(missing)}")

    def test_no_documented_var_is_invented(self):
        stale = documented_vars() - env_vars_read_by_code()
        self.assertEqual(stale, set(), f".env.example 里有代码根本不读的变量：{sorted(stale)}")

    #: 变量上方 6 行内必须出现过注释（分组注释也算），既有说明又不逼人逐行写
    COMMENT_LOOKBACK = 6

    def test_every_switch_has_a_default_value_and_a_comment(self):
        """每个变量都要有值，且上方不远处有说明，否则用户只能猜。"""
        lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        problems = []
        last_comment = -99
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                last_comment = index
                continue
            if not stripped or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if not value.strip():
                problems.append(f"{name} 没有默认值")
            if index - last_comment > self.COMMENT_LOOKBACK:
                problems.append(f"{name} 上方 {self.COMMENT_LOOKBACK} 行内没有注释")
        self.assertEqual(problems, [], f"缺值或缺注释：{problems}")


class RequirementsTest(unittest.TestCase):
    def test_dev_requirements_extend_runtime_and_pin_httpx(self):
        self.assertTrue(DEV_REQS.exists(), "缺少 requirements-dev.txt")
        text = DEV_REQS.read_text(encoding="utf-8")
        self.assertIn("-r requirements.txt", text, "开发依赖必须吃进运行时依赖")
        self.assertRegex(text, r"(?m)^httpx[<>=]", "httpx 必须显式声明：TestClient 靠它，"
                                                    "如今是被 qdrant-client 顺手装进来的")

    def test_qdrant_client_matches_image_and_is_new_enough(self):
        client_pin = re.search(r"(?m)^qdrant-client==(\d+)\.(\d+)\.(\d+)", (ROOT / "requirements.txt").read_text(encoding="utf-8"))
        image = re.search(r"(?m)^\s*image:\s*qdrant/qdrant:v(\d+)\.(\d+)\.(\d+)", (ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        self.assertIsNotNone(client_pin, "requirements.txt 必须把 qdrant-client 钉到具体版本")
        self.assertIsNotNone(image, "docker-compose.yml 必须把镜像钉到具体版本")
        client_minor = client_pin.group(1, 2)
        image_minor = image.group(1, 2)
        self.assertEqual(client_minor, image_minor, "客户端与镜像 minor 必须对齐")
        self.assertGreaterEqual(tuple(int(x) for x in image.groups()), (1, 16, 2),
                                "镜像必须 ≥1.16.2：1.15.0 在大集合优化期会 panic")


class DefaultsMatchCodeTest(unittest.TestCase):
    """`.env.example` 里的默认值必须和代码里的默认值一致 —— 否则复制一份就开始骗自己。"""

    def _literal_defaults(self) -> dict[str, str]:
        """从源代码里抽 os.getenv 带**字面量默认值**的调用（两个参数都是字符串）。"""
        pattern = re.compile(r'getenv\(\s*"([A-Z][A-Z_0-9]+)"\s*,\s*"([^"]*)"\s*\)')
        found: dict[str, str] = {}
        for path in (ROOT / "src").rglob("*.py"):
            for name, value in pattern.findall(path.read_text(encoding="utf-8", errors="ignore")):
                found[name] = value
        return found

    def _queue_defaults(self) -> dict[str, str]:
        """队列变量：前缀 × 字段，值来自 `make_queue_from_env` 签名与两处调用实参。"""
        import inspect

        from src.rag.api import queue as queue_module

        params = inspect.signature(queue_module.make_queue_from_env).parameters
        app = (ROOT / "src" / "rag" / "api" / "app.py").read_text(encoding="utf-8")
        retrieve = re.search(r'prefix="RAG_RETRIEVE".*?name="检索"', app, re.S).group(0)
        retrieve_args = {
            "MAX_INFLIGHT": "4",
            "MAX_QUEUE": "2",
            "QUEUE_TIMEOUT": "5.0",
            "RETRY_AFTER": re.search(r"retry_after_s=([\d.]+)", retrieve).group(1),
        }
        assert params["retry_after_s"].default == 60.0, "生成队列的 Retry-After 默认值变了"
        ask_args = {
            "MAX_INFLIGHT": "1",                       # concurrency 默认值
            "MAX_QUEUE": "2",                          # queue 默认值
            "QUEUE_TIMEOUT": "120.0",                  # timeout_s 默认值
            "RETRY_AFTER": "60.0",                     # retry_after_s 默认值
        }
        merged = {f"RAG_{k}": v for k, v in ask_args.items()}
        merged.update({f"RAG_RETRIEVE_{k}": v for k, v in retrieve_args.items()})
        return merged

    def test_documented_defaults_match_code(self):
        documented = {}
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                documented[key.strip()] = value.strip()

        expected = self._literal_defaults()
        queue_defaults = self._queue_defaults()
        checked = 0
        drifted = []
        for name, value in queue_defaults.items():
            if name in documented:
                checked += 1
                if _same(documented[name], value):
                    continue
                drifted.append(f"{name}: 文档 {documented[name]!r} ≠ 代码 {value!r}")
        for name, value in expected.items():
            if name not in documented:
                continue
            checked += 1
            if _same(documented[name], value):
                continue
            drifted.append(f"{name}: 文档 {documented[name]!r} ≠ 代码 {value!r}")
        self.assertGreaterEqual(checked, 25, f"只比对了 {checked} 个变量，覆盖太少")
        self.assertEqual(drifted, [], "文档默认值与代码默认值不一致：" + "；".join(drifted))


def _same(document: str, code: str) -> bool:
    """数值按数值比（`5.0` 与 `5` 等价），其余按字符串比。"""
    try:
        return float(document) == float(code)
    except (TypeError, ValueError):
        return document == code


if __name__ == "__main__":
    unittest.main()
