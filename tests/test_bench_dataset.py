"""基准数据集（`datasets/k8s-zh-en-v1/`）的守卫。

守三件事：

1. **transform 真的剥干净了**：Hugo 的 front matter / HTML 注释 / 模板短代码不是正文，
   留着会被切进 chunk、进向量、进召回（实测第一版有 30% 的字节是这些噪声）；
2. **manifest 与数据对得上**：sha256 + 体积，少一个文件、多一个文件都要报；
3. **许可证说清楚**：这批文档是 CC BY 4.0（不是本仓库的 Apache-2.0），
   NOTICE 里必须写明归属与「只做格式归一、未改写正文」。
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "k8s-zh-en-v1"

import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_bench_dataset", ROOT / "scripts" / "build_bench_dataset.py")
fetch = _load("fetch_dataset", ROOT / "scripts" / "fetch_dataset.py")


class TransformTest(unittest.TestCase):
    """剥元数据的规则要与数据集里实际用的那条一致。"""

    def test_strips_comment_front_matter_markers_and_shortcodes(self):
        raw = (
            "\n<!--\nreviewers:\n- alice\n-->\n"
            "---\ntitle: EndpointSlices\ncontent_type: concept\n---\n"
            "\n<!-- overview -->\n\n"
            '{{< feature-state for_k8s_version="v1.21" state="stable" >}}\n\n'
            "<!-- body -->\n\n## EndpointSlice API\n\n正文第一段。\n"
        )
        out = build.strip_front_matter(raw)
        self.assertEqual(out, "## EndpointSlice API\n\n正文第一段。\n")

    def test_strips_two_consecutive_front_matter_blocks(self):
        raw = ("---\nreviewers:\n- alice\ntitle: X\n---\n"
               "---\ntitle: X\nweight: 60\n---\n# 标题\n\n正文\n")
        self.assertEqual(build.strip_front_matter(raw), "# 标题\n\n正文\n")

    def test_strips_multiline_shortcodes(self):
        raw = '{{< glossary_tooltip\n    text="drained"\n    term_id="drain" >}}\n\n正文\n'
        self.assertEqual(build.strip_front_matter(raw), "正文\n")

    def test_keeps_go_template_examples(self):
        """文档里真的有 Go 模板示例，那是正文，不许当短代码删掉。"""
        raw = "示例：\n\n    name: {{ .Values.name }}\n"
        self.assertIn("{{ .Values.name }}", build.strip_front_matter(raw))

    def test_normalizes_bom_and_crlf(self):
        raw = "\ufeff---\ntitle: X\n---\r\n正文\r\n"
        self.assertEqual(build.strip_front_matter(raw), "正文\n")

    def test_is_idempotent(self):
        raw = "---\ntitle: X\n---\n<!-- body -->\n\n正文\n"
        once = build.strip_front_matter(raw)
        self.assertEqual(build.strip_front_matter(once), once)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((DATASET / "sources.json").read_text(encoding="utf-8"))

    def test_license_and_attribution_are_explicit(self):
        self.assertEqual(self.manifest["license"], "CC-BY-4.0")
        self.assertIn("Kubernetes Authors", self.manifest["attribution"])
        self.assertEqual(self.manifest["license_file"], "LICENSE-CC-BY-4.0")

    def test_upstream_commit_is_pinned(self):
        commit = self.manifest["upstream"]["commit"]
        self.assertRegex(commit, r"^[0-9a-f]{40}$", "上游 commit 必须钉死，否则不可复现")
        self.assertEqual(self.manifest["upstream"]["repo"], "kubernetes/website")

    def test_selection_rule_is_recorded(self):
        selection = self.manifest["selection"]
        self.assertEqual(selection["files"], len(self.manifest["files"]))
        self.assertEqual(selection["pages_per_language"] * 2, selection["files"])
        self.assertIn("reference", selection["excluded_sections"])

    def test_every_file_entry_is_checkable(self):
        commit = self.manifest["upstream"]["commit"]
        for entry in self.manifest["files"]:
            self.assertIn(entry["lang"], ("zh-cn", "en"))
            self.assertTrue(entry["path"].endswith(".md"))
            self.assertIn(commit, entry["url"], "URL 必须指向钉死的 commit")
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)

    def test_archive_hash_is_recorded(self):
        """Release 资产要能校验「包本身」，不能只校验解包后的文件。"""
        self.assertRegex(self.manifest.get("archive_sha256", ""), r"^[0-9a-f]{64}$")
        self.assertGreater(self.manifest.get("archive_bytes", 0), 0)
        self.assertIn("releases/download", self.manifest.get("release_url", ""))


class LocalDataTest(unittest.TestCase):
    """只有数据取回来了才跑（仓库里不放数据本体）。"""

    def setUp(self):
        if not (DATASET / "docs").is_dir():
            self.skipTest("数据本体不在（正常：由 scripts/fetch_dataset.py 取回）")
        self.manifest = json.loads((DATASET / "sources.json").read_text(encoding="utf-8"))

    def test_every_file_matches_manifest(self):
        self.assertEqual(fetch.verify(DATASET, self.manifest), [])

    def test_no_metadata_noise_left(self):
        docs = DATASET / "docs"
        offenders = []
        for path in docs.rglob("*.md"):
            head = path.read_text(encoding="utf-8").lstrip()
            if head.startswith(("<!--", "---")):
                offenders.append(f"{path.relative_to(docs).as_posix()}: 元数据开头")
            elif "{{<" in head or "{{%" in head:
                offenders.append(f"{path.relative_to(docs).as_posix()}: 模板短代码")
        self.assertEqual(offenders, [], f"这些文件里还有元数据噪声：{offenders[:5]}")


class FetchVerifyTest(unittest.TestCase):
    """`--verify` 必须能真的抓出「少文件 / 内容变了」，并给可复制的修复命令。"""

    def _make_dataset(self, tmp: str) -> Path:
        root = Path(tmp) / "bench"
        (root / "docs" / "zh-cn").mkdir(parents=True)
        (root / "docs" / "en").mkdir(parents=True)
        body = {"zh-cn/a.md": "# 标题\n\n正文\n".encode("utf-8"),
                "en/a.md": "# Title\n\nbody\n".encode("utf-8")}
        entries = []
        for rel, data in body.items():
            path = root / "docs" / rel
            path.write_bytes(data)
            entries.append({"lang": rel.split("/")[0], "path": "a.md",
                            "url": "https://example.com", "sha256": fetch.sha256_of(path),
                            "bytes": len(data)})
        (root / "sources.json").write_text(json.dumps({
            "dataset": "bench", "license": "CC-BY-4.0", "license_file": "LICENSE-CC-BY-4.0",
            "attribution": "x", "upstream": {"repo": "r", "commit": "0" * 40, "url": "u"},
            "selection": {"rule": "r", "sections": [], "excluded_sections": [],
                          "pages_per_language": 1, "files": 2},
            "transform": "t", "files": entries,
        }), encoding="utf-8")
        return root

    def test_ok_when_everything_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(tmp)
            self.assertEqual(fetch.verify(root, fetch.load_manifest(root)), [])

    def test_reports_missing_and_modified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(tmp)
            (root / "docs" / "zh-cn" / "a.md").unlink()
            (root / "docs" / "en" / "a.md").write_text("被改过\n", encoding="utf-8")
            problems = fetch.verify(root, fetch.load_manifest(root))
            joined = "\n".join(problems)
            self.assertIn("缺文件", joined)
            self.assertIn("内容不一致", joined)

    def test_reports_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(tmp)
            (root / "docs" / "zh-cn" / "b.md").write_text("多出来的\n", encoding="utf-8")
            problems = fetch.verify(root, fetch.load_manifest(root))
            self.assertTrue(any("多出来的文件" in p for p in problems), problems)

    def test_main_exits_nonzero_with_fix_command(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dataset(tmp)
            (root / "docs" / "en" / "a.md").unlink()
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = fetch.main(["--root", str(root), "--verify"])
            self.assertEqual(code, 1)
            self.assertIn("scripts/fetch_dataset.py", err.getvalue(), "必须给可复制的修复命令")


class DatasetLegalTest(unittest.TestCase):
    def test_notice_names_the_license_and_author(self):
        notice = (DATASET / "NOTICE").read_text(encoding="utf-8")
        for must in ("CC BY 4.0", "The Kubernetes Authors", "kubernetes/website",
                     "没有改写任何一句正文"):
            self.assertIn(must, notice, f"数据集 NOTICE 缺关键说明：{must}")

    def test_notice_explains_double_license(self):
        notice = (DATASET / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("Apache-2.0", notice, "必须说明它与仓库主许可证是两套")

    def test_license_file_is_the_real_text(self):
        text = (DATASET / "LICENSE-CC-BY-4.0").read_text(encoding="utf-8", errors="ignore")
        # 全文的关键条款都在，才算「真许可证」而不是一句话
        for must in ("Attribution 4.0 International", "Section 2 -- Scope",
                     "Section 3 -- License Conditions", "a. Attribution",
                     "Section 5 -- Disclaimer of Warranties and Limitation of Liability"):
            self.assertIn(must, text, f"许可证全文缺条款：{must}")

    def test_root_notice_mentions_the_benchmark(self):
        text = (ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("k8s-zh-en-v1", text, "根 NOTICE 必须点出这个独立许可的数据集")


if __name__ == "__main__":
    unittest.main()
