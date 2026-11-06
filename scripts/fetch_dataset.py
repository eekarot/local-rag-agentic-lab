"""取回 / 校验基准数据集（`datasets/<name>/docs`）。

两种来源，都不需要联网情形之外的凭据：

1. **Release 资产**（默认）：下载 `bench-<name>.tar.gz`，校验 sha256 后解包；
2. **本地目录**：`--from-dir <已有的 docs 目录>`，直接复制并校验。

为什么需要它：数据本体（944 个 Markdown，约 11 MB）不进 git —— 它是**只读基准**，
进 git 会让每次 clone 变慢，而内容一旦变动就该叫 v2。仓库里只提交 manifest。

    python scripts/fetch_dataset.py                  # 下载（或已存在则只校验）
    python scripts/fetch_dataset.py --verify         # 只校验，不下载
    python scripts/fetch_dataset.py --from-dir /path/to/docs

校验口径：`sources.json` 里每个文件的 sha256 + bytes。少文件、多文件、内容变了，
都算失败，并打印**可复制的修复命令**（不给错药）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

DEFAULT_NAME = "k8s-zh-en-v1"


def load_manifest(root: Path) -> dict:
    path = root / "sources.json"
    if not path.is_file():
        raise SystemExit(f"找不到 manifest：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, manifest: dict) -> list[str]:
    """返回问题清单（空 = 全对）。**只读**，不动任何文件。"""
    docs = root / "docs"
    problems: list[str] = []
    if not docs.is_dir():
        return [f"缺少数据目录：{docs}"]
    for entry in manifest["files"]:
        path = docs / entry["lang"] / entry["path"]
        if not path.is_file():
            problems.append(f"缺文件：{path.relative_to(root).as_posix()}")
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(f"内容不一致（sha256 不符）：{path.relative_to(root).as_posix()}")
    extra = {p.relative_to(docs).as_posix() for p in docs.rglob("*.md")}
    expected = {f"{e['lang']}/{e['path']}" for e in manifest["files"]}
    for name in sorted(extra - expected)[:5]:
        problems.append(f"多出来的文件：{name}")
    if len(extra - expected) > 5:
        problems.append(f"……另有 {len(extra - expected) - 5} 个多余文件")
    return problems


def copy_from(docs_src: Path, root: Path) -> None:
    docs_dst = root / "docs"
    if docs_dst.exists():
        shutil.rmtree(docs_dst)
    shutil.copytree(docs_src, docs_dst)


def download(manifest: dict, root: Path, url: str | None) -> None:
    target = url or manifest.get("release_url")
    if not target:
        raise SystemExit(
            "manifest 里没有 release_url，请用 --url <下载地址> 或 --from-dir <本地目录>。\n"
            "（数据本体不打进 git：944 个文件约 11 MB，进 git 会让每次 clone 变慢）"
        )
    print(f"下载：{target}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "bench.tar.gz"
        with urlopen(target, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        expected = manifest.get("archive_sha256")
        if expected and sha256_of(archive) != expected:
            raise SystemExit(f"下载的压缩包 sha256 不符：{archive}\n先别解包，删掉重下。")
        with tarfile.open(archive) as tar:
            tar.extractall(root)          # 解到 datasets/<name>/，包内顶层就是 docs/


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_dataset", description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--root", type=Path, default=None, help="数据集目录（默认 datasets/<name>）")
    parser.add_argument("--url", default=None, help="Release 资产地址（默认读 manifest.release_url）")
    parser.add_argument("--from-dir", type=Path, default=None, help="从本地 docs 目录复制")
    parser.add_argument("--verify", action="store_true", help="只校验，不下载")
    args = parser.parse_args(argv)

    root = args.root or Path("datasets") / args.name
    manifest = load_manifest(root)

    if args.from_dir:
        copy_from(args.from_dir, root)
    elif not args.verify and not (root / "docs").is_dir():
        download(manifest, root, args.url)

    problems = verify(root, manifest)
    if problems:
        print("校验未通过：", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        print("\n修复：python scripts/fetch_dataset.py --from-dir <上游 docs 目录>", file=sys.stderr)
        print("     或：python scripts/fetch_dataset.py --url <下载地址>", file=sys.stderr)
        return 1

    print(f"OK：{len(manifest['files'])} 个文件全部与 manifest 一致"
          f"（{manifest['dataset']}，上游 {manifest['upstream']['commit'][:7]}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
