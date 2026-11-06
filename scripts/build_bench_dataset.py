"""构建基准数据集 `datasets/k8s-zh-en-v1/`（可复现）。

为什么要自己写一个构建脚本而不是直接 `cp`：

1. **要可复现**：记录上游 commit、每页 sha256、选择规则与变换，别人能用同一条命令
   从上游重建出**逐字节相同**的文件；
2. **要脱掉 Hugo front matter**：实测 `MarkdownParser` 没有剥离 YAML front matter，
   首页三大块会变成 `title: ... / theme_lock / card anchors` 这类元数据噪声，
   它们会进 chunk、进向量、进召回——基准里不能带这种噪声；
3. **要留证据**：manifest (`sources.json`) 就是证据，包含 URL、commit、sha256、体积。

选择规则（写死在代码里，也可从 manifest 读到）：
    content/{zh-cn,en}/docs 两棵树**同相对路径**的交集
    ∩ 一级章节 ∈ {concepts, tasks, tutorials, setup}
    （排除 reference：1,070 页自动生成的 API 清单，不适合出题）

用法：
    python scripts/build_bench_dataset.py --upstream <kubernetes/website 的本地克隆> \
        --out datasets/k8s-zh-en-v1
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import sys
from pathlib import Path

SECTIONS = ("concepts", "tasks", "tutorials", "setup")
LANGUAGES = ("zh-cn", "en")
UPSTREAM_REPO = "kubernetes/website"
UPSTREAM_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
DATASET_NAME = "k8s-zh-en-v1"
TRANSFORM = "strip_leading_metadata_v6"
LICENSE = "CC-BY-4.0"
#: Hugo 模板调用（`{{< include ... >}}` / `{{% heading "x" %}}`）：不是文档内容，纯噪声
#: 短代码可能**跨行**（`{{< glossary_tooltip` 换行写参数，最后才是 `>}}`），
#: 所以带 re.S；只用 `{{<` / `{{%` 开头匹配，避免误删 Go 模板示例里的 `{{ .Values.x }}`。
_SHORTCODE_ANY = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
#: HTML 注释：既可能是文件头的 reviewers 名单，也可能是正文里的 `<!-- overview -->` /
#: `<!-- body -->` 分段标记。两种都不是给读者看的正文。
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def strip_front_matter(text: str) -> str:
    """去掉文档开头的元数据：HTML 注释（reviewers 名单）与 Hugo YAML front matter。
    为什么要循环剥：实测 944 个文件里 **840 个**开头是 `<!-- reviewers: ... -->`，
    剥完注释后面才是 `---` front matter；只处理其中一种，元数据就照样进 chunk。
    第一版只认「以 `---` 开头」，于是 449 个文件的开头 400 字符里还留着 `title:`
    —— 这类噪声会进向量、进召回，基准里不能有。

    行尾统一成 `\\n`、尾部留一个换行，保证逐字节可复现。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    # 交替剥：三段元数据会交叉出现（front matter → `<!-- overview -->` → 短代码 →
    # `<!-- body -->`），只剥一轮就会停在短代码上，而 `<!-- body -->` 仍留在开头
    # （实测 115 个文件就是这样漏下来的）。
    for _ in range(6):
        before = text
        text = _SHORTCODE_ANY.sub("", text)
        text = _HTML_COMMENT.sub("", text)
        text = text.lstrip()
        if text.startswith("---"):
            lines = text.split("\n")
            closer = next((i for i in range(1, min(len(lines), 200))
                           if lines[i].strip() == "---"), None)
            block = "\n".join(lines[1:closer]) if closer is not None else ""
            # 只有块内确实像 front matter（有 key: 形式）才剥；
            # 否则可能只是正文里的一条水平线。
            if closer is not None and ":" in block:
                text = "\n".join(lines[closer + 1:])
        if text == before:
            break
    text = re.sub(r"\n{3,}", "\n\n", text)          # 连续空行压成一个（sha256 更稳定）
    return text.rstrip() + "\n"


def pack_docs(root: Path, archive: Path) -> None:
    """把 `docs/` 打成**可复现**的 tar.gz。

    为什么不用 `tar -czf`：它会把 mtime / 属主 / 目录项顺序写进包里，同样的内容
    两次打包 sha256 不同 —— 那 `archive_sha256` 就永远对不上（本轮实测：
    第一次打包 a6a0ea…，重打一次 767f3d…，内容完全一样）。
    这里固定排序、mtime=0、属主 0，并用 `gzip -n`（不写时间戳）。
    """
    import gzip
    import io
    import tarfile

    archive.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in (root / "docs").rglob("*") if p.is_file()),
                   key=lambda p: p.relative_to(root).as_posix())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in files:
            info = tarfile.TarInfo(name=f"docs/{path.relative_to(root / 'docs').as_posix()}")
            data = path.read_bytes()
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    archive.write_bytes(gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0))


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(upstream: Path) -> list[str]:
    """两棵树同路径的交集，按相对路径排序（稳定顺序 = 可复现）。"""
    trees = {}
    for lang in LANGUAGES:
        base = upstream / "content" / lang / "docs"
        if not base.is_dir():
            raise SystemExit(f"上游目录不存在：{base}（要用 kubernetes/website 的稀疏克隆）")
        trees[lang] = {p.relative_to(base).as_posix() for p in base.rglob("*.md")}
    both = trees["zh-cn"] & trees["en"]
    picked = sorted(path for path in both if path.split("/")[0] in SECTIONS)
    if not picked:
        raise SystemExit("交集为空：检查上游克隆是否包含 content/{zh-cn,en}/docs")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_bench_dataset", description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="kubernetes/website 本地克隆")
    parser.add_argument("--out", type=Path, default=Path("datasets") / DATASET_NAME)
    parser.add_argument("--commit", default=None, help="上游 commit（默认读 git rev-parse HEAD）")
    parser.add_argument("--archive", type=Path, default=None,
                        help="同时记录打好的 tar.gz 的 sha256（Release 资产用）")
    parser.add_argument("--release-url", default=None, help="Release 资产地址模板")
    args = parser.parse_args(argv)

    commit = args.commit
    if not commit:
        head = args.upstream / ".git"
        if not head.exists():
            raise SystemExit("上游不是 git 仓库，请显式给 --commit <sha>")
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.upstream,
                                capture_output=True, text=True, check=True).stdout.strip()

    paths = collect(args.upstream)
    entries = []
    for path in paths:
        for lang in LANGUAGES:
            src = args.upstream / "content" / lang / "docs" / path
            data = strip_front_matter(src.read_text(encoding="utf-8")).encode("utf-8")
            dst = args.out / "docs" / lang / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            entries.append({
                "lang": lang,
                "path": path,
                "url": UPSTREAM_RAW.format(repo=UPSTREAM_REPO, commit=commit,
                                           path=f"content/{lang}/docs/{path}"),
                "sha256": sha256_of(data),
                "bytes": len(data),
            })

    manifest = {
        "dataset": DATASET_NAME,
        "license": LICENSE,
        "license_file": "LICENSE-CC-BY-4.0",
        "attribution": "© The Kubernetes Authors — 文档来自 kubernetes/website，按 CC BY 4.0 使用",
        "upstream": {
            "repo": UPSTREAM_REPO,
            "commit": commit,
            "url": f"https://github.com/{UPSTREAM_REPO}/tree/{commit}",
        },
        "selection": {
            "rule": "intersection(content/zh-cn/docs, content/en/docs) ∩ sections",
            "sections": list(SECTIONS),
            "excluded_sections": ["reference"],
            "pages_per_language": len(paths),
            "files": len(entries),
        },
        "transform": TRANSFORM,
        "files": entries,
    }
    # Release 资产信息：让 fetch_dataset.py 能校验下载的压缩包本身，
    # 而不是只校验解包后的文件（半个文件也能"解包成功"）。
    if args.archive:
        pack_docs(args.out, args.archive)
        data = args.archive.read_bytes()
        manifest["archive_bytes"] = len(data)
        manifest["archive_sha256"] = sha256_of(data)
    if args.release_url:
        manifest["release_url"] = args.release_url
    (args.out / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total = sum(e["bytes"] for e in entries)
    print(f"页面 {len(paths)} × 语言 {len(LANGUAGES)} = {len(entries)} 个文件，"
          f"共 {total / 1e6:.1f} MB")
    print(f"manifest：{args.out / 'sources.json'}")
    print(f"上游 commit：{commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
