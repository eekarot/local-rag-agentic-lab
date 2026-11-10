"""取回并归一 **SciFact**（BEIR 版的 scifact）—— 只做格式归一，不改写正文。

为什么走原始 release 而不是 BEIR 的 zip：
  * BEIR 的 `scifact.zip` 挂在 `public.ukp.informatik.tu-darmstadt.de`，本机实测 **157 B/s**
    （2.8 MB 要 5 小时）；HuggingFace 目前不可达（HTTP 000）；
  * 原始 release（`scifact.s3-us-west-2.amazonaws.com`，约 12 KB/s，3.1 MB）能下完；
  * 两者的 query / qrels 口径可以对齐：BEIR-SciFact 的 queries 1,109 = SciFact 的
    `claims_train` 809 + `claims_dev` 300；BEIR 的 test qrels 339 条 =
    `claims_dev` 里 `cited_doc_ids` 的并集条数（下面 `derive_doc_qrels()` 会当场对账，
    数字不一致就报错，避免悄悄变成另一个基准）。

许可证（`allenai/scifact/LICENSE.md` 原文，已实查）：
  * claims 与 evidence 标注 —— **CC BY 4.0**
  * corpus 摘要来自 S2ORC —— **ODC-By 1.0**

用法：

    python scripts/fetch_beir_scifact.py            # 下载 + 归一 + 写 manifest
    python scripts/fetch_beir_scifact.py --verify   # 只校验本地产物与 manifest 一致
    python scripts/fetch_beir_scifact.py --offline  # 用已解压的归档重建 docs/ 与 qrels

产物（`datasets/scifact-v1/`）：

    docs/<doc_id>.md      5,183 篇（gitignore，不入库）
    queries.jsonl         1,109 条 claim（id / claim / split）
    qrels/test.tsv        300 题、339 条 doc 级判断（BEIR 口径）
    qrels/train.tsv       809 题
    sources.json          sha256 / 体积 / 计数 / transform 说明（可复现依据）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "scifact-v1"
CACHE_DIR = ROOT / "storage" / "scifact" / "_download"
RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
UPSTREAM = "https://github.com/allenai/scifact"

#: 可评测的切分。`claims_test.jsonl` 只有 claim、没有任何标注（官方留作 leaderboard），
#: 拿它算指标只会得到「全部 0 命中」的假数字 —— 所以它不在这个字典里。
SPLITS = {"train": "claims_train.jsonl", "dev": "claims_dev.jsonl"}
#: 落盘用的文件名：BEIR 把 dev 当作 test 发布（`qrels/test.tsv`），这里保持同名，
#: 否则拿 BEIR 的数字来对账时会找不到文件（踩过一次，manifest 直接 FileNotFound）。
QREL_FILES = {"train": "train.tsv", "dev": "test.tsv"}
TEST_SPLIT_NOTE = "claims_test.jsonl 无标注（官方留作 leaderboard），不参与评测；可评测的是 dev"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_doc_markdown(doc: Mapping) -> str:
    """一篇摘要 → 一个 Markdown 文件：标题当一级标题，正文原样。"""
    title = str(doc.get("title") or "").strip() or f"doc {doc.get('doc_id')}"
    abstract = doc.get("abstract")
    if isinstance(abstract, (list, tuple)):
        body = " ".join(str(part).strip() for part in abstract if str(part).strip())
    else:
        body = str(abstract or "").strip()
    return f"# {title}\n\n{body}\n"


def derive_doc_qrels(claims: Sequence[Mapping]) -> dict[str, list[str]]:
    """从 claim 的 `cited_doc_ids`（缺了就从 evidence 里捞）推出 doc 级 qrels。

    只有「至少有一条相关文档」的 claim 才进 qrels —— 这就是 BEIR 的口径
    （SciFact 里标成 NOT_ENOUGH_INFO 的 claim 没有相关文档）。
    """
    qrels: dict[str, list[str]] = {}
    for claim in claims:
        doc_ids = {str(d) for d in (claim.get("cited_doc_ids") or [])}
        if not doc_ids:
            for items in (claim.get("evidence") or {}).values():
                for item in items or ():
                    doc_ids.update(str(d) for d in (item.get("cited_doc_ids") or []))
        if doc_ids:
            qrels[str(claim["id"])] = sorted(doc_ids)
    return qrels


def write_qrels_tsv(path: Path, qrels: Mapping[str, Sequence[str]]) -> None:
    """BEIR 的 qrels 形状：`query-id\\tcorpus-id\\tscore`（二元相关，score 恒为 1）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore"]
    for query_id in sorted(qrels, key=lambda value: (len(value), value)):
        for doc_id in qrels[query_id]:
            lines.append(f"{query_id}\t{doc_id}\t1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str = RELEASE_URL, dest: Path | None = None) -> Path:
    """带断点续传的下载（本机实测这条链路很慢，中断是常态）。"""
    dest = dest or (CACHE_DIR / "data.tar.gz")
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = None
    request = urllib.request.Request(url, method="HEAD")
    request.add_header("User-Agent", "local-rag-agentic-lab/fetch-beir-scifact")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:   # noqa: S310 - 固定 URL
            expected = int(response.headers.get("Content-Length") or 0) or None
    except Exception:                                    # noqa: BLE001 - 拿不到长度就继续下
        expected = None

    for _attempt in range(6):
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            break
        request = urllib.request.Request(url)
        request.add_header("User-Agent", "local-rag-agentic-lab/fetch-beir-scifact")
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                mode = "ab" if have and response.status == 206 else "wb"
                with dest.open(mode) as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
        except Exception as exc:                          # noqa: BLE001 - 断了就续
            print(f"  下载中断（{type(exc).__name__}），续传…", file=sys.stderr)
        have = dest.stat().st_size if dest.exists() else 0
        print(f"  已下载 {have} / {expected or '?'} 字节", file=sys.stderr)
    if expected and dest.stat().st_size < expected:
        raise SystemExit(f"下载不完整：{dest.stat().st_size} / {expected}")
    return dest


def extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.startswith("data/") and member.isfile() and "/cross_validation/" not in member.name:
                tar.extract(member, into, filter="data")
    return into / "data"


def build(data_dir: Path, out_dir: Path = DATASET_DIR, *, force: bool = False) -> dict:
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    claims = {name: load_jsonl(data_dir / filename) for name, filename in SPLITS.items()}

    docs_dir = out_dir / "docs"
    existing = list(docs_dir.glob("*.md")) if docs_dir.exists() else []
    if len(existing) == len(corpus) and not force:
        # 幂等：数量一致就当作已写过。Windows 上 5,183 个小文件会被杀毒软件逐个扫描，
        # 重写一遍要好几分钟，没必要（--force 才重写）。
        print(f"  docs/ 已有 {len(existing)} 个文件，跳过写入（--force 可重写）", file=sys.stderr)
    else:
        if docs_dir.exists():
            shutil.rmtree(docs_dir)
        docs_dir.mkdir(parents=True)
        for index, doc in enumerate(corpus, start=1):
            (docs_dir / f"{doc['doc_id']}.md").write_text(build_doc_markdown(doc), encoding="utf-8")
            if index % 1000 == 0:
                print(f"  写 docs/：{index}/{len(corpus)}", file=sys.stderr)

    queries = []
    qrels_by_split = {}
    for split, rows in claims.items():
        qrels = derive_doc_qrels(rows)
        qrels_by_split[split] = qrels
        write_qrels_tsv(out_dir / "qrels" / QREL_FILES[split], qrels)
        for claim in rows:
            queries.append({"id": str(claim["id"]), "claim": claim["claim"], "split": split})
    (out_dir / "queries.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in queries), encoding="utf-8")

    test_qrels = qrels_by_split["dev"]
    judgments = sum(len(docs) for docs in test_qrels.values())
    if len(test_qrels) != 300 or judgments != 339:
        raise SystemExit(
            f"qrels 口径对不上 BEIR：{len(test_qrels)} 题 / {judgments} 条（应为 300 / 339）。"
            "先查 claims_dev.jsonl 的 cited_doc_ids 是否被上游改过。"
        )

    manifest = {
        "dataset": "scifact-v1",
        "what": "SciFact（BEIR 版 scifact）的 claim→摘要检索基准",
        "upstream": {
            "project": "SciFact（Wadden et al., EMNLP 2020）",
            "repo": UPSTREAM,
            "release_url": RELEASE_URL,
        },
        "license": {
            "claims_and_evidence": "CC BY 4.0",
            "corpus_abstracts": "ODC-By 1.0（来自 S2ORC）",
            "source": f"{UPSTREAM}/blob/master/LICENSE.md",
        },
        "counts": {
            "corpus_docs": len(corpus),
            "queries": len(queries),
            "test_queries": len(test_qrels),
            "test_judgments": judgments,
            "train_queries": len(qrels_by_split["train"]),
        },
        "splits": {"evaluated": "dev（= BEIR 的 qrels/test.tsv）", "train": "claims_train",
                   "note": TEST_SPLIT_NOTE},
        "transform": "只做格式归一：每篇摘要写成 Markdown（标题→一级标题，摘要句子按原文拼接），"
                     "不删句、不改词、不做任何摘要或改写；qrels 由 cited_doc_ids 反推为 doc 级二元判断",
        "files": {},
    }
    for relative in ("queries.jsonl", "qrels/test.tsv", "qrels/train.tsv"):
        path = out_dir / relative
        manifest["files"][relative] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}
    doc_sizes = sorted((path.stat().st_size for path in docs_dir.glob("*.md")))
    manifest["files"]["docs/"] = {"count": len(doc_sizes), "bytes": sum(doc_sizes),
                                  "bytes_median": doc_sizes[len(doc_sizes) // 2] if doc_sizes else 0}
    (out_dir / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    return manifest


def verify(out_dir: Path = DATASET_DIR) -> int:
    manifest_path = out_dir / "sources.json"
    if not manifest_path.exists():
        print("没有 sources.json：先跑一次 fetch（不要手搓产物）", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for relative, expected in manifest["files"].items():
        if relative.endswith("/"):
            count = len(list((out_dir / relative).glob("*")))
            if count != expected["count"]:
                problems.append(f"{relative} 文件数 {count} != {expected['count']}")
            continue
        path = out_dir / relative
        if not path.exists():
            problems.append(f"{relative} 不存在")
        elif sha256_of(path) != expected["sha256"]:
            problems.append(f"{relative} sha256 不一致（被改过？）")
    if problems:
        print("校验失败：" + chr(10) + "  " + (chr(10) + "  ").join(problems), file=sys.stderr)
        return 1
    counts = manifest["counts"]
    print(f"OK：{counts['corpus_docs']} 篇 / {counts['queries']} 题；"
          f"测试集 {counts['test_queries']} 题、{counts['test_judgments']} 条判断")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="取回 SciFact 并做格式归一")
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="已解压的原始目录（含 corpus.jsonl）")
    parser.add_argument("--archive", type=Path, default=None, help="本地 tar.gz（跳过下载）")
    parser.add_argument("--offline", action="store_true", help="只用本地归档/已解压目录")
    parser.add_argument("--force", action="store_true", help="重写 docs/（默认幂等跳过）")
    parser.add_argument("--verify", action="store_true", help="只校验本地产物与 manifest")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify:
        return verify(args.out)

    data_dir = args.data_dir
    if data_dir is None:
        archive = args.archive
        if archive is None:
            archive = CACHE_DIR / "data.tar.gz"
            if not args.offline and not (archive.exists() and sha256_of(archive)):
                print(f"下载 {RELEASE_URL}", file=sys.stderr)
                archive = download(RELEASE_URL, archive)
        data_dir = extract(archive, CACHE_DIR / "extracted")

    manifest = build(data_dir, args.out, force=args.force)
    counts = manifest["counts"]
    print(f"完成：{counts['corpus_docs']} 篇摘要 → {args.out / 'docs'}")
    print(f"      题集 {counts['queries']} 题（train {counts['train_queries']} + dev {counts['test_queries']}）")
    print(f"      qrels：test {counts['test_judgments']} 条 / train {counts['train_queries']} 题")
    print("      下一步：RAG_COLLECTION=scifact_v1 python scripts/ingest.py --dir "
          f"{args.out.relative_to(ROOT)}/docs …")
    return verify(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
