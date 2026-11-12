"""`ContextBlock.chunk_ids` 必须是「命中的子块 id」，不许拿父块 id 冒充（N60）。

现场：`assembler.py` 里 `chunk_ids=[item["parent_id"] for item in group]` ——
紧跟在 `parent_ids=[item["parent_id"] ...]` 后面，一眼看去像对的，实际是把同一份东西
抄了两遍。后果是 `/ask` 响应与日志里的「引用到哪一块」全指到父块：
拿 `chunk_ids` 回查 Qdrant（子块点）必然查不到，因为父块根本不是 Qdrant 里的点。

上游也有半截责任：`retrieve_with_meta()` 组装 `RerankHit` 时写了
`chunk_id=hit.parent_id`，把路由层（`RouteHit.chunk_id`）拿到的真子块 id 直接丢了。
两处都要修，所以这个文件同时锁两层：

1. 组装器：命中带 `chunk_id` 就用它；没带（老调用方 / 只按父块建的词面索引）才退回
   父块 id，并且**留痕**（`stats["chunk_ids_from_parent"]`）—— 不许悄悄混过去；
2. 检索层：`FusedHit` 只有父块 id，子块 id 必须从路由结果里捞回来。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.context.assembler import assemble  # noqa: E402


class FakeCounter:
    def count(self, text):
        return max(0, len(text) // 4)

    def truncate(self, text, max_tokens):
        return text[: max_tokens * 4]


def _parent(pid, doc, index, text, section=("A",)):
    return {
        "id": pid, "parent_id": pid, "document_id": doc, "parent_index": index,
        "content": text, "text": text, "section_path": list(section),
        "title": doc, "source_uri": "file.md",
    }


PARENTS = {
    "par_0": _parent("par_0", "doc_1", 0, "第一段正文。"),
    "par_1": _parent("par_1", "doc_1", 1, "第二段正文。"),
}


class AssemblerChunkIdsTest(unittest.TestCase):
    def assemble(self, hits):
        return assemble(hits, PARENTS, counter=FakeCounter())

    def test_chunk_ids_are_the_matched_child_ids(self):
        """命中带着子块 id 时，`chunk_ids` 就是它 —— 不是父块 id。"""
        ctx = self.assemble([{"parent_id": "par_0", "chunk_id": "chd_0_a", "score": 1.0}])
        block = ctx.blocks[0]
        self.assertEqual(block.parent_ids, ["par_0"])
        self.assertEqual(block.chunk_ids, ["chd_0_a"])

    def test_missing_chunk_id_falls_back_to_parent_and_is_counted(self):
        """老调用方只给父块 id：允许退回去，但必须留痕（不许静默混过去）。"""
        ctx = self.assemble([{"parent_id": "par_0", "score": 1.0}])
        self.assertEqual(ctx.blocks[0].chunk_ids, ["par_0"])
        self.assertEqual(ctx.stats["chunk_ids_from_parent"], 1,
                         "退回父块 id 的次数要能数出来，否则「混了多少」永远说不清")

    def test_merged_group_keeps_chunk_ids_in_parent_order(self):
        """邻接合并成一块后，`chunk_ids` 要与 `parent_ids` 一一对齐（同序）。"""
        ctx = self.assemble([
            {"parent_id": "par_1", "chunk_id": "chd_1_a", "score": 1.0},
            {"parent_id": "par_0", "chunk_id": "chd_0_a", "score": 0.5},
        ])
        self.assertEqual(len(ctx.blocks), 1, "同文档相邻父块应该合并成一块")
        block = ctx.blocks[0]
        self.assertEqual(block.parent_ids, ["par_0", "par_1"])
        self.assertEqual(block.chunk_ids, ["chd_0_a", "chd_1_a"],
                         "chunk_ids 必须跟着 parent_ids 的顺序走，不能按命中顺序")
        self.assertEqual(ctx.stats["chunk_ids_from_parent"], 0)

    def test_object_hits_expose_chunk_id(self):
        """`/ask` 传进来的是 `RerankHit` 对象（不是 dict），也要带上子块 id。"""
        from src.rag.retrieval.rerank import RerankHit

        hit = RerankHit(chunk_id="chd_0_b", parent_id="par_0", text="第一段正文。",
                        score=1.0, rank=1, rerank_score=0.9)
        ctx = self.assemble([hit])
        self.assertEqual(ctx.blocks[0].chunk_ids, ["chd_0_b"])


class PipelineChunkIdTest(unittest.TestCase):
    """检索层：`FusedHit` 只有父块 id，子块 id 要从路由结果里捞回来。"""

    def _retriever(self, hits_by_route):
        from src.rag.retrieval.routes import RouteHit, RouteResult

        class _Retriever:
            def retrieve(self, query, k, query_en=None):
                from src.rag.retrieval.orchestrator import RetrievalResult

                routes = {
                    name: RouteResult(route=name, enabled=True, candidates=len(hits),
                                      hits=[RouteHit(parent_id=p, chunk_id=c, score=s, route=name)
                                            for p, c, s in hits])
                    for name, hits in hits_by_route.items()
                }
                return RetrievalResult(query_zh=query, query_en=query_en, k=k, routes=routes)

        return _Retriever()

    def test_rerank_hit_carries_the_child_chunk_id(self):
        from src.rag.ask.pipeline import retrieve_with_meta
        from src.rag.ingestion.tokenizer import TokenCounter

        retriever = self._retriever({
            "dense_zh": [("par_0", "chd_0_a", 0.9)],
            "lexical_en": [("par_0", None, 0.4)],
        })
        outcome = retrieve_with_meta("问题", retriever=retriever, parents=PARENTS,
                                     counter=TokenCounter(), reranker=None, pool=5,
                                     rerank=False)
        hit = outcome.hits[0]
        self.assertEqual(hit.parent_id, "par_0")
        self.assertEqual(hit.chunk_id, "chd_0_a",
                         "子块 id 在路由层拿到了就必须带下去，不能拿父块 id 顶替")

    def test_route_without_child_id_keeps_parent_id(self):
        """只有父块 id 的路（词面索引建在父块上）退回父块 id，不虚构子块 id。"""
        from src.rag.ask.pipeline import retrieve_with_meta
        from src.rag.ingestion.tokenizer import TokenCounter

        retriever = self._retriever({"lexical_en": [("par_0", None, 0.4)]})
        outcome = retrieve_with_meta("问题", retriever=retriever, parents=PARENTS,
                                     counter=TokenCounter(), reranker=None, pool=5,
                                     rerank=False)
        self.assertEqual(outcome.hits[0].chunk_id, "par_0")

    def test_context_reports_child_chunk_ids_end_to_end(self):
        """`answer_query` 走完组装后，块上的 `chunk_ids` 必须是子块 id。"""
        from src.rag.ask.pipeline import answer_query
        from src.rag.ingestion.tokenizer import TokenCounter

        retriever = self._retriever({"dense_zh": [("par_0", "chd_0_a", 0.9)]})
        result = answer_query("问题", retriever=retriever, parents=PARENTS,
                              counter=TokenCounter(), reranker=None, ask=None,
                              rerank=False, k=1, pool=5)
        block = result.blocks[0]
        self.assertEqual(block["parent_ids"], ["par_0"])
        self.assertEqual(block["chunk_ids"], ["chd_0_a"])


if __name__ == "__main__":
    unittest.main()
