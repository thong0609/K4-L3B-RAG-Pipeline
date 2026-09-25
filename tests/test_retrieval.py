from copy import deepcopy
import unicodedata

import pytest

from src.contracts import validate_search_results
from src import task5_semantic_search as dense
from src import task6_lexical_search as lexical
from src.task7_reranking import rerank_rrf


def chunk(item_id, content):
    return {
        "id": item_id,
        "content": content,
        "metadata": {
            "source": f"{item_id}.md",
            "title": "Tuyển sinh",
            "doc_type": "legal",
            "url": None,
            "chunk_index": 0,
        },
    }


def unexpected(*args, **kwargs):
    pytest.fail("Unexpected provider call")


@pytest.mark.parametrize("search", [dense.semantic_search, lexical.lexical_search])
@pytest.mark.parametrize("query,top_k", [("  ", 5), ("học phí", 0), ("học phí", -2)])
def test_empty_requests_do_not_call_providers(monkeypatch, search, query, top_k):
    monkeypatch.setattr(dense, "get_collection", unexpected)
    monkeypatch.setattr(dense, "embed_texts", unexpected)
    monkeypatch.setattr(lexical, "get_collection", unexpected)
    assert search(query, top_k) == []


def test_dense_empty_collection_skips_embedding(monkeypatch):
    class EmptyCollection:
        def count(self):
            return 0

        query = unexpected

    monkeypatch.setattr(dense, "get_collection", EmptyCollection)
    monkeypatch.setattr(dense, "embed_texts", unexpected)
    assert dense.semantic_search("học phí") == []


def test_dense_preserves_raw_cosine_and_caps_query_size(monkeypatch):
    items = [chunk("a", "Học phí"), chunk("b", "Tuyển sinh")]

    class Collection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs == {
                "query_embeddings": [[1.0, 0.0]],
                "n_results": 2,
                "include": ["documents", "metadatas", "distances"],
            }
            return {
                "ids": [["a", "b"]],
                "documents": [[item["content"] for item in items]],
                "metadatas": [[item["metadata"] for item in items]],
                "distances": [[1.25, 0.1]],
            }

    def embed(texts, is_query=False):
        assert is_query is True
        assert texts == ["học phí"]
        return [[1.0, 0.0]]

    monkeypatch.setattr(dense, "get_collection", Collection)
    monkeypatch.setattr(dense, "embed_texts", embed)
    output = dense.semantic_search("  học phí  ", top_k=10)
    validate_search_results(output, top_k=10, expected_method="dense")
    assert [item["id"] for item in output] == ["b", "a"]
    assert [item["score"] for item in output] == pytest.approx([0.9, -0.25])
    assert output[1]["metadata"] == items[0]["metadata"]


def test_dense_deduplicates_and_limits_results(monkeypatch):
    class Collection:
        def count(self):
            return 3

        def query(self, **kwargs):
            return {
                "ids": [["a", "b", "a"]],
                "documents": [["A", "B", "A"]],
                "metadatas": [[chunk("a", "A")["metadata"]] * 3],
                "distances": [[0.5, 0.3, 0.1]],
            }

    monkeypatch.setattr(dense, "get_collection", Collection)
    monkeypatch.setattr(dense, "embed_texts", lambda texts, is_query=False: [[1.0]])
    output = dense.semantic_search("test", top_k=1)
    assert len(output) == 1
    assert output[0]["id"] == "a"
    assert output[0]["score"] == pytest.approx(0.9)


def test_bm25_normalizes_vietnamese_and_filters_nonmatching_chunks(monkeypatch):
    corpus = [
        chunk("a", "HỌC PHÍ: đóng theo học kỳ."),
        chunk("b", "Phương thức tuyển sinh"),
        chunk("c", "Thông tin ký túc xá"),
    ]
    original = deepcopy(corpus)
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    monkeypatch.setattr(lexical, "get_collection", unexpected)
    output = lexical.lexical_search(unicodedata.normalize("NFD", "HỌC PHÍ!"))
    validate_search_results(output, expected_method="bm25")
    assert [item["id"] for item in output] == ["a"]
    assert output[0]["score"] > 0
    assert corpus == original
    assert lexical.lexical_search("astronaut") == []
    assert lexical.lexical_search("!!!") == []


def test_bm25_keeps_zero_score_matches(monkeypatch):
    monkeypatch.setattr(lexical, "CORPUS", [
        chunk("unrelated", "library hours"),
        chunk("relevant", "tuition fee"),
    ])
    output = lexical.lexical_search("tuition")
    assert [item["id"] for item in output] == ["relevant"]
    assert output[0]["score"] == 0.0


@pytest.mark.parametrize("corpus", [[], [chunk("a", "!!!")]])
def test_bm25_empty_vocabulary(monkeypatch, corpus):
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    assert lexical.build_bm25_index(corpus) is None
    assert lexical.lexical_search("tuition") == []


def test_bm25_handles_empty_token_chunk_duplicates_and_top_k(monkeypatch):
    corpus = [chunk("a", "tuition"), chunk("a", "tuition"),
              chunk("b", "tuition fee"), chunk("c", "!!!")]
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    output = lexical.lexical_search("tuition", top_k=10)
    validate_search_results(output, expected_method="bm25")
    assert {item["id"] for item in output} == {"a", "b"}
    assert lexical.lexical_search("tuition", top_k=1) == output[:1]


def test_bm25_cache_refreshes_when_content_changes():
    corpus = [chunk("a", "tuition"), chunk("b", "library")]
    first = lexical.build_bm25_index(corpus)
    assert lexical.build_bm25_index(deepcopy(corpus)) is first
    corpus[0]["content"] = "admissions"
    second = lexical.build_bm25_index(corpus)
    assert second is not first
    assert "admissions" in second.idf
    assert "tuition" not in second.idf


def test_shared_collection_flows_through_dense_bm25_and_rrf(monkeypatch):
    corpus = [chunk("a", "tuition fee"), chunk("b", "library hours")]

    class Collection:
        def count(self):
            return len(corpus)

        def get(self, **kwargs):
            assert kwargs == {"include": ["documents", "metadatas"]}
            return {
                "ids": [item["id"] for item in corpus],
                "documents": [item["content"] for item in corpus],
                "metadatas": [item["metadata"] for item in corpus],
            }

        def query(self, **kwargs):
            return {
                "ids": [["b", "a"]],
                "documents": [[item["content"] for item in reversed(corpus)]],
                "metadatas": [[item["metadata"] for item in reversed(corpus)]],
                "distances": [[0.1, 0.2]],
            }

    monkeypatch.setattr(dense, "get_collection", Collection)
    monkeypatch.setattr(dense, "embed_texts", lambda texts, is_query=False: [[1.0]])
    monkeypatch.setattr(lexical, "get_collection", Collection)
    monkeypatch.setattr(lexical, "CORPUS", None)
    dense_results = dense.semantic_search("tuition")
    sparse_results = lexical.lexical_search("tuition")
    fused = rerank_rrf([dense_results, sparse_results])
    validate_search_results(fused, expected_method="hybrid")
    assert fused[0]["id"] == "a"
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert dense_results[0]["score"] == pytest.approx(0.9)
    # Same count, new content/metadata: the next search must see the new snapshot.
    corpus[0] = chunk("new", "admissions")
    assert lexical.lexical_search("tuition") == []
    assert lexical.lexical_search("admissions")[0]["id"] == "new"


@pytest.mark.parametrize("search,module", [
    (dense.semantic_search, dense), (lexical.lexical_search, lexical)
])
def test_provider_errors_are_not_silently_hidden(monkeypatch, search, module):
    def unavailable():
        raise RuntimeError("collection unavailable")

    monkeypatch.setattr(lexical, "CORPUS", None)
    monkeypatch.setattr(module, "get_collection", unavailable)
    with pytest.raises(RuntimeError, match="collection unavailable"):
        search("tuition")
