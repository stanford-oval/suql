import numpy as np

import suql.faiss_embedding as faiss_embedding
from suql.faiss_embedding import EmbeddingStore, _make_id_selector


class ModernFaiss:
    received = None

    @classmethod
    def IDSelectorBatch(cls, indices):
        cls.received = indices
        return ("modern", indices.tolist())


class LegacyFaiss:
    received = None

    @staticmethod
    def swig_ptr(indices):
        return ("pointer", indices)

    @classmethod
    def IDSelectorBatch(cls, *args):
        if len(args) == 1:
            raise TypeError("missing 1 required positional argument: 'indices'")
        cls.received = args
        return ("legacy", args[0])


def test_make_id_selector_uses_array_wrapper_when_available():
    selector = _make_id_selector(ModernFaiss, [3, 1, 3])

    assert selector == ("modern", [3, 1, 3])
    assert ModernFaiss.received.dtype == np.int64
    assert ModernFaiss.received.flags.c_contiguous


def test_make_id_selector_falls_back_to_swig_constructor():
    selector = _make_id_selector(LegacyFaiss, [4, 2])

    assert selector == ("legacy", 2)
    size, pointer = LegacyFaiss.received
    assert size == 2
    assert pointer[0] == "pointer"
    assert pointer[1].dtype == np.int64
    assert pointer[1].flags.c_contiguous


def test_make_id_selector_filters_a_real_faiss_search():
    import faiss

    vectors = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    query = np.asarray([[1, 0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    selector = _make_id_selector(faiss, [0, 2])
    _, indices = index.search(
        query, 3, params=faiss.SearchParametersIVF(sel=selector)
    )

    assert indices.tolist() == [[0, 2, -1]]


def test_embedding_store_filtered_search_paths(monkeypatch):
    import faiss

    vectors = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    store = EmbeddingStore.__new__(EmbeddingStore)
    store.faiss = faiss
    store.embeddings = index
    store.id2document = {10: [0, 2]}
    store.document2embedding = {0: [0], 1: [1], 2: [2]}
    store.embedding2document = {0: 0, 1: 1, 2: 2}
    store.document2id = {0: 10, 1: 20, 2: 10}
    store.all_free_text = ["positive", "orthogonal", "negative"]
    monkeypatch.setattr(
        faiss_embedding,
        "embed_query",
        lambda query: np.asarray([[1, 0]], dtype=np.float32),
    )
    monkeypatch.setattr(faiss_embedding, "MULTIPLE_COLUMN_SEL", 2)

    assert store.dot_product([10], "query", 2) == [
        (10, ["positive"]),
        (10, ["negative"]),
    ]
    assert store.dot_product_with_value([10], "query") == [
        (10, 1.0, ["positive"]),
        (10, -1.0, ["negative"]),
    ]
