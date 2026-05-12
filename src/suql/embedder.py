"""
Embedder abstraction for SUQL.

A single ``Embedder`` instance owns the model + tokenizer and exposes a
stable interface (``name``, ``dimension``, ``embed_query``,
``embed_documents``) so the rest of the codebase does not need to know
which backbone is in use.
"""

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    def embed_query(self, query: str) -> np.ndarray:
        """Return a ``(1, dimension)`` float32 array for a single query."""

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Return an ``(N, dimension)`` float32 array for N documents."""


class BGELargeEnV15Embedder:
    """Default SUQL embedder: BAAI/bge-large-en-v1.5 via FlagEmbedding."""

    name = "BAAI/bge-large-en-v1.5"
    dimension = 1024

    def __init__(self, use_fp16: bool = True):
        from FlagEmbedding import FlagModel

        self._model = FlagModel(
            self.name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=use_fp16,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode_queries([query])

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        return self._model.encode(documents)


class SentenceTransformersEmbedder:
    """
    Generic sentence-transformers backend. Used for Qwen3-Embedding and any
    other HF model that ships an ST config.

    The query instruction is applied via ST's ``prompt_name="query"``
    mechanism when the model defines a ``query`` prompt; otherwise it is
    prepended manually if ``query_instruction`` is set.
    """

    def __init__(
        self,
        model_name: str,
        dimension: int,
        query_instruction: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.dimension = dimension
        self._query_instruction = query_instruction

        self._model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs or {},
            tokenizer_kwargs=tokenizer_kwargs or {},
        )
        self._has_query_prompt = "query" in (self._model.prompts or {})

    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        if is_query and self._has_query_prompt:
            vectors = self._model.encode(texts, prompt_name="query")
        elif is_query and self._query_instruction:
            prompted = [f"{self._query_instruction}{t}" for t in texts]
            vectors = self._model.encode(prompted)
        else:
            vectors = self._model.encode(texts)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        return self._encode(list(documents), is_query=False)


class Qwen3Embedder(SentenceTransformersEmbedder):
    """
    Qwen3-Embedding family (0.6B / 4B / 8B). Defaults to the 4B variant.

    Native dimensions (Matryoshka allows truncation but we keep native here):
      0.6B -> 1024, 4B -> 2560, 8B -> 4096.
    """

    _NATIVE_DIMS = {
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "Qwen/Qwen3-Embedding-4B": 2560,
        "Qwen/Qwen3-Embedding-8B": 4096,
    }

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        if model_name not in self._NATIVE_DIMS:
            raise ValueError(
                f"Unknown Qwen3 embedding model: {model_name}. "
                f"Expected one of {sorted(self._NATIVE_DIMS)}."
            )
        super().__init__(
            model_name=model_name,
            dimension=self._NATIVE_DIMS[model_name],
            tokenizer_kwargs={"padding_side": "left"},
        )


def default_embedder() -> Embedder:
    """Return SUQL's historical default (BGE-large-en-v1.5)."""
    return BGELargeEnV15Embedder()
