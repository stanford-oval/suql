# Release 1.1.6 (4/29/24)

`faiss_embedding.py` now by default stores a cached embedding at the user's cache directory (determined by `platformdirs`).

If `cache_embedding` is enabled (which is turned on by default), this file computes a hash of the free text values. The next time when the server is run, if the database values remains unchanged, this file will directly use the cached embeddings. If there are changes to the underlying values, this file will recompute the embeddings.

Check API documentation for the `add` function at [here](https://stanford-oval.github.io/suql/suql/faiss_embedding.html#suql.faiss_embedding.MultipleEmbeddingStore.add).

# Release 1.1.5 (4/25/24)

Bug fix for #15.