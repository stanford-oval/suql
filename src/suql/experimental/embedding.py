from suql.faiss_embedding import MultipleEmbeddingStore

embedding_store = MultipleEmbeddingStore()
embedding_store.add(table_name="log_normal",
                    primary_key_field_name="line_id",
                    free_text_field_name="content",
                    db_name="postgres",
                    user="select_user",
                    password="select_user")

host = "127.0.0.1"
port = 8501
embedding_store.start_embedding_server(host=host, port=port)
