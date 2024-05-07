# stores a list of embeddings for reviews
import hashlib
from collections import OrderedDict

import os
import faiss
import hashlib
import pickle
from FlagEmbedding import FlagModel
from flask import Flask, request
from tqdm import tqdm
from platformdirs import user_cache_dir

from suql.postgresql_connection import execute_sql
from suql.utils import chunk_text

# change this line for custom embedding model
# embedding model output dimension
EMBEDDING_DIMENSION = 1024

# number of rows to consider for multi-column operations
MULTIPLE_COLUMN_SEL = 1000

# currently using https://huggingface.co/BAAI/bge-large-en-v1.5
# change this line for custom embedding model
model = FlagModel(
    "BAAI/bge-large-en-v1.5",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,
)  # Setting use_fp16 to True speeds up computation with a slight performance degradation


def embed_query(query):
    """
    Embed a query for dot product matching
    """
    # change this line for custom embedding model
    q_embedding = model.encode_queries([query])
    return q_embedding


def embed_documents(documents):
    """
    Embed a list of docuemnts to store in vector store
    """
    # change this line for custom embedding model
    embeddings = model.encode(documents)
    return embeddings


def compute_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def consistent_tuple_hash(tuple_input):
    # Serialize the tuple to bytes using pickle
    tuple_bytes = pickle.dumps(tuple_input, protocol=pickle.HIGHEST_PROTOCOL)
    # Return the SHA-256 hash of the serialized bytes
    return hashlib.sha256(tuple_bytes).hexdigest()


# A set that also preserves insertion order
class OrderedSet:
    def __init__(self, iterable=None):
        self.items = OrderedDict()
        if iterable:
            self.add_all(iterable)

    def add(self, item):
        self.items[item] = None

    def add_all(self, iterable):
        for item in iterable:
            self.add(item)

    def union(self, other):
        # Create a new OrderedSet for the union
        union_set = OrderedSet(self)
        union_set.add_all(other)
        return union_set

    def __iter__(self):
        return iter(self.items)

    def __contains__(self, item):
        return item in self.items

    def __len__(self):
        return len(self.items)


def compute_top_similarity_documents(documents, query, chunking_param=0, top=3):
    """
    Directly call the model to compute the top documents based on
    dot product with query
    """
    chunked_documents_tuple = [
        (i, doc)
        for (i, document) in enumerate(documents)
        for doc in chunk_text(document, k=chunking_param, use_spacy=True)
    ]
    chunked_documents_embeddings = embed_documents(
        list(map(lambda x: x[1], chunked_documents_tuple))
    )
    embeddings = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    embeddings.add(chunked_documents_embeddings)

    _, I = embeddings.search(embed_query(query), len(chunked_documents_embeddings))
    # attempt to re-construct the top queries, keeping going untill we actually get all top
    iter_chunk = top * 2
    doc_ids = OrderedSet()
    for i in range(0, len(I[0]), iter_chunk):
        doc_ids = doc_ids.union(
            OrderedSet(chunked_documents_tuple[i][0] for i in I[0][i : i + iter_chunk])
        )
        if len(doc_ids) >= min(top, len(documents)):
            return [documents[index] for index in list(doc_ids)[:top]]
    return [documents[index] for index in list(doc_ids)[:top]]


def construct_reverse_dict(res_individual_id, id_list):
    res = {}
    for individual_id, id in zip(res_individual_id, id_list):
        if individual_id not in res:
            res[individual_id] = []
        res[individual_id].append(id)
    return res


class EmbeddingStore:
    def __init__(
        self,
        table_name,
        primary_key_field_name,
        free_text_field_name,
        db_name="",
        user="select_user",
        password="select_user",
        chunking_param=0,
        cache_embedding=True,
        force_recompute=False
    ) -> None:
        # stores three lists:
        # 1. PSQL primary key for each row
        # 2. list of strings in this field
        # 3. matrix of embedding stored on GPU, each chunk is a row
        self.psql_row_ids = []
        self.all_free_text = []
        self.embeddings = None
        
        assert chunking_param >= 0
        self.chunking_param = chunking_param
        self.chunked_text = []

        # stores a table for bidirectional mapping (for each PSQL table and free text field) between
        # PSQL row ID <-> corresponding indexs for strings <-> corresponding embeddings stored on GPU
        self.document2embedding = {}
        self.embedding2document = {}
        self.id2document = {}
        self.document2id = {}

        # stores PSQL login credentails
        self.user = user
        self.password = password
        
        # store caching flag
        assert type(cache_embedding) == bool
        self.cache_embedding = cache_embedding
        assert type(force_recompute) == bool
        self.force_recompute = force_recompute

        self.initialize_from_sql(
            table_name, primary_key_field_name, free_text_field_name, db_name
        )
        print(f"initializing embeddings for DB: {db_name}; TABLE: {table_name}; FREE_TEXT_FIELD: {free_text_field_name}")
        self.initialize_embedding()

    def initialize_from_sql(
        self, table_name, primary_key_field_name, free_text_field_name, db_name
    ):
        sql = "SELECT {}, {} FROM {}".format(
            primary_key_field_name, free_text_field_name, table_name
        )
        if db_name == "":
            res = execute_sql(sql, user=self.user, password=self.password)[0]
        else:
            res = execute_sql(
                sql, database=db_name, user=self.user, password=self.password
            )[0]

        print(
            "initializing storage and mapping for {} <-> {}".format(
                primary_key_field_name, free_text_field_name
            )
        )
        document_counter = 0
        embedding_counter = 0

        for id, free_texts in tqdm(res):
            self.psql_row_ids.append(id)
            if free_texts is None:
                free_texts = ""

            if type(free_texts) == str:
                free_texts = [free_texts]

            if type(free_texts) == list:
                self.all_free_text.extend(free_texts)
                self.id2document[id] = [
                    num
                    for num in range(
                        document_counter, document_counter + len(free_texts)
                    )
                ]
                for num in range(document_counter, document_counter + len(free_texts)):
                    self.document2id[num] = id

                # chunk the text and prepare the two mappings
                for document in free_texts:
                    chunked_text = chunk_text(
                        document, k=self.chunking_param, use_spacy=True
                    )
                    document_embedding_len = len(chunked_text)
                    self.chunked_text.extend(chunked_text)
                    self.document2embedding[document_counter] = [
                        num
                        for num in range(
                            embedding_counter,
                            embedding_counter + document_embedding_len,
                        )
                    ]
                    for num in range(
                        embedding_counter, embedding_counter + document_embedding_len
                    ):
                        self.embedding2document[num] = document_counter
                    embedding_counter += document_embedding_len
                    document_counter += 1

            else:
                raise ValueError("Expecting type Str")

    def compute_hash(self):
        # Convert lists to tuples for hashing
        psql_row_ids_tuple = tuple(self.psql_row_ids)
        all_free_text_tuple = tuple(self.all_free_text)
        
        # Create a combined tuple of all objects
        combined_data = (psql_row_ids_tuple, all_free_text_tuple, self.chunking_param)
        
        # Compute and return the hash of the combined tuple
        return consistent_tuple_hash(combined_data)

    def initialize_embedding(self):
        hash = self.compute_hash()
        _user_cache_dir = user_cache_dir('suql')
        faiss_cache_location = os.path.join(_user_cache_dir, f'{hash}.faiss_index')
        if (os.path.exists(faiss_cache_location) and not self.force_recompute):
            try:
                print(f"initializing from existing faiss embedding index at {faiss_cache_location}")
                self.embeddings = faiss.read_index(faiss_cache_location)
                return
            except Exception:
                print(f"reading {faiss_cache_location} failed. Re-computing embeddings")
        
        self.embeddings = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        indexs = embed_documents(self.chunked_text)
        self.embeddings.add(indexs)
        
        print(f"writing computed faiss embedding to {faiss_cache_location}")
        os.makedirs(_user_cache_dir, exist_ok=True)
        faiss.write_index(self.embeddings, faiss_cache_location)

    def dot_product(self, id_list, query, top, individual_id_list=[]):
        # given a list of id and a particular query, return the top ids and documents according to similarity score ranking

        if individual_id_list == []:
            document_indices = [
                item
                for sublist in map(lambda x: self.id2document[x], id_list)
                for item in sublist
            ]
        else:
            document_indices = [
                item
                for sublist in map(lambda x: self.id2document[x], individual_id_list)
                for item in sublist
            ]
        embedding_indices = [
            item
            for sublist in map(lambda x: self.document2embedding[x], document_indices)
            for item in sublist
        ]

        query_embedding = embed_query(query)

        sel = faiss.IDSelectorBatch(embedding_indices)
        if top < 0:
            D, I = self.embeddings.search(
                query_embedding,
                len(embedding_indices),
                params=faiss.SearchParametersIVF(sel=sel),
            )
        else:
            if top > self.embeddings.ntotal:
                top = self.embeddings.ntotal
            D, I = self.embeddings.search(
                query_embedding, top, params=faiss.SearchParametersIVF(sel=sel)
            )

        embeddings_indices_max = I[0]

        # append the top documents as result
        # there exists repeated documents when mapping embedding_indices_max directly to document_id
        if top > 0:
            res_document_ids = OrderedSet()
            i = 0
            # append 15 elements together at the same time for efficiency
            # 15 is a performance parameter that can be tuned
            while len(res_document_ids) < top and i <= len(embeddings_indices_max):
                res_document_ids = res_document_ids.union(
                    OrderedSet(
                        [
                            self.embedding2document[index]
                            for index in embeddings_indices_max[i : i + 15]
                        ]
                    )
                )
                i += 15
            res_document_ids = list(res_document_ids)[:top]
        else:
            res_document_ids = [
                self.embedding2document[index] for index in embeddings_indices_max
            ]

        if individual_id_list == []:
            return [
                (self.document2id[index], [self.all_free_text[index]])
                for index in res_document_ids
            ]
        else:
            reverse_dict = construct_reverse_dict(individual_id_list, id_list)
            # this reverse dict would map individual ids to the special join id
            return [
                (reverse_dict[self.document2id[index]], [self.all_free_text[index]])
                for index in res_document_ids
            ]

    def dot_product_with_value(self, id_list, query, individual_id_list=[]):
        if individual_id_list == []:
            individual_id_list = id_list
        document_indices = [
            item
            for sublist in map(lambda x: self.id2document[x], individual_id_list)
            for item in sublist
        ]
        embedding_indices = [
            item
            for sublist in map(lambda x: self.document2embedding[x], document_indices)
            for item in sublist
        ]

        # chunking param = 0 makes sure that we don't chunk the query
        # this is actually a 2-D array, matching what faiss expects
        query_embedding = embed_query(query)

        sel = faiss.IDSelectorBatch(embedding_indices)
        D, I = self.embeddings.search(
            query_embedding,
            MULTIPLE_COLUMN_SEL,
            params=faiss.SearchParametersIVF(sel=sel),
        )
        embedding_indices = I[0]
        dot_products = D[0]

        # when joins are invovled, a new id field will be created, stored in id_list
        # individual_id_list instead would store the corresponding column-specific id for this predicate
        # so, first build a mapping from `individual_id_list` to `id_list`
        if individual_id_list:
            individual2id_list_mapping = construct_reverse_dict(
                individual_id_list, id_list
            )
            res = [
                (
                    individual_id,
                    dot_product,
                    [self.all_free_text[self.embedding2document[indice]]],
                )
                for indice, dot_product in zip(embedding_indices, dot_products)
                for individual_id in individual2id_list_mapping[
                    self.document2id[self.embedding2document[indice]]
                ]
            ]
        else:
            res = [
                (
                    self.document2id[self.embedding2document[indice]],
                    dot_product,
                    [self.all_free_text[self.embedding2document[indice]]],
                )
                for indice, dot_product in zip(embedding_indices, dot_products)
            ]

        return res


class MultipleEmbeddingStore:
    def __init__(self) -> None:
        # table name -> free text field name -> EmbeddingStore
        self.mapping = {}

    def add(
        self,
        table_name: str,
        primary_key_field_name: str,
        free_text_field_name: str,
        db_name: str,
        user="select_user",
        password="select_user",
        chunking_param=0,
        cache_embedding=True,
        force_recompute=False
    ):
        """
        Add a free text field to the SUQL embedding store to make it
        available to the compiler.

        # Parameters:

        `table_name` (str): Table name of the free text field.
        
        `primary_key_field_name` (str): Primary key field name of the table `table_name`.
        
        `free_text_field_name` (str): Free text field name to be embedded.
        
        `db_name` (str): PostgreSQL database name of the table `table_name`.
        
        `user` (str, optional): User name with `SELECT` privilege on the table `table_name`.
        Defaults to "select_user".
        
        `password` (str, optional): Password for the above user with `SELECT` privilege
        on the table `table_name`. Defaults to "select_user".
        
        `chunking_param` (int, optional): Chunking parameter for embedding.
        It denotes how many tokens (determined by `en_core_web_sm`) to chunk to for each
        free text value. Defaults to 0, which denotes no chunking.
        
        `cache_embedding` (bool, optional): Whether to cache embeddings to disk. If cached,
        this file computes a hash of the free text values. If the database values remains
        unchanged, this file will directly use the cached embeddings. If there are changes
        to the underlying values, this file will recompute the embeddings.
        Defaults to True.
        
        `force_recompute` (bool, optional): Whether to force recomputing embeddings.
        If set to True, this file will re-compute the embedding even if a cache exists
        for the underlying values. Defaults to False.
        """
        if (
            table_name in self.mapping
            and free_text_field_name in self.mapping[table_name]
        ):
            print(
                "Table {} for free text field {} already in storage. Negelecting...".format(
                    table_name, free_text_field_name
                )
            )
            return
        if table_name not in self.mapping:
            self.mapping[table_name] = {}
        self.mapping[table_name][free_text_field_name] = EmbeddingStore(
            table_name,
            primary_key_field_name,
            free_text_field_name,
            db_name,
            user=user,
            password=password,
            chunking_param=chunking_param,
            cache_embedding=cache_embedding,
            force_recompute=force_recompute
        )

    def retrieve(self, table_name, free_text_field_name):
        return self.mapping[table_name][free_text_field_name]

    def _dot_product(self, id_list, field_query_list, top, single_table):
        # with joins, `field_query_list` stores the table and free text field as a tuple
        if len(id_list) == 0:
            return []

        # optimization for a single column - an easier case
        if len(field_query_list) == 1:
            free_text_field_table, free_text_field_name = field_query_list[0][0]
            query = field_query_list[0][1]

            if single_table:
                return self.retrieve(
                    free_text_field_table, free_text_field_name
                ).dot_product(id_list, query, top)
            else:
                return self.retrieve(
                    free_text_field_table, free_text_field_name
                ).dot_product(
                    id_list["_id_join"],
                    query,
                    top,
                    individual_id_list=id_list[free_text_field_table],
                )

        res = []
        for free_text_field, query in field_query_list:
            free_text_field_table, free_text_field_name = free_text_field
            if single_table:
                one_predicate_result = self.retrieve(
                    free_text_field_table, free_text_field_name
                ).dot_product_with_value(id_list, query)
            else:
                one_predicate_result = self.retrieve(
                    free_text_field_table, free_text_field_name
                ).dot_product_with_value(
                    id_list["_id_join"],
                    query,
                    individual_id_list=id_list[free_text_field_table],
                )

            # first time, simply overwrite res
            if not res:
                res = one_predicate_result
            else:
                res = [
                    (x[0], x[1] + y[1], x[2] + y[2])
                    for x in res
                    for y in one_predicate_result
                    if x[0] == y[0]
                ]

        sorted_res = sorted(res, key=lambda x: x[1], reverse=True)[:top]
        return list(map(lambda x: (x[0], x[2]), sorted_res))

    def dot_product(self, data):
        res = self._dot_product(
            data["id_list"], data["field_query_list"], data["top"], data["single_table"]
        )
        return res

    def start_embedding_server(self, host="127.0.0.1", port=8501):
        app = Flask(__name__)

        @app.route("/search", methods=["POST"])
        def search():
            data = request.get_json()
            res = {"result": self.dot_product(data)}

            return res

        app.run(host=host, port=port)


if __name__ == "__main__":
    embedding_store = MultipleEmbeddingStore()
    embedding_store.add(
        table_name="restaurants",
        primary_key_field_name="_id",
        free_text_field_name="reviews",
        db_name="restaurants",
        user="select_user",
        password="select_user",
    )
    embedding_store.add(
        table_name="restaurants",
        primary_key_field_name="_id",
        free_text_field_name="popular_dishes",
        db_name="restaurants",
        user="select_user",
        password="select_user",
    )

    # Set the server address, if running through command line
    host = "127.0.0.1"
    port = 8501

    embedding_store.start_embedding_server(host=host, port=port)
