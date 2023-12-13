# stores a list of embeddings for reviews

from tqdm import tqdm
import pymongo
from pathlib import Path
import sys
import torch
from flask import request, Flask
# Append parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from reviews_server import compute_sha256, _compute_single_embedding
from postgresql_connection import execute_sql

# cuda_ok = torch.cuda.is_available()
# if cuda_ok:
#     device = torch.device("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
client = pymongo.MongoClient('localhost', 27017)
cache_db = client['free_text_cache']['hash_to_embeddings']

# Set the server address
host = "127.0.0.1"
port = 8509
embedding_server_address = 'http://{}:{}'.format(host, port)
app = Flask(__name__)

class EmbeddingStore():
    def __init__(self, table_name, primary_key_field_name, free_text_field_name) -> None:
        # stores three lists:
        # 1. PSQL primary key for each row
        # 2. list of strings in this field
        # 3. matrix of embedding stored on GPU, each chunk is a row
        self.psql_row_ids = []
        self.all_free_text = []
        self.embeddings = None

        # stores a table for bidirectional mapping (for each PSQL table and free text field) between
        # PSQL row ID <-> corresponding indexs for strings <-> corresponding embeddings stored on GPU
        self.document2embedding = {}
        self.embedding2document = {}
        self.id2document = {}
        self.document2id = {}
        
        self.initialize_from_sql(table_name, primary_key_field_name, free_text_field_name)
        self.initialize_embedding()
        
    def initialize_from_sql(self, table_name, primary_key_field_name, free_text_field_name):
        sql = "SELECT {}, {} FROM {}".format(primary_key_field_name, free_text_field_name, table_name)
        res = execute_sql(sql)[0]
        
        print("initializing storage and mapping for {} <-> {}".format(primary_key_field_name, free_text_field_name))
        document_counter = 0
        for id, free_texts in res:
            self.psql_row_ids.append(id)
            if free_texts is None:
                free_texts = ""

            if type(free_texts) == str:
                free_texts = [free_texts]

            if type(free_texts) == list:
                self.all_free_text.extend(free_texts)
                self.id2document[id] = [num for num in range(document_counter, document_counter + len(free_texts))]
                for num in range(document_counter, document_counter + len(free_texts)):
                    self.document2id[num] = id
                
                document_counter += len(free_texts)
                
            else:
                # TODO handle easier cases with only one string as opposed to a list
                raise ValueError()

    def initialize_embedding(self):
        current_counter = 0
        
        print("initializing embeddings for all documents")
        for i, document in tqdm(list(enumerate(self.all_free_text))):
            found_res = cache_db.find_one({"_id": compute_sha256(document)})
            if found_res:
                document_embeddings = found_res["embeddings"]
                document_embeddings = torch.tensor(document_embeddings, device=device)
            else:
                document_embeddings = _compute_single_embedding([document])
            
            # initialize the first `existing_documents`
            if current_counter == 0:
                existing_embeddings = document_embeddings
            else:
                existing_embeddings = torch.cat((existing_embeddings, document_embeddings), dim=0)

            # stores the bidirectional mapping
            self.document2embedding[i] = [num for num in range(current_counter, current_counter + document_embeddings.size(0))]
            for num in range(current_counter, current_counter + document_embeddings.size(0)):
                self.embedding2document[num] = i
                
            current_counter += document_embeddings.size(0)
        
        self.embeddings = existing_embeddings
        # torch.cuda.empty_cache()
        
        # Calculate memory
        memory_bytes = self.embeddings.element_size() * self.embeddings.nelement()

        # Convert to a human-readable format and print
        if memory_bytes < 1024:
            print(f"embedding initialized and now using {memory_bytes} bytes")
        elif memory_bytes < 1024 ** 2:
            print(f"embedding initialized and now using {memory_bytes / 1024:.2f} KB")
        elif memory_bytes < 1024 ** 3:
            print(f"embedding initialized and now using {memory_bytes / (1024 ** 2):.2f} MB")
        else:
            print(f"embedding initialized and now using {memory_bytes / (1024 ** 3):.2f} GB")
    
    def dot_product(self, id_list, query, top):
        # given a list of id and a particular query, return the top ids and documents according to similarity score ranking
        
        document_indices = [item for sublist in map(lambda x: self.id2document[x], id_list) for item in sublist]
        embedding_indices = [item for sublist in map(lambda x: self.document2embedding[x], document_indices) for item in sublist]
        
        # chunking param = 0 makes sure that we don't chunk the query
        query_embedding = _compute_single_embedding([query], chunking_param=0)[0]
        
        dot_products = torch.sum(self.embeddings[torch.tensor(embedding_indices, device=device)] * query_embedding, dim=1)

        if top > 0:
            _, indices_max = torch.topk(dot_products, top)
        else:
            indices_max = torch.argsort(dot_products, descending=True)

        embeddings_indices_max = [embedding_indices[index] for index in indices_max.tolist()]
        return [(self.document2id[self.embedding2document[index]], [self.all_free_text[self.embedding2document[index]]]) for index in embeddings_indices_max]

    def dot_product_with_value(self, id_list, query):
        document_indices = [item for sublist in map(lambda x: self.id2document[x], id_list) for item in sublist]
        embedding_indices = [item for sublist in map(lambda x: self.document2embedding[x], document_indices) for item in sublist]
    
        # chunking param = 0 makes sure that we don't chunk the query
        query_embedding = _compute_single_embedding([query], chunking_param=0)[0]
        dot_products = torch.sum(self.embeddings[torch.tensor(embedding_indices, device=device)] * query_embedding, dim=1)
        
        # for each id, we would do a sorting based on each retrieval score
        # to determine the top similarity score for each id, and based on which document
        counter = 0
        res = []
        for id in id_list:
            # this records the embedding indices for a given id
            id_embedding_indices = [item for sublist in map(lambda x: self.document2embedding[x], self.id2document[id]) for item in sublist]
            # this gets the top value and index of similarity score based on the big `dot_products` matrix
            # remember, index here is with respect to the id_embedding_indices above
            if not id_embedding_indices:
                top_value = -1
                top_document = ""
            else:
                top_value, top_index = torch.topk(dot_products[torch.tensor(list(range(counter, counter + len(id_embedding_indices))))], 1)
                # getting the actual document requires going from top_index to actual embedding index to document index
                top_document = self.all_free_text[self.embedding2document[id_embedding_indices[top_index]]]
                counter += len(id_embedding_indices)
            res.append((id, top_value, top_document))
        
        return res     
        

class MultipleEmbeddingStore():
    def __init__(self) -> None:
        # table name -> free text field name -> EmbeddingStore
        self.mapping = {}
    
    def add(self, table_name, primary_key_field_name, free_text_field_name):
        if table_name in self.mapping and free_text_field_name in self.mapping[table_name]:
            print("Table {} for free text field {} already in storage. Negelecting...".format(table_name, free_text_field_name))
            return
        if table_name not in self.mapping:
            self.mapping[table_name] = {}
        self.mapping[table_name][free_text_field_name] = EmbeddingStore(table_name, primary_key_field_name, free_text_field_name)
    
    def retrieve(self, table_name, free_text_field_name):
        return self.mapping[table_name][free_text_field_name]
    
    def dot_product(self, table_name, id_list, field_query_list, top):
        # table_name and id_list must be the same
        # field_query_list stores a list of (free text field, query)
        assert(type(field_query_list) == list)
        if len(id_list) == 0:
            return []
        
        if len(field_query_list) == 1:
            free_text_field = field_query_list[0][0]
            query = field_query_list[0][1]
            
            return self.retrieve(table_name, free_text_field).dot_product(id_list, query, top)
        
        res = {}
        for free_text_field, query in field_query_list:
            for id, top_value, top_document in self.retrieve(table_name, free_text_field).dot_product_with_value(id_list, query):
                if id not in res:
                    res[id] = [top_value, [top_document]]
                else:
                    res[id][0] += top_value
                    res[id][1].append(top_document)
        
        sorted_res = sorted(res.items(), key=lambda x: x[1][0], reverse=True)[:top]
        return list(map(lambda x: (x[0], x[1][1]), sorted_res))

    
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    res = {
        "result" : embedding_store.dot_product(data["table_name"], data["id_list"], data["field_query_list"], data["top"])
    }
    
    return res

if __name__ == "__main__":
    embedding_store = MultipleEmbeddingStore()
    embedding_store.add("courses", "course_id", "description")
    # TODO: add this later to improve perfomance
    # embedding_store.add("courses", "course_id", "title")

    # TODO: add this later once GPU driver is installed
    embedding_store.add("ratings", "course_id", "reviews")
    app.run(host=host, port=port)