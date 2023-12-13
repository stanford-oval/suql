from transformers import AutoModel, AutoTokenizer
import pymongo
from typing import List
import os
from flask import request, Flask
import time
import torch
from utils import linearize, chunk_text
import json
from tqdm import tqdm
import hashlib
from functools import reduce
import operator
from utils import num_tokens_from_string

# cuda_ok = torch.cuda.is_available()
model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco")
# if cuda_ok:
#     device = torch.device("cuda")
#     model = model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")

mongo = os.environ.get('COSMOS_CONNECTION_STRING')
client = pymongo.MongoClient(mongo)
db = client['yelpbot']
collection = db['yelp_data']
schematized = db['schematized']
cache_db = client['free_text_cache']['hash_to_embeddings']

# Set the server address
host = "127.0.0.1"
port = 8500
review_server_address = 'http://{}:{}'.format(host, port)
app = Flask(__name__)


def compute_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

def filter_reviews(restaurants: List[str], keyword: str) -> List[str]:
    """
    params:
        restaruants: a list of restaruant IDs
        keyword: a 'filter criteria'

    return:
        rest_recommendations = a list of restaurants
    """
    similarities = []  # tuple of (sentence, similarity to the first one)

    for r_id in restaurants:
        query = {'id': r_id}
        result = collection.find(query)
        reviews = [keyword]
        restaurants = ['']  # contains all the restaurant IDs corresponding to the reviews

        for doc in result:
            reviews.extend(doc['reviews'])
            restaurants.extend([doc['id'] for i in range(len(doc['reviews']))])

        inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
        # if cuda_ok:
        #     inputs = inputs.to(device)
        # embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]\
        #     .squeeze(1)  # the embedding of the [CLS] token after the final layer
        inputs = inputs.to(device)
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]\
            .squeeze(1)  # the embedding of the [CLS] token after the final layer

        # if cuda_ok:
        #     embeddings[0].to(device)
        # for i in range(1, len(embeddings)):
        #     embeddings[i].to(device)
        #     similarities.append((reviews[i], embeddings[0] @ embeddings[i], restaurants[i]))

        embeddings[0].to(device)
        for i in range(1, len(embeddings)):
            embeddings[i].to(device)
            similarities.append((reviews[i], embeddings[0] @ embeddings[i], restaurants[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    num_to_get = min(len(similarities), 5)

    top_restaurants = similarities[:num_to_get]
    rest_recommendations = set()

    for r in top_restaurants:
        # since reviews can only be of type String in Genie, the returned result will separate reviews by `\t`
        # thus, we make sure here that the review text does not contain `\t`
        
        # r[0] is the review
        # r[1] is the similarity score
        # r[2] is the restaurant id
        rest_recommendations.add((r[0].replace('\t', ''), r[1].item(), r[2]))

    res = list(rest_recommendations)
    res = sorted(res, key=lambda x: x[1], reverse=True)
    return res

def baseline_filter(to_query):
    """
    params:
		rest_ids: a list of restaruant ids 
		to_query: a 'filter criteria' 
	return:
		rest_recommendations: a list of restaurants
    """
    client = pymongo.MongoClient('localhost', 27017)
    db = client['yelpbot']
    collection = db['schematized']
    similarities = []  # tuple of sentence, similarity to the query
    rest_ids = []

    cursor = collection.find()

    for doc in cursor:
        rest_similarities = []
        info = [to_query]
        doc_data = linearize(doc, 100)
        info.extend(doc_data)
        
        inputs = tokenizer(info, padding=True, truncation=True, return_tensors='pt').to(device)
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:,:1].squeeze(1)

        embeddings[0].to(device)

        for i in range(1, len(embeddings)):
            embeddings[i].to(device)
            rest_similarities.append((embeddings[0] @ embeddings[i], doc))

        scores = [rest_similarities[i][0].item() for i in range(len(rest_similarities))]
        max_score = max(scores)
        max_idx = scores.index(max_score)
        similarities.append((info[max_idx + 1], max_score, doc))

        # torch.cuda.empty_cache()
        print('done')
    
    similarities.sort(key= lambda x: x[1], reverse=True)

    num_to_get = min(len(similarities), 5)
    top_restaurants = similarities[:num_to_get]

    collection = db['yelp_data']

    rest_recommendations = []
    for rec in top_restaurants:
        docs = collection.find({'id': rec[2]['id']})
        for d in docs:
            rest_recommendations.append(d)
            break

    return rest_recommendations


@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    data = request.get_json()
    print("/query receieved request {}".format(data))
    
    # input params in this `data`    
    # data["keyword"] : keyword to query
    # data["restaurant_ids"] : list of restaurant ids to query reviews
    # data["num_max_restaurants"] : max number of restaurants to query (optional)

    if "keyword" not in data or "restaurant_ids" not in data:
        return None
    
    num_max_restaurants = data["num_max_restaurants"] if "num_max_restaurants" in data else 5

    res = filter_reviews(data["restaurant_ids"][:num_max_restaurants], data["keyword"])
    end_time = time.time()
    print(res)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    return res

@app.route('/getReviews', methods=['POST'])
def getReviews():
    data = request.get_json()
    print("/getReviews receieved request {}".format(data))
        
    # input params in this `data`    
    # data["restaurant_ids"] : list of restaurant ids to query reviews

    if "restaurant_ids" not in data:
        return None
    
    res = {}
    for r_id in data["restaurant_ids"]:
        query = {'id': r_id}
        result = collection.find_one(query)
        
        if result and "reviews" in result:
            res[r_id] = result["reviews"]
        else:
            res[r_id] = []
    
    print(res)
    return res

@app.route('/getMenus', methods=['POST'])
def getMenus():
    data = request.get_json()
    print("/getMenus receieved request {}".format(data))
        
    # input params in this `data`    
    # data["restaurant_ids"] : list of restaurant ids to query menu

    if "restaurant_ids" not in data:
        return None
    
    res = {}
    for r_id in data["restaurant_ids"]:
        query = {'id': r_id}
        result = collection.find_one(query)
        # filter out \t in reviews
        if result and "dishes" in result:
            res[r_id] = [i[0] for i in result["dishes"]]
        else:
            res[r_id] = []
    
    print(res)
    return res

@app.route('/answer', methods=['POST'])
def answer():
    from prompt_continuation import llm_generate
    data = request.get_json()
    # print("/answer receieved request {}".format(data))
        
    # input params in this `data`    
    # data["text"] : text to QA upon
    # data["question"] : question to answer

    if "text" not in data or "question" not in data:
        return None
    
    if not data["text"]:
        return {
            "result": "no information"
        }
    
    text_res = []
    if isinstance(data["text"], list):
        documents = _compute_single_embedding_with_mapping(data["text"], data["question"], top=5)
        for i in documents:
            if num_tokens_from_string('\n'.join(text_res + [i])) < 3800:
                text_res.append(i)
            else:
                break
    else:
        text_res = [data["text"]]
    
    continuation, _ = llm_generate(
        'prompts/review_qa.prompt',
        {'reviews': text_res, 'question': data["question"] + "; consider all relevant information."},
        engine='gpt-3.5-turbo-0613',
        max_tokens=200,
        temperature=0.0,
        stop_tokens=['\n'],
        postprocess=False
    )
    
    res = {
        "result" : continuation
    }
    print(res)
    return res

@app.route('/summary', methods=['POST'])
def summary():
    from prompt_continuation import llm_generate
    data = request.get_json()
    # print("/answer receieved request {}".format(data))
        
    # input params in this `data`    
    # data["text"] : text to QA upon
    # (optional) data["focus"] : focus of summary

    if "text" not in data:
        return None
    
    if not data["text"]:
        return {
            "result": "no information"
        }
    
    text_res = []
    if isinstance(data["text"], list):
        for i in data["text"]:
            if num_tokens_from_string('\n'.join(text_res + [i])) < 3800:
                text_res.append(i)
            else:
                break
    else:
        text_res = [data["text"]]
    
    continuation, _ = llm_generate(
        'prompts/review_qa.prompt',
        {'reviews': text_res, 'question': "give me a summary of this document"},
        engine='gpt-3.5-turbo',
        max_tokens=200,
        temperature=0.0,
        stop_tokens=['\n'],
        postprocess=False,
        attempts=0
    )
    
    res = {
        "result" : continuation
    }
    print(res)
    return res

def _compute_single_embedding(documents, chunking_param=15, safe_assume_exists = False):
    documents_hashes = list(map(compute_sha256, documents))
    cache_results = list(cache_db.find({"_id": {"$in": documents_hashes}}))
    
    existing_embeddings = reduce(operator.add, map(lambda x: x["embeddings"], cache_results)) if cache_results else []
    existing_embeddings = torch.tensor(existing_embeddings, device=device)
    
    if safe_assume_exists:
        return existing_embeddings
    
    index_dict = {hash: index for index, hash in enumerate(documents_hashes)}

    for doc in cache_results:
        index_dict.pop(doc["_id"], None)
    
    missing_hashes_with_indices = list(index_dict.items())
    for missing_hash, missing_index in missing_hashes_with_indices:
        chunked_documents = chunk_text(documents[missing_index], k=chunking_param, use_spacy=True)
        inputs = tokenizer(chunked_documents, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():  # Disables gradient calculation to save memory
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1).to(device)  # the embedding of the [CLS] token after the final layer
        existing_embeddings = torch.cat((existing_embeddings, embeddings), dim=0)
        cache_db.insert_one({
            "_id": missing_hash,
            "embeddings": embeddings.tolist()
        })
    
    return existing_embeddings

def _compute_single_embedding_with_mapping(documents, question, chunking_param=15, top=1):
    existing_embeddings = None
    embedding2document = {}
    embedding_counter = 0
    for index, document in enumerate(documents):
        document_hash = compute_sha256(document)
        cache_results = cache_db.find_one({"_id": document_hash})
        
        if not cache_results:
            chunked_documents = chunk_text(document, k=chunking_param, use_spacy=True)
            inputs = tokenizer(chunked_documents, padding=True, truncation=True, return_tensors="pt").to(device)
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1).to(device)  # the embedding of the [CLS] token after the final layer
            cache_db.insert_one({
                "_id": document_hash,
                "embeddings": embeddings.tolist()
            })
        else:
            embeddings = torch.tensor(cache_results["embeddings"], device=device)
        
        if existing_embeddings is None:
            existing_embeddings = embeddings
        else:
            existing_embeddings = torch.cat((existing_embeddings, embeddings), dim=0)
            
        for embedding_index in range(embedding_counter, embedding_counter + embeddings.size()[0]):
            embedding2document[embedding_index] = index
        embedding_counter += embeddings.size()[0]
        
    question_embedding = _compute_single_embedding([question], chunking_param=chunking_param)[0]
    dot_products = torch.mv(existing_embeddings, question_embedding)
    _, indices_max = torch.topk(dot_products, existing_embeddings.size()[0])
    
    res = []
    for index in indices_max:
        index = int(index.item())
        if documents[embedding2document[index]] not in res:
            res.append(documents[embedding2document[index]])
        if len(res) >= top:
            break
        
    # torch.cuda.empty_cache()
    return res


def _compute_embeddings(documents, question, chunking_param=15, top=1):
    print(len(documents))
    assert(type(documents) == list)
    assert(type(question) == str)
    
    if not documents:
        return 0
    
    documents_embeddings = _compute_single_embedding(documents, chunking_param=chunking_param, safe_assume_exists=True)
    question_embedding = _compute_single_embedding([question], chunking_param=chunking_param)[0]
    
    # compute final embedding
    mid_time = time.time()
    dot_products = torch.mv(documents_embeddings, question_embedding)

    values_max, indices_max = torch.topk(dot_products, top)
    # torch.cuda.empty_cache()
    end_time = time.time()
    print("time for computing dot product {}".format(end_time - mid_time))
    return values_max, indices_max


def boolean_retrieve_reviews(reviews, question):
    """Given a list of reviews, return whether any matches the question.

    Args:
        reviews (_type_): _description_
        question (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    highest_similarity = _compute_embeddings(reviews, question)[0][0]

    if (highest_similarity > 208.5):
        return True
    else:
        return False

def get_highest_embedding(reviews, question):
    while True:
        try:
            highest_similarity = _compute_embeddings(reviews, question)[0][0]
            break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # torch.cuda.empty_cache()
                print("Out of memory error occurred. Try reducing the batch size or model size.")
            else:
                print("Runtime error occurred:", e)
                break
            
    return highest_similarity


@app.route('/booleanAnswer', methods=['POST'])
def boolean_answer():
    data = request.get_json()
    # input params in this `data`    
    # data["text"] : text to filter upon, either a string or a list
    # data["question"] : question to answer

    if "text" not in data or "question" not in data:
        return None
    
    if isinstance(data["text"], list):
        input_text = data["text"]
    elif isinstance(data["text"], str):
        input_text = [data["text"]]
    else:
        return None
    
    result = boolean_retrieve_reviews(input_text, data["question"])
    res = {
        "result" : result
    }
    print(res)
    return res


@app.route('/booleanAnswerScore', methods=['POST'])
def boolean_answer_score():
    data = request.get_json()
    # print("/booleanAnswerScore receieved request {}".format(data))
        
    # input params in this `data`    
    # data["text"] : text to filter upon, either a string or a list
    # data["question"] : question to answer

    if "text" not in data or "question" not in data:
        return None
    
    if isinstance(data["text"], list):
        input_text = data["text"]
    elif isinstance(data["text"], str):
        input_text = [data["text"]]
    else:
        return None
    
    start_time = time.time()
    result = get_highest_embedding(input_text, data["question"])
    res = {
        "result" : result
    }
    end_time = time.time()
    execution_time = end_time - start_time
    print(res, "; {} seconds".format(execution_time))
    return res


@app.route('/stringEquals', methods=['POST'])
def string_equals():
    data = request.get_json()
    print("/stringEquals receieved request {}".format(data))
        
    # input params in this `data`    
    # data["comp_value"]  : text to compare against
    # data["field_value"] : text in the db to compare
    # data["field_name"]  : field name of this value

    if "comp_value" not in data or "field_value" not in data or "field_name" not in db:
        return None


def get_all_processed():
    restaurants = list(schematized.find())
    for i in tqdm(restaurants):
        print(i["name"])
        if i["reviews"] != []:
            _compute_embeddings(i["reviews"], "")

if __name__ == "__main__":
    #print(baseline_filter("this is a good restaurant for large groups"))
    app.run(host=host, port=port)
    # get_all_processed()
