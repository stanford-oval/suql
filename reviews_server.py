from transformers import AutoModel, AutoTokenizer
import pymongo
from typing import List
import os
from flask import request, Flask
import time
import torch
from prompt_continuation import llm_generate

cuda_ok = torch.cuda.is_available()
model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco")
if cuda_ok:
    device = torch.device("cuda")
    model = model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")

mongo = os.environ.get('COSMOS_CONNECTION_STRING')
client = pymongo.MongoClient(mongo)
db = client['yelpbot']
collection = db['yelp_data']

# Set the server address
host = "127.0.0.1"
port = 8500
review_server_address = 'http://{}:{}'.format(host, port)
app = Flask(__name__)

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
        if cuda_ok:
            inputs = inputs.to(device)
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]\
            .squeeze(1)  # the embedding of the [CLS] token after the final layer

        if cuda_ok:
            embeddings[0].to(device)
        for i in range(1, len(embeddings)):
            if cuda_ok:
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

def baseline_filter(rest_ids, to_query):
    """
    params:
		rest_ids: a list of restaruant ids
		to_query: a 'filter criteria' 
	return:
		rest_recommendations: a list of restaurants
    """
    similarities = []  # tuple of sentence, similarity to the query

    for id in rest_ids:
        query = {'id': id}
        result = collection.find(query)
        info = [to_query]

        rest_similarities = []

        for doc in result:
            info.extend(doc['data'])
        
        inputs = tokenizer(info, padding=True, truncation=True, return_tensors='pt').to(device)
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hiddent_states[-1][:,:1].squeeze(1)

        embeddings[0].to(device)

        for i in range(1, len(embeddings)):
            embeddings[i].to(device)
            rest_similarities.append((embeddings[0] @ embeddings[i], id))

        scores = [rest_similarities[i][0].item() for i in range(len(rest_similarities))]
		max_score = max(scores)
		max_idx = scores.index(max_score)

		similarities.append((info[max_idx + 1], max_score, id))
	

	similarities.sort(key= lambda x: x[1], reverse=True)

    num_to_get = min(len(similarities) - 1, 5)
    top_restaurants = similarities[:num_to_get]

    rest_recommendations = [similarities[i][2] for i in range(num_to_get)]
	# rest_info = collection.find({'id': similarities[0][2]})
	
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
    data = request.get_json()
    print("/answer receieved request {}".format(data))
        
    # input params in this `data`    
    # data["text"] : text to QA upon
    # data["question"] : question to answer

    if "text" not in data or "question" not in data:
        return None
    
    text_res = []
    if isinstance(data["text"], list):
        # TODO: to be precise one needs to use the openAI tokenzer. for now I am just using some
        # ad-hoc hard-coded character count
        for i in data["text"]:
            if len('\n'.join(text_res + [i])) < 14000:
                text_res.append(i)
            else:
                break
    else:
        text_res = [data["text"]]
    
    continuation, _ = llm_generate(
        'prompts/review_qa.prompt',
        {'reviews': text_res, 'question': data["question"]},
        engine="text-davinci-003",
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


def boolean_retrieve_reviews(reviews, question):
    """Given a list of reviews, return whether any matches the question.

    Args:
        reviews (_type_): _description_
        question (_type_): _description_

    Returns:
        _type_: _description_
    """
    similarities = []  # tuple of (sentence, similarity to the first one)
    
    # the first element of reviews is the question itself
    reviews = [question] + reviews

    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
    if cuda_ok:
        inputs = inputs.to(device)
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]\
        .squeeze(1)  # the embedding of the [CLS] token after the final layer

    if cuda_ok:
        embeddings[0].to(device)
    for i in range(1, len(embeddings)):
        if cuda_ok:
            embeddings[i].to(device)
        similarities.append((reviews[i], (embeddings[0] @ embeddings[i]).item()))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print(similarities)
    if (not similarities):
        return False

    if (similarities[0][1] > 208.5):
        return True
    else:
        return False


@app.route('/booleanAnswer', methods=['POST'])
def boolean_answer():
    data = request.get_json()
    print("/booleanAnswer receieved request {}".format(data))
        
    # input params in this `data`    
    # data["text"] : text to filter upon, either a string or a list
    # data["question"] : question to answer

    if "text" not in data or "question" not in data:
        return None
    
    if isinstance(data["text"], list):
        result = boolean_retrieve_reviews(data["text"], data["question"])
        res = {
            "result" : result
        }
        print(res)
        return res
    
    return None
    

if __name__ == "__main__":
    app.run(host=host, port=port)