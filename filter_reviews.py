from transformers import AutoModel, AutoTokenizer
import pymongo
from typing import List, Dict
import os
from flask import request, Flask

model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco")
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")

mongo = os.environ.get('COSMOS_CONNECTION_STRING')
client = pymongo.MongoClient(mongo)
db = client['yelpbot']
collection = db['yelp_data']

# Set the server address
host = "127.0.0.1"
port = 8500
review_server_address = 'http://{}:{}/query'.format(host, port)
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
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1]\
            .squeeze(1)  # the embedding of the [CLS] token after the final layer

        for i in range(1, len(embeddings)):
            similarities.append((reviews[i], embeddings[0] @ embeddings[i], restaurants[i]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    num_to_get = min(len(similarities), 5)

    top_restaurants = similarities[:num_to_get]
    rest_recommendations = set()

    for r in top_restaurants:
        # since reviews can only be of type String in Genie, the returned result will separate reviews by `\t`
        # thus, we make sure here that the review text does not contain `\t`
        rest_recommendations.add((r[0], r[2].replace('\t', '')))

    return list(rest_recommendations)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    
    # input params in this `data`    
    # data["keyword"] : keyword to query
    # data["restaurant_ids"] : list of restaurant ids to query reviews
    # data["num_max_restaurants"] : max number of restaurants to query (optional)

    if "keyword" not in data or "restaurant_ids" not in data:
        return None
    
    num_max_restaurants = data["num_max_restaurants"] if "num_max_restaurants" in data else 5

    res = filter_reviews(data["restaurant_ids"][:num_max_restaurants], data["keyword"])
    print(res)
    return res

if __name__ == "__main__":
    app.run(host=host, port=port)