from transformers import AutoModel, AutoTokenizer
import pymongo
from typing import List, Dict
import os

model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco")
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco")

mongo = os.environ.get('COSMOS_CONNECTION_STRING')
client = pymongo.MongoClient(mongo)
db = client['yelpbot']
collection = db['yelp_data']


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
    print(top_restaurants)
    rest_recommendations = set()

    for r in top_restaurants:
        rest_recommendations.add(r[2])

    return list(rest_recommendations)
