CREATE FUNCTION answer (source TEXT[20], question TEXT)
  RETURNS TEXT
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/answer"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
    "question": question
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION boolean_answer (source TEXT[20], question TEXT)
  RETURNS BOOLEAN
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/booleanAnswer"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
    "question": question
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION summary (source TEXT[20])
  RETURNS TEXT
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/answer"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
    "question": "general information about the restaurant"
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE TABLE restaurants (
    _id SERIAL PRIMARY KEY,
    id VARCHAR(50) ,
    name TEXT,
    location TEXT,
    cuisines TEXT[],
    price TEXT,
    rating NUMERIC(2,1),
    num_reviews INTEGER,
    address TEXT,
    popular_dishes TEXT[],
    phone_number TEXT,
    reviews TEXT[],
    opening_hours TEXT
);