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
if source is None or len(source) == 0:
  return False

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

CREATE OR REPLACE FUNCTION boolean_answer (source TEXT, question TEXT)
  RETURNS BOOLEAN
AS $$
if not source:
  return False

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



CREATE OR REPLACE FUNCTION boolean_answer_score (source TEXT[20], question TEXT)
  RETURNS NUMERIC(10, 6)
AS $$
if source is None or len(source) == 0:
  return 0

import requests
import json

URL = "http://127.0.0.1:8500/booleanAnswerScore"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
    "question": question
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION boolean_answer_score (source TEXT, question TEXT)
  RETURNS NUMERIC(10, 6)
AS $$
if not source:
  return 0

import requests
import json

URL = "http://127.0.0.1:8500/booleanAnswerScore"

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

URL = "http://127.0.0.1:8500/summary"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION _string_equals(field_value TEXT, comp_value TEXT, field_name TEXT) ()
  RETURNS BOOLEAN
AS $$

if field_value == comp_value:
  return True

results = plpy.execute("SELECT DISTINCT {} FROM restaurants".format(field_name))
results = [result[field_name] for result in results]

import requests
import json

URL = "http://127.0.0.1:8500/stringEquals"

response = requests.post(url=URL, data=json.dumps({
    "comp_value" : comp_value,
    "field_value" : field_value,
    "field_name" : field_name
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]

$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION _in_any (comp_value TEXT, field_values TEXT[])
  RETURNS BOOLEAN
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/inAny"

response = requests.post(url=URL, data=json.dumps({
    "comp_value" : comp_value,
    "field_values": field_values
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION _equals (comp_value TEXT, field_value TEXT)
  RETURNS BOOLEAN
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/equals"

response = requests.post(url=URL, data=json.dumps({
    "comp_value" : comp_value,
    "field_values": field_value
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$$ LANGUAGE plpython3u;

CREATE OR REPLACE FUNCTION verify (source TEXT[20], question TEXT)
  RETURNS BOOLEAN
AS $$
import requests
import json

URL = "http://127.0.0.1:8500/answer"

response = requests.post(url=URL, data=json.dumps({
    "text" : source,
    "question": question + ", yes or no?"
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object

if "yes" in parsed_result["result"].lower():
  return True
else:
  return False
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