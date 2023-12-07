CREATE FUNCTION answer (source TEXT[], question TEXT)
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

CREATE FUNCTION answer (source TEXT, question TEXT)
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

CREATE OR REPLACE FUNCTION summary (source TEXT[])
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

CREATE OR REPLACE FUNCTION summary (source TEXT)
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
