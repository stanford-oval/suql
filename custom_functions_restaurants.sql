-- Function below is only used in the restaurant database

CREATE OR REPLACE FUNCTION search_by_opening_hours(opening_hours jsonb, opening_hours_request text)
 RETURNS boolean
 LANGUAGE plpython3u
AS $function$
import requests
import json
URL = "http://127.0.0.1:8500/search_by_opening_hours"
response = requests.post(url=URL, data=json.dumps({
    "opening_hours" : opening_hours,
    "opening_hours_request": opening_hours_request
}), headers={'Content-Type': 'application/json'})
response.raise_for_status()  # Raise an exception if the request was not successful
parsed_result = response.json()  # Assuming the response is JSON, parse it into a Python object
return parsed_result["result"]
$function$;