from flask import request, Flask
import json
from utils import num_tokens_from_string, handle_opening_hours
import re
from sql_free_text_support.faiss_embedding import compute_top_similarity_documents
    
# Set the server address
host = "127.0.0.1"
port = 8500
review_server_address = 'http://{}:{}'.format(host, port)
app = Flask(__name__)


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
        documents = compute_top_similarity_documents(data["text"], data["question"], top=5)
        for i in documents:
            if num_tokens_from_string('\n'.join(text_res + [i])) < 3800:
                text_res.append(i)
            else:
                break
    else:
        text_res = [data["text"]]
        
    type_prompt = ""
    if "type_prompt" in data:
        if data["type_prompt"] == "date":
            type_prompt = f" Output in date format, for instance 2001-09-28."
        if data["type_prompt"] == "int4":
            type_prompt = f" Output an integer."
    
    continuation, _ = llm_generate(
        'prompts/review_qa.prompt',
        {'reviews': text_res, 'question': data["question"], "type_prompt": type_prompt},
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
        {'reviews': text_res, 'question': "what is the summary of this document?"},
        engine='gpt-3.5-turbo-0613',
        max_tokens=200,
        temperature=0.0,
        stop_tokens=['\n'],
        postprocess=False,
    )
    
    res = {
        "result" : continuation
    }
    print(res)
    return res

### Functions below are used by the restaurants application only
       
@app.route('/search_by_opening_hours', methods=['POST'])
def search_by_opening_hours():
    data = request.get_json()
    restaurant_hours = data["opening_hours"]
    hours_request = data["opening_hours_request"]
    print("RET_HOURS", restaurant_hours)
    print("RES_REQ", hours_request)
    result = opening_hours_match(restaurant_hours, hours_request)
    return {"result": result}

def get_hours_request_extracted(hours_request):
    intervals = hours_request.split("-")
    hours_request_extracted = [[int(y) for y in x.split(".")] for x in intervals]
    print(hours_request_extracted)
    return hours_request_extracted

def get_restaurant_hours_extracted(restaurant_hours):
    restaurant_hours = json.loads(restaurant_hours)
    restaurant_hours = handle_opening_hours(restaurant_hours)
    restaurant_hours_extracted = [] 
    for hours in restaurant_hours:
        hours_tokenized = re.split("open from | to | on ", hours)
        _, start, end, day = hours_tokenized
        days = {"Monday":0, "Tuesday":1, "Wednesday":2, 
                "Thursday":3, "Friday":4, "Saturday":5, 
                "Sunday":6}
        day = int(days[day])
        end = [int(end[:2]), int(end[2:])]
        start = [int(start[:2]), int(start[2:])] 
        hours_extracted = []
        if end[0] <= start[0]:
            hours_extracted = [day] + [start[0], start[1]] + [23, 59]
            restaurant_hours_extracted.append(hours_extracted)
            hours_extracted = [(day + 1) % 7] + [0,0] + [end[0], end[1]] 
            restaurant_hours_extracted.append(hours_extracted)
        else:
            hours_extracted = [day] + start + end 
        restaurant_hours_extracted.append(hours_extracted)
    return restaurant_hours_extracted

def hours_intersect(restaurant_hours, hours_request):
    day_1, sh_1, sm_1, eh_1, em_1 = restaurant_hours
    day_2, sh_2, sm_2, eh_2, em_2 = hours_request

    if day_1 != day_2: 
        return False
    
    ts_1, te_1 = sh_1*60 + sm_1, eh_1*60 + em_1
    ts_2, te_2 = sh_2*60 + sm_2, eh_2*60 + em_2
    

    if te_1 < ts_2 or te_2 < ts_1:
        return False

    return True

def opening_hours_match(restaurant_opening_hours, opening_hours_request):
    if restaurant_opening_hours == None:
        return False
    restaurant_hours_extracted = get_restaurant_hours_extracted(restaurant_opening_hours)
    hours_request_extracted = get_hours_request_extracted(opening_hours_request)
    print(restaurant_hours_extracted, hours_request_extracted)
    for hours_request in hours_request_extracted:
        for restaurant_hours in restaurant_hours_extracted:
            if hours_intersect(restaurant_hours, hours_request):
                return True

if __name__ == "__main__":
    app.run(host=host, port=port)
