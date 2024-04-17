import json
import re

from flask import Flask, request

from suql.faiss_embedding import compute_top_similarity_documents
from suql.utils import num_tokens_from_string

app = Flask(__name__)

# # Default top number of results to send to LLM answer function
# # if given a list of strings
# k = 5

# # Max number of input tokens for the `summary` function
# max_input_token = 3800

# # Default LLM engine for `answer` and `summary` functions
# engine = "gpt-3.5-turbo-0613"


def _answer(source, query, type_prompt = None, k=5, max_input_token=3800, engine="gpt-3.5-turbo-0613"):
    from suql.prompt_continuation import llm_generate
    if not source:
        return {"result": "no information"}

    text_res = []
    if isinstance(source, list):
        documents = compute_top_similarity_documents(
            source, query, top=k
        )
        for i in documents:
            if num_tokens_from_string("\n".join(text_res + [i])) < max_input_token:
                text_res.append(i)
            else:
                break
    else:
        text_res = [source]

    type_prompt = ""
    if type_prompt:
        if type_prompt == "date":
            type_prompt = f" Output in date format, for instance 2001-09-28."
        if type_prompt == "int4":
            type_prompt = f" Output an integer."

    continuation, _ = llm_generate(
        "prompts/answer_qa.prompt",
        {
            "reviews": text_res,
            "question": query,
            "type_prompt": type_prompt,
        },
        engine=engine,
        max_tokens=200,
        temperature=0.0,
        stop_tokens=["\n"],
        postprocess=False,
    )
    return {"result": continuation}

def start_free_text_fncs_server(
    host="127.0.0.1", port=8500, k=5, max_input_token=3800, engine="gpt-3.5-turbo-0613"
):
    """
    Set up a free text functions server for the free text
    `answer` and `summary` functions.

    Args:
        host (str, optional): The host running this server. Defaults to "127.0.0.1" (localhost).
        port (int, optional): The port running this server. Defaults to 8500.
        k (int, optional): Default top number of results to send to LLM answer function
            if given a list of strings. Defaults to 5.
        max_input_token (int, optional): Max number of input tokens for the `summary` function.
            Defaults to 3800.
        engine (str, optional): Default LLM engine for `answer` and `summary` functions.
            Defaults to "gpt-3.5-turbo-0613".
    """

    @app.route("/answer", methods=["POST"])
    def answer():
        """
        LLM-based answer function, set up as a server for PSQL to call.

        Expected input params in request.get_json():

        data["text"] (str or List[str]): text to QA upon
        data["question"] (str): question to answer

        If data["text"] is a list of string, compute embedding to find top k
        documents to send the LLM to answer with (Default set to 5);
        Include those in the LLM prompt until `max_input_token` is reached
        in the same order (Default set to 3800).

        Returns:
        {
            "result" (str): answer function result
        }
        """
        from suql.prompt_continuation import llm_generate

        data = request.get_json()

        if "text" not in data or "question" not in data:
            return None

        return _answer(
            data["text"],
            data["question"],
            type_prompt=data["type_prompt"] if "type_prompt" in data else None,
            k = k,
            max_input_token = max_input_token,
            engine = engine
        )
    

    @app.route("/summary", methods=["POST"])
    def summary():
        """
        LLM-based summary function, set up as a server for PSQL to call.
        `summary(text)` is a syntactic sugar for `answer(text, 'what is the summary of this document?')`
        By default, append as many documents as possible until `max_input_token` is reached
        (Default set to 3800).

        Expected input params in request.get_json():

        data["text"] : text to summarize upon.

        Returns:
        {
            "result" (str): summary function result
        }
        """
        from suql.prompt_continuation import llm_generate

        data = request.get_json()

        if "text" not in data:
            return None

        if not data["text"]:
            return {"result": "no information"}

        text_res = []
        if isinstance(data["text"], list):
            for i in data["text"]:
                if num_tokens_from_string("\n".join(text_res + [i])) < max_input_token:
                    text_res.append(i)
                else:
                    break
        else:
            text_res = [data["text"]]

        continuation, _ = llm_generate(
            "prompts/answer_qa.prompt",
            {"reviews": text_res, "question": "what is the summary of this document?"},
            engine=engine,
            max_tokens=200,
            temperature=0.0,
            stop_tokens=["\n"],
            postprocess=False,
        )

        res = {"result": continuation}
        return res

    # start Flask server
    app.run(host=host, port=port)


# Functions below are used by the restaurants application only.


@app.route("/search_by_opening_hours", methods=["POST"])
def search_by_opening_hours():
    data = request.get_json()
    restaurant_hours = data["opening_hours"]
    hours_request = data["opening_hours_request"]
    result = opening_hours_match(restaurant_hours, hours_request)
    return {"result": result}


def get_hours_request_extracted(hours_request):
    intervals = hours_request.split("-")
    hours_request_extracted = [[int(y) for y in x.split(".")] for x in intervals]
    return hours_request_extracted


def get_restaurant_hours_extracted(restaurant_hours):
    def handle_opening_hours(input_dict):
        """
        Custom function to convert opening hours into LLM-readable formats.
        """
        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        def get_order_index(x):
            try:
                return order.index(x["day_of_the_week"])
            except ValueError:
                return len(order)  # or any default value

        res = []
        input_dict = sorted(input_dict, key=lambda x: get_order_index(x))
        for i in input_dict:
            res.append(
                f'open from {i["open_time"]} to {i["close_time"]} on {i["day_of_the_week"]}'
            )
        return res

    restaurant_hours = json.loads(restaurant_hours)
    restaurant_hours = handle_opening_hours(restaurant_hours)
    restaurant_hours_extracted = []
    for hours in restaurant_hours:
        hours_tokenized = re.split("open from | to | on ", hours)
        _, start, end, day = hours_tokenized
        days = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        day = int(days[day])
        end = [int(end[:2]), int(end[2:])]
        start = [int(start[:2]), int(start[2:])]
        hours_extracted = []
        if end[0] <= start[0]:
            hours_extracted = [day] + [start[0], start[1]] + [23, 59]
            restaurant_hours_extracted.append(hours_extracted)
            hours_extracted = [(day + 1) % 7] + [0, 0] + [end[0], end[1]]
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

    ts_1, te_1 = sh_1 * 60 + sm_1, eh_1 * 60 + em_1
    ts_2, te_2 = sh_2 * 60 + sm_2, eh_2 * 60 + em_2

    if te_1 < ts_2 or te_2 < ts_1:
        return False

    return True


def opening_hours_match(restaurant_opening_hours, opening_hours_request):
    if restaurant_opening_hours == None:
        return False
    restaurant_hours_extracted = get_restaurant_hours_extracted(
        restaurant_opening_hours
    )
    hours_request_extracted = get_hours_request_extracted(opening_hours_request)
    for hours_request in hours_request_extracted:
        for restaurant_hours in restaurant_hours_extracted:
            if hours_intersect(restaurant_hours, hours_request):
                return True


if __name__ == "__main__":
    start_free_text_fncs_server(
        host="127.0.0.1",
        port=8500,
    )
