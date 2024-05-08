"""
Main file for a SUQL-powered agent loop
"""

import argparse
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from decimal import Decimal
from typing import List

import requests

from suql.prompt_continuation import llm_generate
from suql.sql_free_text_support.execute_free_text_sql import suql_execute
from suql.utils import input_user, num_tokens_from_string, print_chatbot

logger = logging.getLogger(__name__)


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        db_results: str = None,
        temp_target: str = None,
        user_target: str = None,
        time_statement: dict = None,
        cache: dict = {},
        results_for_ned: dict = None,
    ):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.db_results = db_results
        self.temp_target = temp_target
        self.user_target = user_target
        self.results_for_ned = results_for_ned
        self.cache = cache
        self.time_statement = time_statement

    agent_utterance: str
    user_utterance: str
    db_results: str
    user_target: str
    temp_target: str
    time_statement: dict
    results_for_ned: dict
    cache: dict

    def to_text(self, they="They", you="You"):
        """
        Format:
        You:
        They:
        [You check the database for "find chinese restaurants near palo alto"]
        [Database returns "I see Jing Jing Chinese Gourmet. It is a Chinese restaurant rated 3.5 stars."]
        Restaurant reviews: [
        Review 1: ...
        Summary:
        Review 2: ...
        Summary:
        Review 3: ...
        Summary:
        ]
        """
        ret = ""

        ret += they + ": " + self.user_utterance
        if self.db_results is not None:
            ret += "\n" + '[Database returns "' + self.db_results + '"]'
        if self.agent_utterance is not None:
            ret += "\n" + you + ": " + self.agent_utterance
        return ret


def dialogue_history_to_text(
    history: List[DialogueTurn], they="They", you="You"
) -> str:
    """
    From the agent's point of view, it is 'You:'. The agent starts the conversation.
    """
    ret = ""
    for i in range(len(history)):
        ret += "\n" + history[i].to_text(they=they, you=you)

    if len(history) > 0:
        # remove the extra starting newline
        if ret[0] == "\n":
            ret = ret[1:]

    return ret


# this function extracts only the _id and name fields from the database results, if any
def extract_id_name(results, column_names):
    """
    Custom function to define a "NED" (Named Entity Disambiguation) module in dialog settings.
    This function should be customized to your database.

    ### Why is a NED module needed?

    Consider the following dialog:
    User: I'd like a McDonalds in Palo Alto.
    Agent: I found two McDonald's, one on University Ave and one on El Camino Real.
    User: Tell me more about the one on University Ave.

    Now, there is a ambiguity to be resolved. Which McDonald is the user referring to?
    If we parse this user query simply as:
    ```
    SELECT * FROM restaurants WHERE name = 'McDonald''s';
    ```
    Then we are not guaranteed to get back the one on University Ave.
    If we parse with location, then it could introduce more problems.

    In this case, the better solution is to directly ask LLM to associate this restaurant with
    its ID. For instance,
    ```
    SELECT * FROM restaurants WHERE _id = 2398;
    ```
    where 2398 is the ID for the McDonald's on University Ave.

    This function does exactly that. It extracts the name and _id association for each
    result presented to the user, which is used in the next turn.
    See the rows starting with `Results:` in `parser_suql.prompt`.
    """
    results_for_ned = []
    for result in results:
        temp = dict(
            (column_name, each_result)
            for column_name, each_result in zip(column_names, result)
        )
        if "_id" in temp and "name" in temp:
            results_for_ned.append({"_id": temp["_id"], "name": temp["name"]})
    return results_for_ned


def clean_up_response(results, column_names):
    """
    Custom function to define what to include in the final response prompt.
    This function should be customized to your database.

    This function currently is used for the restuarants domain, with some processing code
    to deal with:
    (1) `rating` (a float issue) and;
    (2) `opening_hours` (represented as a dictionary)

    It also cuts out too long DB results at the end.
    """

    def if_usable_restaurants(field: str):
        """
        Custom function to define what fields to not show to users.
        """
        NOT_USABLE_FIELDS = [
            # reviews are too long
            "reviews",
            # no need to show IDs
            "_id",
            "id",
            # location related fields
            "_location",
            "longitude",
            "latitude",
        ]

        if field in NOT_USABLE_FIELDS:
            return False

        return True

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

    final_res = []
    for res in results:
        temp = dict(
            (column_name, result)
            for column_name, result in zip(column_names, res)
            if if_usable_restaurants(column_name)
        )
        for i in temp:
            if isinstance(temp[i], Decimal):
                temp[i] = float(temp[i])

        if "opening_hours" in temp:
            temp["opening_hours"] = handle_opening_hours(temp["opening_hours"])

        # here is some simple heuristics to deal with too long DB results,
        # thus cutting it at some point
        if num_tokens_from_string(json.dumps(final_res + [temp], indent=4)) > 3500:
            break

        final_res.append(temp)
    return final_res


def parse_execute_sql(dlgHistory, user_query, prompt_file="prompts/parser_suql.prompt"):
    """
    Call an LLM to predict a SUQL, execute it and return results.
    """
    generated_suql, generated_sql_time = llm_generate(
        template_file=prompt_file,
        engine="gpt-3.5-turbo-0613",
        stop_tokens=["Agent:"],
        max_tokens=300,
        temperature=0,
        prompt_parameter_values={"dlg": dlgHistory, "query": user_query},
        postprocess=False,
    )
    print("directly generated SUQL query: {}".format(generated_suql))
    postprocessed_suql = postprocess_suql(generated_suql)

    suql_execute_start_time = time.time()
    final_res, column_names, cache = suql_execute(
        postprocessed_suql,
        {"restaurants": "_id"},
        "restaurants",
        fts_fields=[("restaurants", "name")]
    )
    suql_execute_end_time = time.time()

    results_for_ned = extract_id_name(final_res, column_names)
    final_res = clean_up_response(final_res, column_names)

    return (
        final_res,
        generated_suql,
        postprocessed_suql,
        generated_sql_time,
        suql_execute_end_time - suql_execute_start_time,
        cache,
        results_for_ned,
    )


def postprocess_suql(suql_query):
    """
    Define your custom functions here to post-process a generated SUQL.
    This function should be customized to your database.

    In restaurants, we post-process a generated SUQL
    query with:
    (1) a simple `suql_query.replace("\\'", "''")` to escape a single '
    (2) mandating that all generated queries need to have a LIMIT 3 clause if it does not have a LIMIT clause
    (3) substitute location with lat and long in the database
    (4) substitute opening hours with custom function

    A more in-depth solution could involve SQL parsers like pglast.
    """
    # in PSQL, '' is used to espace a single ', instead of backslashes
    suql_query = suql_query.replace("\\'", "''")

    # by default, we mandate that all generated queries need to have a LIMIT clause
    if not ("LIMIT" in suql_query):
        suql_query = re.sub(r";$", " LIMIT 3;", suql_query, flags=re.MULTILINE)

    def process_query_location(suql_query_):
        """
        Uses regex to change all location clauses into longitude and latitude clauses
        """

        def get_location_from_azure(query):

            EARTH_RADIUS = 6371000  # meters
            TOLERANCE = 1500  # meters

            subscription_key = os.environ["AZURE_MAP_KEY"]
            # API endpoint
            url = "https://atlas.microsoft.com/search/address/json"

            # Parameters for the request
            params = {
                "subscription-key": subscription_key,
                "api-version": "1.0",
                "language": "en-US",
                "query": query,
            }
            # Sending the GET request
            response = requests.get(url, params=params)

            # Extracting the JSON response
            response_json = response.json()

            if response_json["results"][0]["type"] == "Geography":
                bbox = response_json["results"][0]["boundingBox"]
                latitude_north, longitude_west = (
                    bbox["topLeftPoint"]["lat"],
                    bbox["topLeftPoint"]["lon"],
                )
                latitude_south, longitude_east = (
                    bbox["btmRightPoint"]["lat"],
                    bbox["btmRightPoint"]["lon"],
                )
            else:
                # get coords
                coord = response_json["results"][0]["position"]
                longitude, latitude = coord["lon"], coord["lat"]

                # Get location range
                delta_longitude = TOLERANCE / EARTH_RADIUS * 180 / math.pi
                delta_latitude = (
                    TOLERANCE
                    / (EARTH_RADIUS * math.cos(latitude / 180 * math.pi))
                    * 180
                    / math.pi
                )

                longitude_west = longitude - delta_longitude
                longitude_east = longitude + delta_longitude
                latitude_south = latitude - delta_latitude
                latitude_north = latitude + delta_latitude

            return longitude_west, longitude_east, latitude_south, latitude_north

        pattern = r"location\s*=\s*'([^']*)'"

        def replacer(match):
            location_string = match.group(1)
            longitude_west, longitude_east, latitude_south, latitude_north = (
                get_location_from_azure(location_string)
            )
            return f"longitude BETWEEN {longitude_west} AND {longitude_east} AND latitude BETWEEN {latitude_south} AND {latitude_north}"

        return re.sub(pattern, replacer, suql_query_)

    def process_query_opening_hours(suql_query_):
        """
        Uses opening hours to change all opening hour queries into custom functions
        """

        def convert_opening_hours_query(opening_hours_query):
            response, _ = llm_generate(
                "prompts/opening_hours.prompt",
                {"opening_hours_query": opening_hours_query},
                engine="gpt-3.5-turbo-0613",
                max_tokens=200,
                temperature=0.0,
                stop_tokens=["\n"],
                postprocess=False,
            )
            return response

        pattern_1 = r"'([^']*)'\s*=\s*opening_hours"
        pattern_2 = r"opening_hours\s*=\s*'([^']*)'"

        def replacer(match, pattern):
            opening_hours_query = match.group(0).split("=")[pattern]
            opening_hours_translated = convert_opening_hours_query(opening_hours_query)
            return (
                'search_by_opening_hours("opening_hours", '
                + "'"
                + opening_hours_translated
                + "')"
            )

        res = re.sub(pattern_1, lambda x: replacer(x, 0), suql_query_)
        return re.sub(pattern_2, lambda x: replacer(x, 1), res)

    suql_query = process_query_location(suql_query)
    suql_query = process_query_opening_hours(suql_query)
    return suql_query


def compute_next_turn(
    dlgHistory: List[DialogueTurn], user_utterance: str, enable_classifier=True
):
    first_classification_time = 0
    semantic_parser_time = 0
    suql_execution_time = 0
    final_response_time = 0
    cache = {}

    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))

    # determine whether to use database
    if enable_classifier:
        continuation, first_classification_time = llm_generate(
            template_file="prompts/if_db_classification.prompt",
            prompt_parameter_values={"dlg": dlgHistory},
            engine="gpt-3.5-turbo-0613",
            max_tokens=50,
            temperature=0.0,
            stop_tokens=["\n"],
            postprocess=False,
        )

    if not enable_classifier or continuation.startswith("Yes"):
        (
            results,
            first_sql,
            second_sql,
            semantic_parser_time,
            suql_execution_time,
            cache,
            results_for_ned,
        ) = parse_execute_sql(
            dlgHistory, user_utterance, prompt_file="prompts/parser_suql.prompt"
        )
        dlgHistory[-1].db_results = json.dumps(results, indent=4)
        dlgHistory[-1].user_target = first_sql
        dlgHistory[-1].temp_target = second_sql
        dlgHistory[-1].results_for_ned = results_for_ned

        # cut it out if no response returned
        if not results:
            response, final_response_time = llm_generate(
                template_file="prompts/yelp_response_no_results.prompt",
                prompt_parameter_values={"dlg": dlgHistory},
                engine="gpt-3.5-turbo-0613",
                max_tokens=400,
                temperature=0.0,
                stop_tokens=[],
                top_p=0.5,
                postprocess=False,
            )

            dlgHistory[-1].agent_utterance = response
            dlgHistory[-1].time_statement = {
                "first_classification": first_classification_time,
                "semantic_parser": semantic_parser_time,
                "suql_execution": suql_execution_time,
                "final_response": final_response_time,
            }
            return dlgHistory

    response, final_response_time = llm_generate(
        template_file="prompts/yelp_response_SQL.prompt",
        prompt_parameter_values={"dlg": dlgHistory},
        engine="gpt-3.5-turbo-0613",
        max_tokens=400,
        temperature=0.0,
        stop_tokens=[],
        top_p=0.5,
        postprocess=False,
    )
    dlgHistory[-1].agent_utterance = response

    dlgHistory[-1].time_statement = {
        "first_classification": first_classification_time,
        "semantic_parser": semantic_parser_time,
        "suql_execution": suql_execution_time,
        "final_response": final_response_time,
    }
    dlgHistory[-1].cache = cache

    return dlgHistory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="log.log",
        help="Where to write the outputs, pertaining only to CLI testing.",
    )
    parser.add_argument(
        "--quit_commands",
        type=str,
        default=["quit", "q"],
        help="The conversation will continue until this string is typed in, pertaining only to CLI testing.",
    )
    parser.add_argument(
        "--no_logging",
        action="store_true",
        help="Do not output extra information about the intermediate steps.",
    )
    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(
            level=logging.CRITICAL, format=" %(name)s : %(levelname)-8s : %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=" %(name)s : %(levelname)-8s : %(message)s"
        )

    # The dialogue loop. The agent starts the dialogue
    dlgHistory = []
    print_chatbot(dialogue_history_to_text(dlgHistory, they="User", you="Chatbot"))

    try:
        while True:
            user_utterance = input_user()
            if user_utterance in args.quit_commands:
                break

            dlgHistory = compute_next_turn(
                dlgHistory,
                user_utterance,
            )
            print_chatbot("Chatbot: " + dlgHistory[-1].agent_utterance)
            print(dlgHistory[-1].time_statement)

    finally:
        with open(args.output_file, "a") as output_file:
            output_file.write(
                "=====\n"
                + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                + "\n"
                + dialogue_history_to_text(dlgHistory, they="User", you="Chatbot")
                + "\n"
            )
