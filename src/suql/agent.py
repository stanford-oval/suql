"""
Main file for a SUQL-powered agent loop
"""

import argparse
import chainlit as cl
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

from suql.prompt_continuation import llm_generate, async_generate_chainlit
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
            "reviews",
            "id",

            # schematized fields        
            "ambiance",
            "specials",
            "reservation_info",
            "nutrition_info",
            "signature_cocktails",
            "has_private_event_spaces",
            "promotions",
            "parking_options",
            "game_day_specials",
            "live_sports_events",
            "dress_code",
            "happy_hour_info",
            "highlights",
            "service",
            "has_outdoor_seating",
            "drinks",
            "dietary_restrictions",
            "experience",
            "nutritious_options",
            "creative_menu",
            "has_student_discount",
            "has_senior_discount",
            "local_cuisine",
            "trendy",
            "wheelchair_accessible",
            "noise_level",
            "kids_menu",
            "childrens_activities",
            "if_family_friendly",
            "wait_time",
            "has_live_music",
            "serves_alcohol",
            "michelin",
            "accomodates_large_groups",
            
            # the citation fields        
            "ambiance_citation",
            "specials_citation",
            "reservation_info_citation",
            "nutrition_info_citation",
            "signature_cocktails_citation",
            "has_private_event_spaces_citation",
            "promotions_citation",
            "parking_options_citation",
            "game_day_specials_citation",
            "live_sports_events_citation",
            "dress_code_citation",
            "happy_hour_info_citation",
            "highlights_citation",
            "service_citation",
            "has_outdoor_seating_citation",
            "drinks_citation",
            "dietary_restrictions_citation",
            "experience_citation",
            "nutritious_options_citation",
            "creative_menu_citation",
            "has_student_discount_citation",
            "has_senior_discount_citation",
            "local_cuisine_citation",
            "trendy_citation",
            "wheelchair_accessible_citation",
            "noise_level_citation",
            "kids_menu_citation",
            "childrens_activities_citation",
            "if_family_friendly_citation",
            "wait_time_citation",
            "has_live_music_citation",
            "serves_alcohol_citation",
            "michelin_citation",
            "accomodates_large_groups_citation",
            
            # special internal fields
            "_score",
            "_schematization_results",
            
            # outdated location
            "_location",
            "longitude",
            "latitude"
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


async def generate_sql(dlgHistory, user_query):
    async with cl.Step(name="SUQL", type="llm", language="sql", disable_feedback=False) as step:
        first_sql = await async_generate_chainlit(
            'prompts/parser_suql.prompt',
            {'dlg': dlgHistory, 'query': user_query},
            step,
            'gpt-3.5-turbo-0613',
            max_tokens=300,
            temperature=0,
            stop=["Agent:"])
    
    first_sql = first_sql.replace("\\'", "''")
    if not ("LIMIT" in first_sql):
        first_sql = re.sub(r';$', ' LIMIT 3;', first_sql, flags=re.MULTILINE)

    second_sql = postprocess_suql(first_sql)
    step.output = second_sql
    await step.update()
    
    return first_sql, second_sql


def json_to_markdown_table(data):
    # Assuming the JSON data is a list of dictionaries
    # where each dictionary represents a row in the table
    if not data or not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON data is not in the expected format for a table")

    def rearrange_list(lst):
        if "_id" in lst:
            lst.remove("_id")
            lst.insert(0, "_id")
        return lst

    # Extract headers, and put _id up front
    headers = rearrange_list(list(data[0].keys()))
    
    # Start building the Markdown table
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["-" * len(header) for header in headers]) + " |\n"

    # Add rows
    for row in data:
        markdown_table += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"

    return markdown_table

async def execute_sql(sql):
    suql_execute_start_time = time.time()
    async with cl.Step(name="Results", type="llm", disable_feedback=False) as step:
        final_res, column_names, cache = suql_execute(
            sql,
            {"restaurants": "_id"},
            fts_fields=[("restaurants", "name")],
            # NOTE: different from default 8501, this is due to set up on our VM
            embedding_server_address="http://127.0.0.1:8509"
        )
        results_for_ned = extract_id_name(final_res, column_names)
        final_res = clean_up_response(final_res, column_names)
        if final_res:
            step.output = json_to_markdown_table(final_res)
        else:
            step.output = "SUQL returned no results"
        
    suql_execute_end_time = time.time()
    
    return final_res, suql_execute_end_time - suql_execute_start_time, results_for_ned

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

        pattern = r"'([^']*)'\s*=\s*ANY\(CAST opening_hours AS ARRAY\)"

        def replacer(match):
            opening_hours_query = match.group(0).split(" = ")[0]
            opening_hours_translated = convert_opening_hours_query(opening_hours_query)
            return (
                'search_by_opening_hours("opening_hours", '
                + "'"
                + opening_hours_translated
                + "')"
            )

        return re.sub(pattern, replacer, suql_query_)

    suql_query = process_query_location(suql_query)
    suql_query = process_query_opening_hours(suql_query)
    return suql_query

async def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str):
    
    first_classification_time = 0
    semantic_parser_time = 0
    suql_execution_time = 0
    final_response_time = 0
    cache = {}

    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))
    
    # determine whether to use database
    continuation, first_classification_time = llm_generate(template_file='prompts/if_db_classification.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine='gpt-3.5-turbo-0613',
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)

    if continuation.startswith("Yes"):
        first_sql, location_sql = await generate_sql(dlgHistory, user_utterance)
        results, suql_execution_time, results_for_ned = await execute_sql(location_sql)
        dlgHistory[-1].user_target = first_sql
        dlgHistory[-1].temp_target = ""
        dlgHistory[-1].db_results = json.dumps(results, indent=4)
        dlgHistory[-1].results_for_ned = results_for_ned

        # cut it out if no response returned
        if not results:
            msg = cl.Message(content="")
            response = await async_generate_chainlit(
                'prompts/yelp_response_no_results.prompt',
                {'dlg': dlgHistory},
                msg,
                'gpt-3.5-turbo-0613',
                max_tokens=400,
                temperature=0.0,
                stop=[])
            await msg.send()
            
            dlgHistory[-1].agent_utterance = response
            dlgHistory[-1].time_statement = {
                "first_classification": first_classification_time,
                "semantic_parser": semantic_parser_time,
                "suql_execution": suql_execution_time,
                "final_response": final_response_time
            }
            return dlgHistory
    
    msg = cl.Message(content="")
    response = await async_generate_chainlit(
        'prompts/yelp_response_SQL.prompt',
        {'dlg': dlgHistory},
        msg,
        'gpt-3.5-turbo-0613',
        max_tokens=1000,
        temperature=0.0,
        stop=[])
    await msg.send()

    dlgHistory[-1].agent_utterance = response
    
    dlgHistory[-1].time_statement = {
        "first_classification": first_classification_time,
        "semantic_parser": semantic_parser_time,
        "suql_execution": suql_execution_time,
        "final_response": final_response_time
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
