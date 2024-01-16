
 

"""
GPT-3 + Yelp Genie skill
"""

import sys
import os
import re
from typing import List
import argparse
import logging
import requests
from datetime import datetime
import html
import json
from utils import print_chatbot, input_user, num_tokens_from_string, if_usable_restaurants, handle_opening_hours
import readline  # enables keyboard arrows when typing in the terminal
import time
from postgresql_connection import execute_sql
# from query_reviews import review_server_address
from sql_free_text_support.execute_free_text_sql import suql_execute
from pglast import parse_sql
from pglast.stream import RawStream
from decimal import Decimal
import chainlit as cl

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from prompt_continuation import llm_generate, async_generate_chainlit

logger = logging.getLogger(__name__)

class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        genie_utterance: str = None,
        temp_target : str = None,
        user_target : str = None,
        sys_type : str = None,
        time_statement : dict = None,
        db_results : list = [],
        cache : dict = {},
    ):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.genie_utterance = genie_utterance
        self.temp_target = temp_target
        self.user_target = user_target
        self.sys_type = sys_type
        self.db_results = db_results
        time_statement = time_statement
        
    agent_utterance: str
    user_utterance: str
    genie_utterance: str
    user_target: str
    temp_target: str
    sys_type: str
    time_statement: dict
    db_results: list

    def to_text(self, they='They', you='You'):
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
        ret = ''
        
        ret += they + ': ' + self.user_utterance
        if self.genie_utterance is not None:
            # print(self.genie_utterance)
            ret += '\n' + '[Database returns "' + self.genie_utterance + '"]'
        if self.agent_utterance is not None:
            ret += '\n' + you + ': ' + self.agent_utterance
        return ret


def dialogue_history_to_text(history: List[DialogueTurn], they='They', you='You') -> str:
    """
    From the agent's point of view, it is 'You:'. The agent starts the conversation.
    """
    ret = ''
    for i in range(len(history)):
        ret += '\n' + history[i].to_text(they=they, you=you)

    if len(history) > 0:
        # remove the extra starting newline
        if ret[0] == '\n':
            ret = ret[1:]

    return ret

# a custom function to define what to include in the final response prompt
# this function is used for the restuarants domain, with some processing code
# to deal with `rating` (a float issue)
# and `opening_hours` (represented as a dictionary, but is too long)
def clean_up_response(results, column_names):
    final_res = []
    for res in results:
        temp = dict((column_name, result) for column_name, result in zip(column_names, res) if if_usable_restaurants(column_name))
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

def json_to_markdown_table(data):
    # Assuming the JSON data is a list of dictionaries
    # where each dictionary represents a row in the table
    if not data or not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON data is not in the expected format for a table")

    # Extract headers
    headers = data[0].keys()

    # Start building the Markdown table
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["-" * len(header) for header in headers]) + " |\n"

    # Add rows
    for row in data:
        markdown_table += "| " + " | ".join(str(row[header]) for header in headers) + " |\n"

    return markdown_table

async def generate_sql(dlgHistory, user_query):
    async with cl.Step(name="SUQL", type="llm", language="sql", disable_feedback=False) as step:
        first_sql = await async_generate_chainlit(
            'prompts/parser_sql.prompt',
            {'dlg': dlgHistory, 'query': user_query},
            step,
            'gpt-3.5-turbo-0613',
            max_tokens=300,
            temperature=0,
            stop=["Agent:"])
    
    second_sql = first_sql.replace("\\'", "''")
    if not ("LIMIT" in second_sql):
        second_sql = re.sub(r';$', ' LIMIT 3;', second_sql, flags=re.MULTILINE)
    
    return second_sql

async def execute_sql(sql):
    suql_execute_start_time = time.time()
    async with cl.Step(name="Results", type="llm", disable_feedback=False) as step:
        final_res, column_names, cache = suql_execute(sql)
        final_res = clean_up_response(final_res, column_names)
        if final_res:
            step.output = json_to_markdown_table(final_res)
        else:
            step.output = "SUQL returned no results"
        
    suql_execute_end_time = time.time()
    
    return final_res, suql_execute_end_time - suql_execute_start_time

def turn_db_results2name(db_results):
    res = []
    for i in db_results:
        if "name" in i:
            res.append(i["name"])
    return res

async def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str):
    
    first_classification_time = 0
    semantic_parser_time = 0
    suql_execution_time = 0
    final_response_time = 0
    cache = {}

    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))
    dlgHistory[-1].sys_type = "suql_v0102"
    
    # determine whether to send to Genie
    continuation, first_classification_time = llm_generate(template_file='prompts/if_db_classification.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine='gpt-3.5-turbo-0613',
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)

    if continuation.startswith("Yes"):
        first_sql = await generate_sql(dlgHistory, user_utterance)
        results, suql_execution_time = await execute_sql(first_sql)
        dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
        dlgHistory[-1].user_target = first_sql
        dlgHistory[-1].temp_target = ""
        dlgHistory[-1].db_results = turn_db_results2name(results)

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
    dlgHistory[-1].cache = cache
    
    return dlgHistory

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='log.log',
                        help='Where to write the outputs, pertaining only to CLI testing.')
    parser.add_argument('--quit_commands', type=str, default=['quit', 'q'],
                        help='The conversation will continue until this string is typed in, pertaining only to CLI testing.')
    parser.add_argument('--no_logging', action='store_true',
                        help='Do not output extra information about the intermediate steps.')
    parser.add_argument('--record_result', type=str, default=None, help='Write results in TSV format to file')
    parser.add_argument('--batch_process', type=str, default=None, help='A list of QA inputs to run')

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')


    if args.batch_process:
        assert(args.record_result)
        
        with open(args.batch_process, "r") as fd:
            inputs = fd.readlines()
        for each_input in inputs:
            id, utterance = each_input.split('\t')
            utterance = utterance.strip()
            
            dlgHistory = []
            genieDS, genie_aux = None, []
            dlgHistory = compute_next_turn(
                dlgHistory,
                utterance,
            )
            with open(args.record_result, 'a+') as fd:
                fd.write("{}\t{}\t{}\t{}\t{}\n".format(id, dlgHistory[-1].user_utterance, dlgHistory[-1].user_target, dlgHistory[-1].agent_utterance, '\t'.join(dlgHistory[-1].db_results)))

    # The dialogue loop
    # the agent starts the dialogue
    dlgHistory = []
    genieDS, genie_aux = None, []

    print_chatbot(dialogue_history_to_text(dlgHistory, they='User', you='Chatbot'))

    try:
        while True:
            user_utterance = input_user()
            if user_utterance in args.quit_commands:
                break
            
            dlgHistory = compute_next_turn(
                dlgHistory,
                user_utterance,
            )
            print_chatbot('Chatbot: ' + dlgHistory[-1].agent_utterance)
            print(dlgHistory[-1].time_statement)

    finally:
        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(dlgHistory, they='User', you='Chatbot') + '\n')