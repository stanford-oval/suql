
 

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
from parser_server import GPT_parser_address
import time
from postgresql_connection import execute_sql
# from query_reviews import review_server_address
from langchain.output_parsers import CommaSeparatedListOutputParser
from sql_free_text_support.execute_free_text_sql import SelectVisitor
from pglast import parse_sql
from pglast.stream import RawStream


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from prompt_continuation import llm_generate, batch_llm_generate

logger = logging.getLogger(__name__)

def fetch_cuisine_list():
    try:
        with open("cuisines.json", "r") as fd:
            data = json.load(fd)
    
        res = list(data)
    except Exception as e:
        print("regenerating cuisines list from sparql query")
        res = execute_sql("SELECT DISTINCT unnest(cuisines) FROM restaurants;")
        res = list(map(lambda x: x[0], res[0]))
        
        with open("cuisines.json", "w") as fd:
            json.dump(res, fd)
        
    return sorted(res)

CUISINE_LIST = fetch_cuisine_list()


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
        if len(self.genie_reviews) > 0:
            ret += '\nRestaurant reviews: ['
            for i, review in enumerate(self.genie_reviews):
                ret += '\nReview ' + str(i+1) + ': "' + review + '"'
            if self.reviews_query is not None:
                ret += '\nQuestion: "' + self.reviews_query + '"'
            if self.genie_reviews_answer is not None:
                ret += '\nAnswer: "' + self.genie_reviews_answer + '"'
            ret += '\n]'
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


def extract_quotation(s: str) -> str:
    start = s.find('"')
    end = s.find('"', start+1)
    if start < 0 or end <= start:
        # try some heuristics for the case where LLM doesn't generate the quotation marks
        # TODO add more heuristics if needed
        if s.startswith('Yes. You check the database for '):
            return s[len('Yes. You check the database for '): ]

        # if everything fails, raise an error
        raise ValueError('Quotation error while parsing string %s' % s)
    return s[start+1: end]

def get_yelp_reviews(restaurant_id: str) -> List[str]:

    url = 'https://api.yelp.com/v3/businesses/%s/reviews?sort_by=yelp_sort' % restaurant_id
    logger.info('url for reviews: %s', url)
    
    reviews_response = requests.get(url=url, headers={'accept': 'application/json', 'Authorization': 'Bearer ' + os.getenv('YELP_API_KEY')})
    reviews_response = reviews_response.json()['reviews']
    reviews = []
    for r in reviews_response:
        reviews.append(html.unescape(' '.join(r['text'].split()))) # clean up the review text
    return reviews

def get_field_information(name, operator, value, user_target):
    # some heurstics to determine if the substring occurs inside `[` and `]`
    def determine_in_brakets(index):
        index_copy = index
        found = False
        while (index >= 0):
            if user_target[index] == "[":
                found = True
                break
            elif user_target[index] == "]":
                return False
            index -= 1
        
        if not found:
            return False
        
        found = False
        index = index_copy
        while (index < len(user_target)):
            if user_target[index] == "]":
                found = True
                break
            elif user_target[index] == "[":
                return False
            index += 1
            
        if not found:
            return False
        
        return True
    
    res = []
    target_substring = "{} {} {}".format(name, operator, value)
    
    start_index = 0
    index = user_target.find(target_substring, start_index)
    while index >= 0:
        
        to_append = {
            "name": name,
            "operator": operator,
            "value": value
            }

        if_projection = determine_in_brakets(index)
        if if_projection:
            to_append["type"] = "projection"
        else:
            to_append["type"] = "filter"
        
        if (to_append not in res):
            res.append(to_append)
        
        start_index = index + 1
        index = user_target.find(target_substring, start_index)
    
    return res

# a custom function to define what to include in the final response prompt
# this function is used for the restuarants domain, with some processing code
# to deal with `rating` (a float issue)
# and `opening_hours` (represented as a dictionary, but is too long)
def clean_up_response(results, column_names):
    final_res = []
    for res in results:
        temp = dict((column_name, result) for column_name, result in zip(column_names, res) if if_usable_restaurants(column_name))
        if "rating" in temp:
            temp["rating"] = float(temp["rating"])
            
        if "opening_hours" in temp:
            temp["opening_hours"] = handle_opening_hours(temp["opening_hours"])
            
        
        # here is some simple heuristics to deal with too long DB results,
        # thus cutting it at some point
        if num_tokens_from_string(json.dumps(final_res + [temp], indent=4)) > 3500:
            break
        
        final_res.append(temp)
    return final_res

def parse_execute_sql(dlgHistory, user_query, prompt_file='prompts/parser_sql.prompt'):
    first_sql, first_sql_time = llm_generate(template_file=prompt_file,
                engine='gpt-35-turbo',
                stop_tokens=["Agent:"],
                max_tokens=300,
                temperature=0,
                prompt_parameter_values={'dlg': dlgHistory, 'query': user_query},
                postprocess=False)
    second_sql = first_sql.replace("\\'", "''")
    
    print("directly generated SUQL query: {}".format(second_sql))
    
    second_sql_start_time = time.time()

    if not ("LIMIT" in second_sql):
        second_sql = re.sub(r';$', ' LIMIT 3;', second_sql, flags=re.MULTILINE)
    
    cache = {}
    try:
        visitor = SelectVisitor()
        root = parse_sql(second_sql)
        visitor(root)
        second_sql = RawStream()(root)
        cache = visitor.serialize_cache()
        
        print("intermediate (temp) SUQL query executed at the end: {}".format(second_sql))
    
        results, column_names, _ = execute_sql(second_sql)

        # some custom processing code to clean-up results
        final_res = clean_up_response(results, column_names)
    except Exception:
        visitor.drop_tmp_tables()
    finally:
        visitor.drop_tmp_tables()
    second_sql_end_time = time.time()
    
    return final_res, first_sql, second_sql, first_sql_time, second_sql_end_time - second_sql_start_time, cache

def turn_db_results2name(db_results):
    res = []
    for i in db_results:
        if "name" in i:
            res.append(i["name"])
    return res

def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str,
    engine = "text-davinci-003",
    sys_type = "sql_textfcns_v0801"):
    
    print(sys_type)
    assert(sys_type in ["sql_textfcns_v0801", "semantic_index_w_textfncs", "baseline_linearization"])
    
    first_classification_time = 0
    semantic_parser_time = 0
    suql_execution_time = 0
    final_response_time = 0
    cache = {}

    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))
    dlgHistory[-1].sys_type = sys_type
    
    # determine whether to send to Genie
    continuation, first_classification_time = llm_generate(template_file='prompts/if_db_classification.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine='gpt-35-turbo',
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)

    if continuation.startswith("Yes"):
        if sys_type == "sql_textfcns_v0801":
            results, first_sql, second_sql, semantic_parser_time, suql_execution_time, cache = parse_execute_sql(dlgHistory, user_utterance, prompt_file='prompts/parser_sql.prompt')
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
            dlgHistory[-1].user_target = first_sql
            dlgHistory[-1].temp_target = second_sql
            dlgHistory[-1].db_results = turn_db_results2name(results)
        
        elif sys_type == "semantic_index_w_textfncs":
            results, first_sql, second_sql, semantic_parser_time, suql_execution_time, cache = parse_execute_sql(dlgHistory, user_utterance, prompt_file='prompts/parser_sql_semantic_index.prompt')
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
            dlgHistory[-1].user_target = first_sql
            dlgHistory[-1].temp_target = second_sql
            dlgHistory[-1].db_results = turn_db_results2name(results)
            
        elif sys_type == "baseline_linearization":
            from reviews_server import baseline_filter
            
            results = baseline_filter(user_utterance)
            dlgHistory[-1].temp_target = None
            dlgHistory[-1].user_target = None
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)

        # for all systems, cut it out if no response returned
        if not results:
            response, final_response_time = llm_generate(template_file='prompts/yelp_response_no_results.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine='gpt-35-turbo',
                                max_tokens=400, temperature=0.0, stop_tokens=[], top_p=0.5, postprocess=False)
            dlgHistory[-1].agent_utterance = response
            dlgHistory[-1].time_statement = {
                "first_classification": first_classification_time,
                "semantic_parser": semantic_parser_time,
                "suql_execution": suql_execution_time,
                "final_response": final_response_time
            }
            return dlgHistory
            
    response, final_response_time = llm_generate(template_file='prompts/yelp_response_SQL.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine='gpt-35-turbo',
                        max_tokens=400, temperature=0.0, stop_tokens=[], top_p=0.5, postprocess=False)
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
    parser.add_argument('--greeting', type=str, default="Hi! How can I help you?", help="The first thing the agent says to the user")
    parser.add_argument('--output_file', type=str, default='log.log',
                        help='Where to write the outputs.')
    parser.add_argument('--engine', type=str, default='text-davinci-003',
                        choices=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'gpt-35-turbo'],
                        help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
    parser.add_argument('--quit_commands', type=str, default=['quit', 'q'],
                        help='The conversation will continue until this string is typed in.')
    parser.add_argument('--no_logging', action='store_true',
                        help='Do not output extra information about the intermediate steps.')
    parser.add_argument('--use_GPT_parser', action='store_true',
                        help='Use GPT parser as opposed to Genie parser')
    parser.add_argument('--use_direct_sentence_state', action='store_true',
                        help='Directly use GPT parser output as full state')
    parser.add_argument('--sys_type', type=str, default='sql_textfcns_v0801',
                        choices=["sql_textfcns_v0801", "semantic_index_w_textfncs", "baseline_linearization"])
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
                engine=args.engine,
                sys_type=args.sys_type
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
            
            # this is single-user, so feeding in genieDS and genie_aux is unnecessary, but we do it to be consistent with backend_connection.py
            dlgHistory = compute_next_turn(
                dlgHistory,
                user_utterance,
                engine=args.engine,
                sys_type=args.sys_type
            )
            print_chatbot('Chatbot: ' + dlgHistory[-1].agent_utterance)
            print(dlgHistory[-1].time_statement)

    finally:
        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(dlgHistory, they='User', you='Chatbot') + '\n')