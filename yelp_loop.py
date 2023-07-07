
 

"""
GPT-3 + Yelp Genie skill
"""

import sys
import os
from typing import List
import argparse
import logging
import requests
from datetime import datetime
import html
import json
from utils import print_chatbot, input_user, num_tokens_from_string
import readline  # enables keyboard arrows when typing in the terminal
from parser_server import GPT_parser_address
import time
from postgresql_connection import execute_sql
# from query_reviews import review_server_address
from reviews_server import baseline_filter


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from prompt_continuation import llm_generate, batch_llm_generate

logger = logging.getLogger(__name__)


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
        genie_query: str = None,
        genie_utterance: str = None,
        reviews_query: str = None,
        genie_reviews : List[str] = [],
        genie_reviews_answer : str = None,
        user_target : str = None,
    ):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.genie_query = genie_query
        self.genie_utterance = genie_utterance
        self.reviews_query = reviews_query
        self.genie_reviews = genie_reviews
        self.genie_reviews_answer = genie_reviews_answer
        self.user_target = user_target
        
    agent_utterance: str
    user_utterance: str
    genie_query: str
    genie_utterance: str
    reviews_query: str
    genie_reviews: List[str]
    genie_reviews_answer: str
    user_target: str

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
        if self.genie_query is not None:
            ret += '\n' + '[You check the database for "' + \
                self.genie_query + '"]'
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


def summarize_reviews(reviews: str) -> str:
    summaries = batch_llm_generate(template_file='prompts/yelp_review_summary.prompt',
                           engine='text-davinci-003',
                           stop_tokens=None,
                           max_tokens=100,
                           temperature=0.7,
                           prompt_parameter_values=[{'review': r} for r in reviews],
                           postprocess=False)
    return summaries


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

def review_qa(reviews, question, engine):
    review_res = []
    # TODO: to be precise one needs to use the openAI tokenzer. for now I am just using some
    # ad-hoc hard-coded character count
    for i in reviews:
        if len('\n'.join(review_res + [i])) < 14000:
            review_res.append(i)
        else:
            break
    
    if not review_res or (len(review_res) == 1 and not review_res[0]):
        return ""
    
    continuation, elapsed_time = llm_generate(
        'prompts/review_qa.prompt',
        {'reviews': review_res, 'question': question},
        engine=engine,
        max_tokens=200,
        temperature=0.0,
        stop_tokens=['\n'],
        postprocess=False
    )
    
    return continuation, elapsed_time

def parse_execute_sql(dlgHistory, user_query, prompt_file='prompts/parser_sql.prompt'):
    continuation, _ = llm_generate(template_file=prompt_file,
                engine='gpt-35-turbo',
                stop_tokens=["Agent:"],
                max_tokens=300,
                temperature=0,
                prompt_parameter_values={'dlg': dlgHistory, 'query': user_query},
                postprocess=False)
    
    NOT_USABLE_FIELDS = [
        "reviews",
        "_id",
        "id",
        "opening_hours",

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
        "accomodates_large_groups_citation"
        ]
     
    continuation = continuation.rstrip("Agent:")
    # print("generated SQL query {}".format(continuation))
    results, column_names = execute_sql(continuation)

    final_res = []
    for res in results:
        temp = dict((column_name, result) for column_name, result in zip(column_names, res) if column_name not in NOT_USABLE_FIELDS)
        if "rating" in temp:
            temp["rating"] = float(temp["rating"])
        
        if num_tokens_from_string(json.dumps(final_res + [temp], indent=4)) > 3500:
            break
        
        final_res.append(temp)
    
    print(final_res)
    return final_res, continuation

def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str,
    engine = "text-davinci-003",
    sys_type = 'baseline_w_textfcns'):
    
    print(sys_type)
    assert(sys_type in ["semantic_index", "baseline_w_textfcns", "baseline_linearization", "v0614baseline_thingtalk"])
    
    # assign default values
    genie_new_ds = None
    genie_new_aux = []
    genie_user_target = ""
    genie_results = []
    
    first_classification_time = 0
    review_time = 0
    final_response_time = 0
    genie_time = 0

    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))
    dlgHistory[-1].reviews_query = None
    
    # determine whether to send to Genie
    continuation, first_classification_time = llm_generate(template_file='prompts/yelp_genie.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)

    if continuation.startswith("Yes"):
        dlgHistory[-1].genie_query = user_utterance
        if sys_type == "baseline_w_textfcns":
            results, sql = parse_execute_sql(dlgHistory, user_utterance, prompt_file='prompts/parser_sql.prompt')
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
            dlgHistory[-1].user_target = sql
            genie_user_target = sql
            if not results:
                response = "Sorry, I don't have that information."
                dlgHistory[-1].agent_utterance = response
                return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target, ""
        
        elif sys_type == 'semantic_index':
            results, sql = parse_execute_sql(dlgHistory, user_utterance, prompt_file='prompts/parser_sql_semantic_index.prompt')
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
            dlgHistory[-1].user_target = sql
            genie_user_target = sql
            if not results:
                response = "Sorry, I don't have that information."
                dlgHistory[-1].agent_utterance = response
                return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target, ""
            
        elif sys_type == 'baseline_linearization':
            results = baseline_filter(user_utterance)
            sql = None
            dlgHistory[-1].genie_utterance = json.dumps(results, indent=4)
            dlgHistory[-1].user_target = sql
            genie_user_target = sql

            if not results:
                response = "Sorry, I don't have that information."
                dlgHistory[-1].agent_utterance = response
                return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target, ""
            
    response, final_response_time = llm_generate(template_file='prompts/yelp_response_SQL.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                        max_tokens=150, temperature=0.0, stop_tokens=['\n'], top_p=0.5, postprocess=False)
    dlgHistory[-1].agent_utterance = response
    dlgHistory[-1].user_target = genie_user_target
    
    time_stmt = [
        "Initial classifier: {:.2f}s".format(first_classification_time), 
        "Genie (w. semantic parser + review model): {:.2f}s".format(genie_time),
        "Review QA: {:.2f}s".format(review_time),
        "Final response: {:.2f}s".format(final_response_time)
    ]
    
    
    return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target, time_stmt

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--greeting', type=str, default="Hi! How can I help you?", help="The first thing the agent says to the user")
    parser.add_argument('--output_file', type=str, required=True,
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
    parser.add_argument('--sys_type', type=str, default='generate_sql',
                        choices=["semantic_index", "baseline_w_textfcns", "baseline_linearization", "v0614baseline_thingtalk"])
    # parser.add_argument('--use_sql', action='store_true',
    #                     help='Uses sql generation')
    # parser.add_argument('--use_baseline', action=)

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')

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
            dlgHistory, response, gds, gaux, _, _ = compute_next_turn(
                dlgHistory,
                user_utterance,
                engine=args.engine,
                sys_type=args.sys_type
            )
            if genieDS != None:
                # update the genie state only when it is called. This means that if genie is not called in one turn, in the next turn we still provide genie with its state from two turns ago
                genieDS = gds
                genie_aux = gaux
            print_chatbot('Chatbot: ' + response)

    finally:
        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(dlgHistory, they='User', you='Chatbot') + '\n')