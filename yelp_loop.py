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
from utils import print_chatbot, input_user
import readline  # enables keyboard arrows when typing in the terminal
from pyGenieScript import geniescript as gs
from pymongo import MongoClient, ASCENDING
import json

# set up the MongoDB connection
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")

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
        genie_reviews_summary : List[str] = []
        ):
        
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.genie_query = genie_query
        self.genie_utterance = genie_utterance
        self.reviews_query = reviews_query,
        self.genie_reviews = genie_reviews
        self.genie_reviews_summary = genie_reviews_summary

    agent_utterance: str
    user_utterance: str
    genie_query: str
    genie_utterance: str
    reviews_query: str
    genie_reviews: List[str]
    genie_reviews_summary: List[str]

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
        ret += you + ': ' + self.agent_utterance
        if self.user_utterance is not None:
            ret += '\n' + they + ': ' + self.user_utterance
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
                if len(self.genie_reviews_summary) > 0:
                    ret += '\nSummary: "' + self.genie_reviews_summary[i] + '"'
            ret += '\n]'
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


def wrapper_call_genie(
    genie : gs,
    dlgHistory : List[DialogueTurn],
    query: str,
    dialog_state = None,
    aux = [],
    engine = "text-davinci-003"
):
    """A wrapper around the Genie semantic parser, to determine if info if present in the database

    Args:
        genie (gs): pyGenieScript.geniescript.Genie class
        dlgHistory (List[DialogueTurn]): list of dlgHistory, to be modified in place
        query (str): query to be sent to Genie, if info present in the database
        dialog_state (_type_, optional): Genie state. Defaults to None.
        aux (list, optional): Genie aux info. Defaults to [].
        engine (str, optional): LLM engine. Defaults to "text-davinci-003".

    Returns:
        (dialog_state, aux, user_target, genie_results) from genie
    """
    continuation = llm_generate(
        template_file='prompts/genie_wrapper.prompt',
        prompt_parameter_values={'query': query},
        engine=engine,
        max_tokens=10, temperature=0.0, stop_tokens=['\n'], postprocess=False
    ).lower()
    
    if (continuation.startswith("yes")):
        ds, aux, user_target, genie_results = call_genie_internal(genie, dlgHistory, query, dialog_state = dialog_state, aux = aux)
        return ds, aux, user_target, genie_results
    else:
        dlgHistory[-1].genie_utterance = "I don't have that information"
        return None, [], "", []

def call_genie_internal(
    genie : gs,
    dlgHistory : List[DialogueTurn],
    query: str,
    dialog_state = None,
    aux = [],
):
    if dialog_state:
        genie_output = genie.query(query, dialog_state = dialog_state, aux=aux)
    else:
        genie_output = genie.query(query, aux=aux)

    if len(genie_output['response']) > 0:
        genie_utterance = genie_output['response'][0]
    else:
        # a semantic parser error
        genie_utterance = "Error parsing your request"
            
    dlgHistory[-1].genie_utterance = genie_utterance
    return genie_output["ds"], genie_output["aux"], genie_output["user_target"], genie_output["results"]

def reconstruct_dlgHistory(tuples):
    """Given a list of *sorted* mongodb dialog tuples, reconstruct a list of DialogTurns

    Args:
        tuples : a list of mongodb tuples, sorted in the order of first turn to last turn
    """
    return list(map(lambda x: DialogueTurn(**x["dlg_turn"]), tuples))

def reconstruct_genieinfo(tuples):
    """Given a list of *sorted* mongodb dialog tuples, find the latest Genie information

    Args:
        tuples :a list of mongodb tuples, sorted in the order of first turn to last turn
    """
    for i in reversed(tuples):
        if "genieDS" in i:
            return i["genieDS"], i["genieAux"]
    return "null", []

def retrieve_last_reviews(dlgHistory : List[DialogueTurn]):
    for i in reversed(dlgHistory):
        if len(i.genie_reviews) > 0:
            return i.genie_reviews, i.genie_reviews_summary
    return [], []

def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str,
    genie : gs,
    genieDS : str = None,
    genie_aux = [],
    engine = "text-davinci-003"):
    
    # assign default values
    genie_new_ds = "null"
    genie_new_aux = []
    genie_user_target = ""
    genie_results = []
    
    dlgHistory[-1].user_utterance = user_utterance
    dlgHistory[-1].reviews_query = None
    
    # determine whether to send to Genie      
    continuation = llm_generate(template_file='prompts/yelp_genie.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)
    if continuation.startswith("Yes"):
        try:
            genie_query = extract_quotation(continuation)
            dlgHistory[-1].genie_query = genie_query
            genie_new_ds, genie_new_aux, genie_user_target, genie_results = wrapper_call_genie(genie, dlgHistory, genie_query, genieDS, aux=genie_aux, engine=engine)

        except ValueError as e:
            logger.error('%s', str(e))
        
        # determine whether to query reviews
        if len(genie_results) > 0:
            continuation = llm_generate(template_file='prompts/yelp_review.prompt', prompt_parameter_values={'dlg': dlgHistory, "dlg_turn": dlgHistory[-1]}, engine=engine,
                                        max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)
            if continuation.startswith("Yes"):
                try:
                    review_query = extract_quotation(continuation)
                    dlgHistory[-1].reviews_query = review_query

                    reviews = get_yelp_reviews(genie_results[0]['id']['value'])
                    reviews = reviews[:3] # at most 3 reviews
                    dlgHistory[-1].genie_reviews = reviews
                    dlgHistory[-1].genie_reviews_summary = summarize_reviews(reviews)
                    
                except ValueError as e:
                    logger.error('%s', str(e))
            response = llm_generate(template_file='prompts/yelp_response.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                                max_tokens=70, temperature=0.7, stop_tokens=['\n'], top_p=0.5, postprocess=False)
        else:
            # for now, just directly bypass retrieving from reviews
            response = "Sorry, I don't have that information."
    else:
        response = llm_generate(template_file='prompts/yelp_response.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                                max_tokens=70, temperature=0.7, stop_tokens=['\n'], top_p=0.5, postprocess=False)
        logging.info('Nothing to send to Genie')
    
        
    dlgHistory.append(DialogueTurn(agent_utterance=response))
    
    return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target

class BackendConnection:
    def __init__(
        self,
        greeting = "Hi! How can I help you?",
        engine = "text-davinci-003") -> None:
        
        self.genie = gs.Genie()
        self.genie.initialize('localhost', 'yelp')
        
        client = MongoClient(CONNECTION_STRING)
        self.db = client['yelpbot']  # the database name is yelpbot
        self.table = self.db['dialog_turns_dev'] # the collection that stores dialog turns
        self.table.create_index("$**") # necessary to build an index before we can call sort()

        self.greeting = greeting
        self.engine = engine
        
    def compute_next(self, dialog_id, user_utterance, turn_id):
        tuples = list(self.table.find( { "dialogID": dialog_id } ).sort('turn_id', ASCENDING))
        
        # for first turn we initiate dlgHistory as greeting
        # this self.greeting msg is matched in the front end manually for now
        if (not tuples):
            dlgHistory = [DialogueTurn(agent_utterance=self.greeting)]
            genieDS, genie_aux = "null", []
        # otherwise we retrieve the dialog history
        else:
            dlgHistory = reconstruct_dlgHistory(tuples)
            genieDS, genie_aux = reconstruct_genieinfo(tuples)

        dlgHistory, response, genieDS, genie_aux, genie_user_target = compute_next_turn(dlgHistory, user_utterance, self.genie, genieDS=genieDS, genie_aux=genie_aux, engine=self.engine)
        
        # update the current tuple with new DialogTurn
        update_tuple = {"dialogID": dialog_id, "turn_id": turn_id - 1, "dlg_turn" : dlgHistory[-2].__dict__}
        # if current tuple has genie information, updates it as well
        if genieDS != 'null':
            update_tuple.update({"genieDS" : genieDS, "genieAux": genie_aux, "genie_user_target": genie_user_target})
        if (not tuples):
            self.table.insert_one(update_tuple)
        else:
            self.table.update_one( {"dialogID": dialog_id, "turn_id": turn_id - 1}, {"$set": update_tuple} )
        
        # insert a new tuple for next turn usage
        new_tuple = {"dialogID": dialog_id, "dlg_turn" : dlgHistory[-1].__dict__ , "turn_id": turn_id }
        self.table.insert_one(new_tuple)
        
        return response, dlgHistory[-2]
        

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

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')

    # The dialogue loop
    # the agent starts the dialogue
    new_dlg = [DialogueTurn(agent_utterance=args.greeting)]
    print_chatbot(dialogue_history_to_text(
        new_dlg, they='User', you='Chatbot'))

    genie = gs.Genie()

    try:
        genie.initialize('localhost', 'yelp')

        while True:
            user_utterance = input_user()
            if user_utterance in args.quit_commands:
                break
            
            new_dlg, response, _, _, _ = compute_next_turn(new_dlg, user_utterance, genie, engine=args.engine)
            print_chatbot('Chatbot: ' + response)

    finally:
        # not necessary to close genie, but is good practice
        genie.quit()

        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(new_dlg, they='User', you='Chatbot') + '\n')


    # # determine whether to query reviews
    # continuation = llm_generate(template_file='prompts/yelp_review.prompt', prompt_parameter_values={'dlg': dlgHistory, "dlg_turn": dlgHistory[-1]}, engine=engine,
    #                             max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)
    # if continuation.startswith("Yes"):
    #     try:
    #         review_query = extract_quotation(continuation)
    #         dlgHistory[-1].reviews_query = review_query

    #         # if genie returned results, then we fetch reviews from there
    #         if len(genie_results) > 0:
    #             reviews = get_yelp_reviews(genie_results[0]['id']['value'])
    #             reviews = reviews[:3] # at most 3 reviews
    #             dlgHistory[-1].genie_reviews = reviews
    #             dlgHistory[-1].genie_reviews_summary = summarize_reviews(reviews)
            
    #         # otherwise, directly use the last available reviews
    #         else:
    #             reviews, review_summaries =  retrieve_last_reviews(dlgHistory)
    #             dlgHistory[-1].genie_reviews = reviews
    #             dlgHistory[-1].genie_reviews_summary = review_summaries
                
    #     except ValueError as e:
    #         logger.error('%s', str(e))
    # response = llm_generate(template_file='prompts/yelp_response.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
    #                         max_tokens=70, temperature=0.7, stop_tokens=['\n'], top_p=0.5, postprocess=False)