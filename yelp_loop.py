
 

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
        genie_reviews_answer : str = None
        ):
        
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.genie_query = genie_query
        self.genie_utterance = genie_utterance
        self.reviews_query = reviews_query
        self.genie_reviews = genie_reviews
        self.genie_reviews_answer = genie_reviews_answer

    agent_utterance: str
    user_utterance: str
    genie_query: str
    genie_utterance: str
    reviews_query: str
    genie_reviews: List[str]
    genie_reviews_answer: str

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


def wrapper_call_genie(
    genie : gs,
    dlgHistory : List[DialogueTurn],
    query: str,
    dialog_state = None,
    aux = [],
    engine = "text-davinci-003",
    update_parser_address = None
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
        ds, aux, user_target, genie_results = call_genie_internal(genie, dlgHistory, query, dialog_state = dialog_state, aux = aux, update_parser_address=update_parser_address)
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
    update_parser_address = None
):
    if (update_parser_address is not None):
        requests.post(update_parser_address + '/set_dlg_turn', json={
            "dlg_turn": list(map(lambda x: x.__dict__, dlgHistory))
        })
    
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


def retrieve_last_reviews(dlgHistory : List[DialogueTurn]):
    for i in reversed(dlgHistory):
        if len(i.genie_reviews) > 0:
            return i.genie_reviews
    return []

def compute_next_turn(
    dlgHistory : List[DialogueTurn],
    user_utterance: str,
    genie : gs,
    genieDS : str = None,
    genie_aux = [],
    engine = "text-davinci-003",
    update_parser_address = None):
    
    # assign default values
    genie_new_ds = "null"
    genie_new_aux = []
    genie_user_target = ""
    genie_results = []
    
    dlgHistory.append(DialogueTurn(user_utterance=user_utterance))
    dlgHistory[-1].reviews_query = None
    
    # determine whether to send to Genie      
    continuation = llm_generate(template_file='prompts/yelp_genie.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)
    if continuation.startswith("Yes"):
        try:
            genie_query = extract_quotation(continuation)
            dlgHistory[-1].genie_query = genie_query
            genie_new_ds, genie_new_aux, genie_user_target, genie_results = wrapper_call_genie(
                genie, dlgHistory, genie_query, genieDS, aux=genie_aux, engine=engine, update_parser_address=update_parser_address)

        except ValueError as e:
            logger.error('%s', str(e))
        
        if len(genie_results) == 0 and genie_new_ds is not None:
            response = "Sorry, I don't have that information."
            dlgHistory[-1].agent_utterance = response
            return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target
    
    # determine whether to Q&A reviews
    continuation = llm_generate(template_file='prompts/yelp_review.prompt', prompt_parameter_values={'dlg': dlgHistory, "dlg_turn": dlgHistory[-1]}, engine=engine,
                                max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)
    if continuation.startswith("Yes"):
        try:
            review_query = extract_quotation(continuation)
            dlgHistory[-1].reviews_query = review_query
            
            if len(genie_results) > 0:
                reviews = get_yelp_reviews(genie_results[0]['id']['value'])
                reviews = reviews[:3] # at most 3 reviews
            else:
                reviews =  retrieve_last_reviews(dlgHistory)
            dlgHistory[-1].genie_reviews = reviews
            
            # if there are reviews to be queried, do a GPT-3 QA system on it
            if len(reviews) > 0:
                continuation = llm_generate(template_file='prompts/review_qa.prompt', prompt_parameter_values={'review': reviews, "question": review_query}, engine=engine,
                                max_tokens=100, temperature=0.0, stop_tokens=['\n'], postprocess=False)
                dlgHistory[-1].genie_reviews_answer = continuation
            
        except ValueError as e:
            logger.error('%s', str(e))
            
    response = llm_generate(template_file='prompts/yelp_response.prompt', prompt_parameter_values={'dlg': dlgHistory}, engine=engine,
                        max_tokens=150, temperature=0.0, stop_tokens=['\n'], top_p=0.5, postprocess=False)
    dlgHistory[-1].agent_utterance = response
    
    return dlgHistory, response, genie_new_ds, genie_new_aux, genie_user_target

        

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
    GPT_parser_address = 'http://127.0.0.1:8400'

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')

    # The dialogue loop
    # the agent starts the dialogue
    genie = gs.Genie()
    dlgHistory = []
    genieDS, genie_aux = "null", []

    print_chatbot(dialogue_history_to_text(dlgHistory, they='User', you='Chatbot'))

    try:
        genie.initialize(GPT_parser_address, 'yelp')

        while True:
            user_utterance = input_user()
            if user_utterance in args.quit_commands:
                break
            
            # this is single-user, so feeding in genieDS and genie_aux is unnecessary, but we do it to be consistent with backend_connection.py
            dlgHistory, response, gds, gaux, _ = compute_next_turn(
                dlgHistory,
                user_utterance,
                genie,
                genieDS=genieDS,
                genie_aux=genie_aux,
                engine=args.engine,
                update_parser_address=GPT_parser_address if args.use_GPT_parser else None
            )
            if genieDS != 'null':
                # update the genie state only when it is called. This means that if genie is not called in one turn, in the next turn we still provide genie with its state from two turns ago
                genieDS = gds
                genie_aux = gaux
            print_chatbot('Chatbot: ' + response)

    finally:
        # not necessary to close genie, but is good practice
        genie.quit()

        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(dlgHistory, they='User', you='Chatbot') + '\n')