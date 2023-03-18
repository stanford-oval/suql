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
    def __init__(self, agent_utterance: str = None, user_utterance: str = None):
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance
        self.genie_query = None
        self.genie_utterance = None
        self.genie_reviews = []
        self.genie_reviews_summary = []

    agent_utterance: str
    user_utterance: str
    genie_query: str
    genie_utterance: str
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

def call_genie(genie, query: str):
    """
    Calls GenieScript, and fetches Yelp reviews if needed
    """
    genie_output = genie.query(query)
    logger.info('genie_output[response] = %s', genie_output['response'])

    if len(genie_output['response']) > 0:
        genie_utterance = genie_output['response'][0]
    else:
        # a semantic parser error
        genie_utterance = "Error parsing your request"

    reviews = []
    if 'results' in genie_output and len(genie_output['results']) > 0:
        reviews = get_yelp_reviews(restaurant_id=genie_output['results'][0]['id']['value'])

    reviews = reviews[:3] # at most 3 reviews
    return genie_utterance, reviews


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--greeting', type=str, default="Hi! How can I help you?",
                        help='Where to read the partial conversations from.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Where to write the outputs.')
    parser.add_argument('--engine', type=str, default='text-curie-001',
                        choices=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'gpt-35-turbo'],
                        help='The GPT-3 engine to use. (default: text-curie-001)')  # choices are from the smallest to the largest model
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
            new_dlg[-1].user_utterance = user_utterance

            continuation = llm_generate(template_file='prompts/yelp_genie.prompt', prompt_parameter_values={'dlg': new_dlg}, engine=args.engine,
                                        max_tokens=50, temperature=0.0, stop_tokens=['\n'], postprocess=False)

            if continuation.startswith("Yes"):
                try:
                    genie_query = extract_quotation(continuation)
                    genie_utterance, genie_reviews = call_genie(genie, genie_query)
                    logger.info('genie_utterance = %s, genie_reviews = %s', genie_utterance, str(genie_reviews))
                    new_dlg[-1].genie_query = genie_query
                    new_dlg[-1].genie_utterance = genie_utterance
                    new_dlg[-1].genie_reviews = genie_reviews
                    if len(genie_reviews) > 0:
                        new_dlg[-1].genie_reviews_summary = summarize_reviews(genie_reviews)
                except ValueError as e:
                    logger.error('%s', str(e))
            else:
                logging.info('Nothing to send to Genie')

            response = llm_generate(template_file='prompts/yelp_response.prompt', prompt_parameter_values={'dlg': new_dlg}, engine=args.engine,
                                    max_tokens=70, temperature=0.7, stop_tokens=['\n'], top_p=0.5, postprocess=False)

            new_dlg.append(DialogueTurn(agent_utterance=response))
            print_chatbot('Chatbot: ' + response)

    finally:
        # not necessary to close genie, but is good practice
        genie.quit()

        with open(args.output_file, 'a') as output_file:
            output_file.write('=====\n' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") +
                            '\n' + dialogue_history_to_text(new_dlg, they='User', you='Chatbot') + '\n')