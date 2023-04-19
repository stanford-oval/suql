"""
The backend API that runs dialog agents and returns agent utterance to the front-end.

The API has the following three functions that can be used by any front-end.
All inputs/outputs are string, except for `log_object` which is a json object and `turn_id` and `user_naturalness_rating` which are integers.
- `/chat`
Inputs: (experiment_id, new_user_utterance, dialog_id, turn_id, system_name)
Outputs: (agent_utterance, log_object)
Each time a user types something and clicks send, the front-end should make one call per system to /chat. So e.g. it should make two separate calls for two systems.

- `/user_rating`
Inputs: (experiment_id, dialog_id, turn_id, system_name, user_naturalness_rating)
Outputs: None
When the user submits their ratings, the front-end should make one call per system to /user_rating. So e.g. it should make two separate calls for two systems.

- `/user_preference`
Inputs: (experiment_id, dialog_id, turn_id, winner_system, loser_system)
Outputs: None
Each time the user selects one of the agent utterances over the other, you make one call to /user_preference.

`turn_id` starts from 0 and is incremented by 1 after a user and agent turn
"""

import argparse
import os
import string
import json
import logging

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Api, reqparse

from yelp_loop import *


app = Flask(__name__)
CORS(app)
api = Api(app)
logging.basicConfig(level=logging.INFO)
logger = app.logger

# The input arguments coming from the front-end
req_parser = reqparse.RequestParser()
req_parser.add_argument("experiment_id", type=str, location='json',
                        help='Identifier that differentiates data from different experiments.')
req_parser.add_argument("dialog_id", type=str, location='json',
                        help='Globally unique identifier for each dialog')
req_parser.add_argument("turn_id", type=int, location='json',
                        help='Turn number in the dialog')
req_parser.add_argument("user_naturalness_rating", type=int, location='json')
req_parser.add_argument("new_user_utterance", type=str,
                        location='json', help='The new user utterance')
req_parser.add_argument("system_name", type=str, location='json',
                        help='The system to use for generating agent utterances')

# arguments for when a user makes a head-to-head comparison
req_parser.add_argument("winner_system", type=str, location='json',
                        help='The system that was preferred by the user in the current dialog turn')
req_parser.add_argument("loser_systems", type=list, location='json',
                        help='The system(s) that was not preferred by the user in the current dialog turn')

connection = BackendConnection()

@app.route("/chat", methods=["POST"])
def chat():
    """
    Inputs: (experiment_id, new_user_utterance, dialog_id, turn_id, system_name)
    Outputs: (agent_utterance, log_object)
    """
    logger.info('Entered /chat')
    request_args = req_parser.parse_args()
    logger.info('Input arguments received: %s', str(request_args))

    user_utterance = request_args['new_user_utterance']
    dialog_id = request_args['dialog_id']
    turn_id = request_args['turn_id']
    # experiment_id = request_args['experiment_id']
    # system_name = request_args['system_name']
    
    response, dlgItem, genie_user_target = connection.compute_next(dialog_id, user_utterance, turn_id)

    log = {}
    if (dlgItem.genie_query):
        log["genie_query"] = dlgItem.genie_query
        log["genie_utterance"] = dlgItem.genie_utterance
        
        # do some processing on genie_user_target to only preserve the sentence state representation
        try:
            genie_user_target = "$continue" + genie_user_target.split("$continue")[1]
        except Exception as e:
            print(e)
            
        log["genie_user_target"] = genie_user_target
    
    if (dlgItem.reviews_query):
        log["reviews"] = dlgItem.genie_reviews
        log["reviews - Q"] = dlgItem.reviews_query
        log["reviews - A"] = dlgItem.genie_reviews_answer

    return {'agent_utterance': response, 'log_object': log}

@app.route("/user_rating", methods=["POST"])
def user_rating():
    """Front end required function that is not required by yelpbot
    """
    pass

@app.route("/user_preference", methods=["POST"])
def user_preference():
    """Front end required function that is not required by yelpbot
    """
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--greeting', type=str, default="Hi! How can I help you?", help="The first thing the agent says to the user")
    parser.add_argument('--engine', type=str, default='text-davinci-003', choices=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'gpt-35-turbo'],
                        help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
    parser.add_argument('--no_logging', action='store_true',
                        help='Do not output extra information about the intermediate steps.')
    parser.add_argument('--ssl_certificate_file', type=str, default='/etc/letsencrypt/live/backend.yelpbot.genie.stanford.edu/fullchain.pem',
                        help='Where to read the SSL certificate for HTTPS')
    parser.add_argument('--ssl_key_file', type=str, default='/etc/letsencrypt/live/backend.yelpbot.genie.stanford.edu/privkey.pem',
                        help='Where to read the SSL certificate for HTTPS')

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')

    context = (args.ssl_certificate_file, args.ssl_key_file)
    app.run(host="0.0.0.0", port=5001, use_reloader=False, ssl_context=context)