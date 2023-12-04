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
import logging

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Api, reqparse

from yelp_loop import *

from pymongo import MongoClient, ASCENDING
from datetime import datetime

# set up the MongoDB connection
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")


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

class BackendConnection:
    def __init__(
        self,
        greeting = "Hi! How can I help you?",
        engine = "text-davinci-003") -> None:
        
        client = MongoClient(CONNECTION_STRING)
        self.db = client['yelpbot']  # the database name is yelpbot
        self.table = self.db['dialog_turns_studies'] # the collection that stores dialog turns
        self.table.create_index("$**") # necessary to build an index before we can call sort()

        self.greeting = greeting
        self.engine = engine
        
    def compute_next(self, dialog_id, user_utterance, turn_id, system_name, experiment_id) -> DialogueTurn:
        tuples = list(self.table.find( { "dialogID": dialog_id } ).sort('turn_id', ASCENDING))
        
        # initialize empty dialog in the first turn
        if (not tuples):
            dlgHistory = []
        # otherwise we retrieve the dialog history
        else:
            dlgHistory = BackendConnection._reconstruct_dlgHistory(tuples)

        dlgHistory = compute_next_turn(
            dlgHistory,
            user_utterance,
            engine=self.engine,
            sys_type=system_name
        )
        
        # update the current tuple with new DialogTurn
        new_tuple = {"_id": '(' + str(dialog_id) + ', '+ str(turn_id) + ')', "dialogID": dialog_id, "turn_id": turn_id, "created_at": datetime.utcnow(), "dlg_turn" : dlgHistory[-1].__dict__, "experiment_id": experiment_id}
        
        # insert the new dialog turn into DB
        self.table.insert_one(new_tuple)
                
        return dlgHistory[-1]

    @staticmethod
    def _reconstruct_dlgHistory(tuples):
        """Given a list of *sorted* mongodb dialog tuples, reconstruct a list of DialogTurns
        Args:
            tuples : a list of mongodb tuples, sorted in the order of first turn to last turn
        """
        return list(map(lambda x: DialogueTurn(**x["dlg_turn"]), tuples))

    @staticmethod
    def _reconstruct_genieinfo(tuples):
        """Given a list of *sorted* mongodb dialog tuples, find the latest Genie information
        Args:
            tuples :a list of mongodb tuples, sorted in the order of first turn to last turn
        """
        for i in reversed(tuples):
            if "genieDS" in i:
                return i["genieDS"], i["genieAux"]
        return "null", []

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
    system_name = request_args['system_name']
    experiment_id = request_args['experiment_id']
    
    # special case processing when user finishes testing
    if user_utterance == 'FINISHED':
        return {'agent_utterance': f"{dialog_id} is your dialog ID. Please submit it in the Google Form.", 'log_object': {}}
    
    dlgItem = connection.compute_next(dialog_id, user_utterance, turn_id, system_name, experiment_id)

    log = {}
    log["1st_sql"] = dlgItem.user_target
    log["2nd_sql"] = dlgItem.temp_target
    log["db_results"] = json.loads(dlgItem.genie_utterance) if dlgItem.genie_utterance is not None else None

    def pp_time(time_statement):
        return [
            "First classifier: {:.2f}s".format(time_statement["first_classification"]), 
            "Semantic parser: {:.2f}s".format(time_statement["semantic_parser"]),
            "SUQL execution: {:.2f}s".format(time_statement["suql_execution"]),
            "Final response: {:.2f}s".format(time_statement["final_response"])
        ]

    log["Elapsed Time"] = pp_time(dlgItem.time_statement)

    return {'agent_utterance': dlgItem.agent_utterance, 'log_object': log}

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
    parser.add_argument('--engine', type=str, default='text-davinci-003', choices=['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'gpt-3.5-turbo-0613'],
                        help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
    parser.add_argument('--no_logging', action='store_true',
                        help='Do not output extra information about the intermediate steps.')
    parser.add_argument('--ssl_certificate_file', type=str, help='Where to read the SSL certificate for HTTPS')
    parser.add_argument('--ssl_key_file', type=str, help='Where to read the SSL certificate for HTTPS')

    args = parser.parse_args()

    if args.no_logging:
        logging.basicConfig(level=logging.CRITICAL, format=' %(name)s : %(levelname)-8s : %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')

    context = (args.ssl_certificate_file, args.ssl_key_file)
    app.run(host="0.0.0.0", port=5001, use_reloader=False, ssl_context=context)