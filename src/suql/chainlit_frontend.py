import chainlit as cl
from chainlit.types import ThreadDict
from suql.agent import *
from pymongo import MongoClient, ASCENDING
import argparse
from typing import Dict, Optional

# NOTE: this step logs conversations to a mongodb via
# env variable `COSMOS_CONNECTION_STRING`.
# You can log to a MongoDB of your choice here
CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")

class BackendConnection:
    def __init__(
        self,
        greeting = "Hi! How can I help you?") -> None:
        
        client = MongoClient(CONNECTION_STRING)
        self.db = client['yelpbot']  # the database name is yelpbot
        self.table = self.db['dialog_turns_deploy'] # the collection that stores dialog turns
        self.table.create_index("$**") # necessary to build an index before we can call sort()

        self.greeting = greeting
        
    async def compute_next(self, dialog_id, user_utterance, turn_id, system_name, experiment_id) -> DialogueTurn:
        tuples = list(self.table.find( { "dialogID": dialog_id } ).sort('turn_id', ASCENDING))
        
        # initialize empty dialog in the first turn
        if (not tuples):
            dlgHistory = []
        # otherwise we retrieve the dialog history
        else:
            dlgHistory = BackendConnection._reconstruct_dlgHistory(tuples)

        dlgHistory = await compute_next_turn(
            dlgHistory,
            user_utterance
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

connection = BackendConnection()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("turn_id", 0)


@cl.on_message
async def on_message(message: cl.Message):
    user_utterance = message.content
    dialog_id = cl.user_session.get("id")
    turn_id = cl.user_session.get("turn_id")
    cl.user_session.set("turn_id", turn_id + 1)
    system_name = "chainlit_testing"
    experiment_id = "chainlit_testing"
    
    dlgItem = await connection.compute_next(dialog_id, user_utterance, turn_id, system_name, experiment_id)

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


# Code below would enable a login page
# @cl.oauth_callback
# def oauth_callback(
#   provider_id: str,
#   token: str,
#   raw_user_data: Dict[str, str],
#   default_user: cl.User,
# ) -> Optional[cl.User]:
#   return default_user

# @cl.password_auth_callback
# def auth_callback(username: str, password: str) -> Optional[cl.User]:
#   # Fetch the user matching username from your database
#   # and compare the hashed password with the value stored in the database
#   if (username, password) == ("admin", "admin"):
#     return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
#   else:
#     return None
