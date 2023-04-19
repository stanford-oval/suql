import pymongo
import os
import json

# run export COSMOS_CONNECTION_STRING="..." in terminal
client = pymongo.MongoClient(os.environ.get("COSMOS_CONNECTION_STRING"))
db = client["yelpbot"]['dialog_turns']

def get_all_dialog_turns(dialog_id):
    print(f"===== {dialog_id}")
    for x in db.find({'dialogID': {'$regex': dialog_id}}):
        print(json.dumps(x['dlg_turn'], indent=2))
        if "genie_user_target" in x:
            print(x['genie_user_target'])

SEARCH_TERM = "what about an expensive japanese one?"
for x in db.find({'dlg_turn.user_utterance': {'$regex': SEARCH_TERM}}):
    get_all_dialog_turns(x['dialogID'])