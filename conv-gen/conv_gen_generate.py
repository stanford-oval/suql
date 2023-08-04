import os
from prompt_continuation import llm_generate
from pymongo import MongoClient
import re
from tqdm import tqdm
import multiprocessing

class DialogTurn:
    turn : str
    target : str
    
    def __init__(self, turn, target) -> None:
        self.turn = turn
        self.target = target

def get_conv_gen_prompt(product):
    return "prompts/seed_convs_{}.prompt".format(product)

def parse_output(continuation):
    chunks = re.split(r'\d+\.', continuation.strip())
    # chunks = re.split(r'---', continuation.strip())
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks

def generate_conv_gen(product):
    continuation = llm_generate(get_conv_gen_prompt(product), {}, engine="chatGPT", max_tokens=2000, temperature=0, stop_tokens=[], postprocess=False)
    return parse_output(continuation)

def generate_seed_convs(db, domain):
    for item in generate_conv_gen(domain):
        db.insert_one({
            "conv": item,
            "domain": domain
        })

def generate_actual_convs(seed_db, target_db, domain, dup=10):
    for i in range(dup):
        seeds = seed_db.find({
            "domain": domain
        })
        print("beginning iteration {} out of {}".format(i+1, dup))
        for seed in tqdm(list(seeds)):
            continuation = llm_generate('', {}, engine="chatGPT", max_tokens=2000, temperature=0, stop_tokens=[], postprocess=False, filled_prompt=seed["conv"])
            target_db.insert_one({
                "domain": domain,
                "seed_id": seed["_id"],
                "conv": continuation
            })

def finalize_convs(seed_conv, conv_conv, start, end):
    client = MongoClient("localhost", 27017)
    seed_db = client["product_search"][seed_conv]
    conv_db = client["product_search"][conv_conv]
    generated_convs = conv_db.find()
    current_j = -1
    for conv in list(generated_convs):
        current_j += 1        
        if current_j < start:
            continue
        if current_j >= end:
            break
        print("job ({}, {}), currently at {}".format(start, end, current_j))
        seed = seed_db.find_one({
            "_id": conv["seed_id"]
        })
        assert(seed)
        beginning = seed["conv"]
        # print(beginning)
        assert(beginning.split("\n")[0].endswith("Here is a conversation they are having for 20 turns."))        
        first_turn = beginning.split("\n")[1:][0]
        # print(first_turn[0])
        
        conv_seperated = [first_turn] + conv["conv"].replace("\n\n", "\n").split("\n")

        # print("length of simulated conv {}: {}".format(conv["_id"], len(conv_seperated)))
        agent = conv_seperated[0].split(":")[0]
        user = conv_seperated[1].split(":")[0]
        
        chunks = []
        last_i = 0
        for i, turn in enumerate(conv_seperated):
            if turn.startswith(user):
                chunks.append(DialogTurn('\n'.join(conv_seperated[last_i:i+1]), None))
                last_i = i + 1
        # print("length of simulated conv by user turns {}: {}".format(conv["_id"], len(chunks)))
        
        parse_results = []
        for i, chunk in enumerate(chunks):
            parameter_dictionaries = {
                "turns": chunks[:i + 1],
                "user": user,
                "agent": agent
            }
            continuation =  llm_generate('prompts/parser_SQL_field_gen.prompt', parameter_dictionaries, engine="chatGPT", max_tokens=200, temperature=0, stop_tokens=[";", "\n"], postprocess=False)
            chunks[i].target = continuation
            
            parse_results.append({
                    "turn_id": i,
                    "turns": [chunk.turn for chunk in chunks[:i + 1]],
                    "parse": continuation
                })
        conv_db.update_one({"_id": conv["_id"]}, {
            "$set": {
                "parse_results_at": parse_results
            }
        })        
        # print("====")

def parallelize(seed_conv, target_conv, start, end, step):
    tuples = []
    for i in range(start, end+1, step):
        t = (seed_conv, target_conv, i, i+step-1)
        tuples.append(t)
    print(tuples)
    pool = multiprocessing.Pool(processes=len(tuples))
    pool.starmap(finalize_convs, tuples)
    pool.close()
    pool.join()

if __name__ == "__main__":
    PRODUCT_DOMAINS = ["ac", "boardgame", "fridge", "laptop", "tv"]
    client = MongoClient("localhost", 27017)
    seed_db = client["product_search"]["seed_convs"]
    conv_db = client["product_search"]["simulated_convs"]

    for i in PRODUCT_DOMAINS:
        # first generate the seed conversations
        generate_seed_convs(seed_db, i)
        # then generate the actual conversations
        generate_actual_convs(seed_db, conv_db, i)
    
    # generate the SQL parse for all the conversations
    finalize_convs("seed_convs", "simulated_convs", 0, 500)
    # this is a parallel call to the above function
    # parallelize("seed_convs", "simulated_convs", 0, 500, 50)