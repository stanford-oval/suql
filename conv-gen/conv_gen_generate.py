import os
from prompt_continuation import llm_generate
from pymongo import MongoClient
import re

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

def generate_actual_convs(seed_db, target_db, domain):
    for seed in seed_db.find({
        "domain": domain
    }):
        seed_conv = seed["conv"]
        continuation = llm_generate('', {}, engine="chatGPT", max_tokens=2000, temperature=0, stop_tokens=[], postprocess=False, filled_prompt=seed_conv)
        print(continuation)


if __name__ == "__main__":
    client = MongoClient("localhost", 27017)
    seed_conv = client["product_search"]["seed_convs"]
    target_conv = client["product_search"]["simulated_convs"]

    PRODUCT_DOMAINS = ["ac", "boardgame", "fridge", "laptop", "tv"]
    
    for i in PRODUCT_DOMAINS:
        # generate_seed_convs(seed_conv, i)
        generate_actual_convs(seed_conv, target_conv, i)