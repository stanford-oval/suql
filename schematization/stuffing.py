import json
import os
from pymongo import MongoClient, ASCENDING
from prompt_continuation import llm_generate, _fill_template
import datetime
from json.decoder import JSONDecodeError
import tqdm
from tokenizer import num_tokens_from_string
import traceback
import psycopg2

now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M")

def chunk_dictionary(dictionary, chunk_size):
    keys = list(dictionary.keys())
    chunks = [keys[i:i+chunk_size] for i in range(0, len(keys), chunk_size)]
    result = []
    for chunk in chunks:
        chunk_dict = {key: dictionary[key] for key in chunk}
        result.append(chunk_dict)
    return result

def produce_chunks(field_json_path):

    with open(field_json_path, "r") as fd:
        my_dictionary = json.load(fd)
    chunk_size = 9

    chunks = chunk_dictionary(my_dictionary, chunk_size)
    res = []
    for chunk in chunks:
        res.append(json.dumps(chunk, indent=2))
    
    return res

def convert_final_json(raw_json : dict):
    for key in raw_json:
        if isinstance(raw_json[key], dict):
            for sub_key in raw_json[key]:
                if raw_json[key][sub_key] in ["None", "none", "Null", "null"]:
                    raw_json[key][sub_key] = None
                    
            # set value to be null if no citation is found
            if "value" in raw_json[key] and "citation" in raw_json[key]:
                if raw_json[key]["citation"] is None:
                    raw_json[key]["value"] = None

def check_one_None_other_type(a, b, desired):
    return (a == desired and b == desired) or (a == type(None) and b == desired) or (a == desired and b == type(None))
        
def cvtNone2type(x, desired):
    if x is None:
        if desired == str:
            return ""
        elif desired == list:
            return []
    return x


def combine_two_json(json1 : dict, json2 : dict):
    res = {}
    for key in json1:
        if key == "glued (special internal field)":
            continue
        
        if key in json2:
            
            if json1[key] is None:
                json1[key] = {
                    "value": None,
                    "citation": None
                }
            if json2[key] is None:
                json2[key] = {
                    "value": None,
                    "citation": None
                }
            
            type1 = type(json1[key]["value"])
            type2 = type(json2[key]["value"])
            
            if check_one_None_other_type(type1, type2, str):
                res[key] = {
                    "value": cvtNone2type(json1[key]["value"], str) + ";" + cvtNone2type(json2[key]["value"], str)
                }
            elif check_one_None_other_type(type1, type2, list):
                res[key] = {
                    "value": cvtNone2type(json1[key]["value"], list) + cvtNone2type(json2[key]["value"], list)
                }
            elif check_one_None_other_type(type1, type2, bool):
                res[key] = {
                    "value": bool(json1[key]["value"]) or bool(json2[key]["value"])
                }
            else:
                res[key] = {
                    "value": json1[key]["value"],
                }
            try:                
                res[key]["citation"] = json1[key]["citation"] + json2[key]["citation"]
            except Exception:
                if "citation" in json1[key] and json1[key]["citation"] is not None:
                    res[key]["citation"] = json1[key]["citation"]
                elif "citation" in json2[key] and json2[key]["citation"] is not None:
                    res[key]["citation"] = json2[key]["citation"]
                else:
                    res[key]["citation"] = None
    
    res["glued (special internal field)"] = True
    return res
                
        

def handle_one(reviews, field_path_json):
    final_json = {}
    
    for json_representation in produce_chunks(field_path_json):
        
        # print("Hi")
        # determine if the input is too long:
        # filled_prompt_len = num_tokens_from_string(_fill_template("prompts/json_stuffing.prompt", {"json_representation": json_representation, "reviews": reviews}))
        # if filled_prompt_len >= 15370:
        #     midpoint = len(reviews) // 2
        #     return combine_two_json(handle_one(reviews[:midpoint]), handle_one(reviews[midpoint:]))
        
        # try:
        first_try = llm_generate(
            # "prompts/json_stuffing_system.prompt",
            "",
            "prompts/json_stuffing.prompt",
            {
                "json_representation": json_representation,
                "documents": reviews
            },
            engine="gpt-35-turbo-16k",
            max_tokens=1000,
            temperature=0,
            stop_tokens=None,
            all_user=True
        )
        print("first_try" + first_try)
        # except Exception as e:
        #     print("Exception\n", e)
        #     continue


        try:
            res = json.loads(first_try)
            final_json.update(res)
        except JSONDecodeError:
            continuation = llm_generate(
                "prompts/correct_json.prompt",
                {
                    "json": first_try
                },
                engine="chatGPT",
                max_tokens=2500,
                temperature=0,
                stop_tokens=None,
                postprocess=False
            )
            
            try:
                res = json.loads(continuation)
            except JSONDecodeError:
                with open("failed_json.txt", "a") as fd:
                    fd.write("===\n")
                    fd.write("first json:\n")
                    fd.write(first_try)
                    fd.write("\n---\n")
                    fd.write("self-corrected json:\n")
                    fd.write(continuation)
                continue
                            
            final_json.update(res)
            
    convert_final_json(final_json)
    return final_json

# this is a function that returns the free text entries (as a list)
# for a given database entry
# this needs to be customized for each database since each database
# has a different way of storing free text fields
def products_free_text(entry):
    all_free_text = []
    if entry["about"] is not None and entry["about"] != "":
        all_free_text.append(entry["about"])
    if entry["description"] is not None and entry["description"] != "":
        all_free_text.append(entry["description"])
    
    for review in entry["reviews"]:
        if "review" != "":
            all_free_text.append(review)
    return all_free_text


if __name__ == "__main__":
    # Establish a connection to the PostgreSQL database
    domains = [
        {
            "db_name": "ac",
            "table_name": "ac.products",
            "field_path_json": "/home/oval/genie-llm/conv-gen/gen_results/ac.json",
            "free_text_fields_fcn": products_free_text,
            "id_field_name": "asin"
        }
    ]
    
    all_update_tuples = []
    
    for domain in domains:
        conn = psycopg2.connect(
            database=domain["db_name"],
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432"
        )
        cursor = conn.cursor()
        query = "SELECT * FROM {};".format(domain["table_name"])
        cursor.execute(query)
        entries = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        for entry in entries:
            entry_dict = dict((column_name, result) for column_name, result in zip(column_names, entry))
            free_text = domain["free_text_fields_fcn"](entry_dict)
            all_update_tuples.append((free_text, domain["field_path_json"]))
        
    # print(all_update_tuples[0])
    res = handle_one(*(all_update_tuples[0]))
    print(res)
        
        # try:

                # res = handle_one(free_text, domain["field_path_json"])
                # current_schema_results = i["schema_results"] if "schema_results" in i else []
                # updated_res = collection.update_one({
                #     "id" : i["id"]
                # }, 
                # {
                #     "$set": {
                #         "schema_results": [{"time": formatted_time, "results": res}] + current_schema_results
                #     }
                # })
                
                
                # updated_count += updated_res.modified_count
                # print("modified {}, total modified {}".format(updated_res.modified_count, updated_count))
                
            # entries.close()
        # except Exception as e:
        #     print(e)
        #     print(traceback.format_exc())
        #     entries.close()