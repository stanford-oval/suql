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
import threading
from pathlib import Path


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
        # make sure we write citation before value
        for key in chunk:
            assert("citation" in chunk[key])
            assert("value" in chunk[key])
            assert(len(chunk[key].items()) == 2)
            
            chunk[key] = {
                "citation": chunk[key]["citation"],
                "value": chunk[key]["value"],
            }
        
        res.append(chunk)
    
    
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
            
            if json1[key] is None or (not isinstance(json1[key], dict)):
                json1[key] = {
                    "value": None,
                    "citation": None
                }
            if json2[key] is None or (not isinstance(json2[key], dict)):
                json2[key] = {
                    "value": None,
                    "citation": None
                }
            
            if "value" not in json1[key]:
                json1[key]["value"] = None 
            if "value" not in json2[key]:
                json2[key]["value"] = None 
            
            type1 = type(json1[key]["value"])
            type2 = type(json2[key]["value"])
            
            if check_one_None_other_type(type1, type2, str):
                res[key] = {
                    "value": cvtNone2type(json1[key]["value"], str) + "; " + cvtNone2type(json2[key]["value"], str)
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

def handle_one(id, id_field_name, table_name, reviews, field_path_json, conn, log_file_path):
    final_json = {}
    
    for json_representation in produce_chunks(field_path_json):
        
        # determine if the input is too long:
        filled_prompt_len = num_tokens_from_string(_fill_template("prompts/json_stuffing.prompt", {"json_representation": json_representation, "documents": reviews}))
        if filled_prompt_len >= 2495:
            midpoint = len(reviews) // 2
            first_args = (id, id_field_name, table_name, reviews[:midpoint], field_path_json, conn, log_file_path)
            second_args = (id, id_field_name, table_name, reviews[midpoint:], field_path_json, conn, log_file_path)
            
            return combine_two_json(handle_one(*first_args), handle_one(*second_args))
        
        # try:
        first_try = llm_generate(
            "",
            "prompts/json_stuffing.prompt",
            {
                "json_representation": json.dumps(json_representation, indent=2),
                "documents": reviews
            },
            engine="gpt-35-turbo",
            max_tokens=1500,
            temperature=0,
            stop_tokens=None,
            all_system=True,
            log_file_path=log_file_path
        )
        # except Exception as e:
        #     print("Exception\n", e)
        #     continue


        try:
            res = json.loads(first_try)
        except JSONDecodeError:
            continuation = llm_generate(
                "",
                "prompts/correct_json.prompt",
                {
                    "json": first_try
                },
                engine="gpt-35-turbo",
                max_tokens=2500,
                temperature=0,
                stop_tokens=None,
                all_system=True,
                log_file_path=log_file_path
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
                            
        try:
            final_json.update(res)
        except Exception:
            pass
            
    convert_final_json(final_json)
    
    with conn.cursor() as cur:
        cur.execute(f"SELECT _schematization_results FROM {table_name} WHERE {id_field_name} = %s;", (id, ))
        current_schema_results = cur.fetchone()[0]

        if not current_schema_results:
            current_schema_results = []

        new_shema_results = [{"time": formatted_time, "results": final_json}] + current_schema_results
        
        cur.execute(f"UPDATE {table_name} SET _schematization_results = %s WHERE {id_field_name} = %s;", (json.dumps(new_shema_results), id))
        conn.commit()
    
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

def restaurants_free_text(entry):
    all_free_text = []
    if entry["reviews"] is not None and entry["reviews"] != []:
        all_free_text += entry["reviews"]
    
    return all_free_text

def safe_create_exists_schematization(conn, table_name):
    with conn.cursor() as cur:
        # execute the query
        cur.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='{table_name}' and column_name='_schematization_results';
        """)
        
        # fetch the result
        result = cur.fetchone()
        if not result:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN _schematization_results jsonb;")
            conn.commit()

def parallelize_handle_one(arguments, num_threads = 10):
    Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")).mkdir(parents=True, exist_ok=True)
    def handle_many(sub_arguments, thread_id):
        for i, value in enumerate(sub_arguments):
            print("Thread {}, at {}/{}".format(thread_id, i, len(sub_arguments)))
            handle_one(*value, os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs", "schematization_{}_thread_{}.log".format(formatted_time, thread_id)))
    
    def chunks(lst, n):
        """Yield n successive chunks from lst."""
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    threads = []
    
    for i, sub_arguments in enumerate(chunks(arguments, num_threads)):
        t = threading.Thread(target=handle_many, args=(sub_arguments, i, ))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

if __name__ == "__main__":
    # Establish a connection to the PostgreSQL database
    domains = [
        {
            "db_name": "restaurants",
            "table_name": "restaurants",
            "field_path_json": "/home/oval/genie-llm/schematization/schema/restaurants_manual_1.json",
            "free_text_fields_fcn": restaurants_free_text,
            "id_field_name": "id"
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
        
        safe_create_exists_schematization(conn, domain["table_name"])
        
        with conn.cursor() as cur:
            query = "SELECT * FROM {};".format(domain["table_name"])
            cur.execute(query)
            entries = cur.fetchall()
            column_names = [desc[0] for desc in cur.description]
            
            for entry in entries:
                entry_dict = dict((column_name, result) for column_name, result in zip(column_names, entry))
                free_text = domain["free_text_fields_fcn"](entry_dict)
                all_update_tuples.append((entry_dict[domain["id_field_name"]], domain["id_field_name"], domain["table_name"], free_text, domain["field_path_json"], conn))
        
    parallelize_handle_one(all_update_tuples)
    # for tuple in all_update_tuples[:3]:
    #     res = handle_one(*tuple)
    #     print(res)