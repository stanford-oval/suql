from pymongo import MongoClient
import re
from collections import defaultdict
import os
import json


def get_value_for_field(field_name : str, sql : str, end : int):
    # given a field name (including the special symbol @), the sql, and the end of the field name
    rhs = sql[end+1:]
    regex_pattern = [
        {
            "regex": "^(=|<=|>=|<|>|!=|<>|LIKE|ILIKE|SIMILAR TO|~|~\*|!~|!~*) ('[a-zA-Z_0-9]+')",
            "type": "TEXT"
        },
        {
            "regex": "^(=|<=|>=|<|>|!=|<>) ([0-9]+)",
            "type": "NUMBER"
        },
        {
            "regex": "^(=) (TRUE|FALSE)",
            "type": "BOOLEAN"
        },
    ]
    
    
    if rhs.startswith("= '"):
        


def get_all_new_fields_by_domain(db, domain, select_top=None, outfile=None, outfile_json=None):
    all_matches = defaultdict(int)
    all_matches_values = defaultdict(list)
    for i in db.find({
        "domain": domain
    }):
        if "parse_results_at" in i:
            for turn in i["parse_results_at"]:
                parse = turn["parse"]
                pattern = r'@([a-zA-Z0-9_]+)'
                matches = re.finditer(pattern, parse)
                for match in matches:
                    all_matches[match.group(1)] += 1
    sorted_list = sorted(all_matches.items(), key=lambda item: item[1], reverse=True)
    if select_top is not None:
        sorted_list = sorted_list[:select_top]
    sorted_dict = dict(sorted_list)
    
    if outfile is not None:
        import csv
        with open(outfile, 'w', newline='') as file:
            writer = csv.DictWriter(file, delimiter='\t', fieldnames=["column_name", "freq"])

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for key, value in sorted_dict.items():
                writer.writerow({'column_name': key, 'freq': value})
    
    if outfile_json is not None:
        json_entries = {}
        for key, value in sorted_dict.items():
            json_entries.update({
                key: {
                    "citation": "list of free text",
                    "value": "free text"
                }
            })
        with open(outfile_json, 'w', newline='') as file:
            json.dump(json_entries, file, indent=2)
        
    return sorted_dict

if __name__ == "__main__":
    OUTPUT_DIR = "/home/oval/genie-llm/conv-gen/gen_results"
    PRODUCT_DOMAINS = ["ac", "boardgame", "fridge", "laptop", "tv"]
    
    client = MongoClient("localhost", 27017)
    conv_db = client["product_search"]["simulated_convs"]
    for domain in PRODUCT_DOMAINS:
        get_all_new_fields_by_domain(conv_db, domain, outfile=os.path.join(OUTPUT_DIR, domain + ".tsv"), outfile_json=os.path.join(OUTPUT_DIR, domain + ".json"))