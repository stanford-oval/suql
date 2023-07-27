# query = f"""
# SELECT table_schema, table_name, column_name, data_type
# FROM information_schema.columns
# WHERE table_name = '{table_name}'
#   AND (data_type = 'text' OR data_type = 'ARRAY') AND (NOT column_name ILIKE '%_citation');
# """

from sentence_transformers import SentenceTransformer, util
import pymongo

model = SentenceTransformer('all-mpnet-base-v2')
SIMILARITY_THRESHOLD = 0.5
# client = pymongo.MongoClient()
# db = client['yelpbot']
# collection = db['yelp_data']
# schematized = db['schematized']
# cache_db = client['free_text_cache']['all-mpnet-base-v2-cache']


def in_any_no_cache(keyword, keyword_list):
    # first check if keyword is directly inside keyword_list
    if keyword_list == [] or keyword_list is None:
        return False
    if keyword in keyword_list:
        return True
    
    
    embedding_keyword = model.encode(keyword, convert_to_tensor=True)
    embedding_keyword_list = model.encode(keyword_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding_keyword, embedding_keyword_list).tolist()[0]
    res = [(number, keyword) for number, keyword in zip(cosine_scores, keyword_list) if number > SIMILARITY_THRESHOLD]
    
    if res != []:
        print(res)
        return True
    return False
    # print(embedding_keyword.tolist())
    # print(embedding_keyword_list.tolist())
    
def equal_no_cache(comp_value, field_value):
    if comp_value == field_value:
        return True

    embedding_comp = model.encode(comp_value, convert_to_tensor=True)
    embedding_field = model.encode(field_value, convert_to_tensor=True)
    cosine_score = util.cos_sim(embedding_comp, embedding_field).tolist()[0][0]
    # print(cosine_score)
    return cosine_score > SIMILARITY_THRESHOLD

if __name__ == "__main__":
    print(in_any_no_cache("pasta", ["pappardelle", "spaghetti"]))
    print(equal_no_cache("pasta", "meatball spaghetti"))
    print(equal_no_cache("pasta", "pappardelle"))
    print(equal_no_cache("vegan", "vegetarian"))