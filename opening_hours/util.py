import openai, os, psycopg2,re,json,ast,csv
from sqlalchemy import create_engine
# this class is used for text generation
# it guarantees correct initialisation
# it keeps messages as logs
# generation can be customised 
RESTAURANTS = 15
class Chat:
    def __init__(self):
        self.initialise()
        self.logs = []
    def initialise(self):
        #print("Initialising Chat")
        openai.api_base = 'https://ovalopenairesource.openai.azure.com/'
        openai.api_type = 'azure'
        openai.api_version = '2023-05-15' # this may change in the future
        openai.api_key = os.getenv("OPENAI_API_KEY") # needs to be provided by 
    def send(self, msg):
        msg = [{"role":"user", "content":msg}]    
        response = openai.ChatCompletion.create(engine = "gpt-4", messages = msg, temperature = 0)
        text = response['choices'][0]['message']['content']
        self.logs.append(text)
        return text          
    def dump(self):
        for log in self.logs:
            print(log)
        return 0

def get_connection(dbname):
    return psycopg2.connect("dbname=" + dbname + " user=purrania")   
def run_SQL(SQL_command, name = "opening_hours"):
    conn = get_connection(name)
    cursor = conn.cursor()
    cursor.execute(SQL_command)
    conn.commit()
    conn.close()
def fetch_SQL(SQL_query, name = "opening_hours"):
    conn = get_connection(name)
    cursor = conn.cursor()
    cursor.execute(SQL_query)
    answer = cursor.fetchall()
    return answer   

def debug_load_hours(load_hours_text):
    msg = open("load_opening_hours.prompt", "r").read() 
    msg += "\n Follow the rules and turn the following into the required format:"
    msg += load_hours_text
    rsp = Chat().send(msg)
    #print(rsp)
    return rsp
     
def debug_parse_hours(opening_hours_text):
    msg = open("parse_opening_hours.prompt", "r").read()
    msg += opening_hours_text 
    rsp = Chat().send(msg)
    print(rsp)
    return rsp
def debug_add_indices():
    for i in range(1,RESTAURANTS+1):
        opening_hours_list = fetch_SQL("SELECT opening_hours FROM restaurants WHERE CAST(restaurants.id AS INTEGER) = "+ str(i)+";")
        opening_hours_text = ""
        if len(opening_hours_list):
            opening_hours_text = opening_hours_list[0][0]
            string_segments = debug_load_hours(opening_hours_text)
            print(string_segments)
            string_segments = "-".join(ast.literal_eval(string_segments))
            print(string_segments)
            if string_segments != None:
                command = "UPDATE restaurants SET opening_hours_index = '" + str(string_segments) + "' WHERE CAST(restaurants.id AS INTEGER) ="+ str(i)
                print(command)
                run_SQL(command)  
    
def debug_search_hours(prompt):
    print("prompt", prompt)
    parsed_prompt = debug_parse_hours(prompt)
    print("parsed_prompt", parsed_prompt)
    db_response = fetch_SQL(parsed_prompt)
    print("db_response", db_response)
    return db_response
def debug_load_dev_set(filename):
    with open ('devset.csv', 'r') as f:
        reader = csv.reader(f)
        columns = next(reader) 
        query = 'insert into restaurants({0}) values ({1})'
        query = query.format(','.join(columns), ','.join('?' * len(columns)))
        connection = get_connection("opening_hours")
        cursor = connection.cursor()
        for data in reader:
            cursor.execute(query, data)
        cursor.commit()
def debug():
    #debug_load_hours("Open from 8pm on Monday to 0200.")
    debug_add_indices()
    #debug_parse_hours("Which restaurants are open from 1am Monday to 9pm?")
    #debug_search_hours("Which restaurants are open between 10am and 2pm on Monday?")
    #debug_load_dev_set("devset.csv")
if __name__=="__main__":
    debug()
