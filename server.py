from flask import request, Flask
from prompt_continuation import llm_generate
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')


# Set the server address
host = "127.0.0.1"
port = 8501  # change back to 8401 
GPT_parser_address = 'http://{}:{}'.format(host, port)

def process_user_target(user_target: str):
    try:
        user_target = user_target.split("$continue")[1]
    except Exception as e:
        print("error while partitioning {}".format(user_target))
        print(e)
    return user_target

class SemanticParser():
    def __init__(self):
        self.dlg_turns = []
        
    def parse(self, data):
        query = data.get('q')
        continuation = llm_generate(template_file='prompts/parser.prompt',
                        engine='gpt-35-turbo',
                        stop_tokens=["Agent:"],
                        max_tokens=100,
                        temperature=0,
                        prompt_parameter_values={'dlg': self.dlg_turns, 'process_user_target': process_user_target, 'query': query},
                        postprocess=False)
        
        continuation = continuation.rstrip("Agent:")
        # put the result in a list since this is what genie accepts as of now
        thingtalk_res = ['$dialogue @org.thingpedia.dialogue.transaction.execute; $continue ' + continuation]
        print(thingtalk_res)
        
        
        result = {
            'candidates': [
                {
                    "code": thingtalk_res,
                    "score": 1
                }
            ],
            'entities': {},
            'intent': {
                'command': 1,
                'ignore': 0,
                'other': 0
            },
            'tokens': ['show', 'me']
        }
        
        return result
    
    def set_dlg_turns(self, data):
        self.dlg_turns = data["dlg_turn"]
        return self.dlg_turns


if __name__ == '__main__':
    s = SemanticParser()

    @app.route('/en-US/query', methods=['POST'])
    def query():
        data = request.get_json()
        return s.parse(data)

    @app.route('/set_dlg_turn', methods=['POST'])
    def set_dlgs_turns():
        data = request.get_json()
        return s.set_dlg_turns(data)

    app.run(host=host, port=port)