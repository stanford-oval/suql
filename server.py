from flask import request, Flask
from prompt_continuation import llm_generate
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format=' %(name)s : %(levelname)-8s : %(message)s')


class SemanticParser():
    def __init__(self):
        self.dlg_turns = []
        
    def parse(self, data):
        # query = data.get('q')
        continuation = llm_generate(template_file='prompts/parser.prompt',
                        engine='text-davinci-003',
                        stop_tokens=None,
                        max_tokens=100,
                        temperature=0,
                        prompt_parameter_values={'dlg': self.dlg_turns},
                        postprocess=False)
        
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
    

# Set the server address
host = "127.0.0.1"
port = 8400

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