# Installing dependencies
## GPT-3 Prompts
In order to use the core functionality, i.e. using GPT-3 prompts, you only need to follow the following steps:
We recommend using a virtual environment like `virtualenv`, `pipenv`, or `conda` to keep the python requirements for different project separate. For example if you are using conda, run `conda create -n genie_llm python=3.8`. For pyGenieScript to work, use python 3.7 or 3.8.
Install the python packages listed in `requirements.txt`. For example with the command `pip install -r requirements.txt`

## Connecting to a Semantic Parser w/ pyGenieScript
If your application requires Genie (for using a semantic parser to connect with a database), you should use `pyGenieScript`. Refer to [pyGenieScript installtion](https://github.com/stanford-oval/pyGenieScript#installation) for steps to install that.

# Setting up the API Keys
OpenAI's GPT-3 (paid service) and yelp.com's API (has a free tier) both require authentication.
You can find your GPT-3 API key by visiting [here](https://platform.openai.com/account/api-keys). Yelp's API token can be retrieved by registering an account [here](https://fusion.yelp.com/).

Create a text file named `API_KEYS` in this folder and put your OpenAI API key (for access to GPT-3 models) and Perspective API key (for access to toxicity detection models) and Yelp API key in it:

`export OPENAI_API_KEY=<your OpenAI API key>`

`export YELP_API_KEY=<your yelp.com API key>`

**Note that this file is in `.gitignore`. It is important that you never push your API keys to git.**

# Running the Code
You can read `Makefile` to see what the input arguments are.
1. Run `make genie-server`
This command will take some time to finish the first time you run it. It will download the necessary files and semantic parser model to ensure that the yelp database interface works properly.
<br>Tip: If you see Prediction worker had an error: spawn genienlp ENOENT, this means genienlp is not installed correctly.
If successful, the final message you see should be similar to this:
[I 230211 02:15:11 util:299] TransformerSeq2Seq has 139,420,416 parameters
1. Keep the other command running, and in a new terminal Run `make yelpbot`.
