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
We also use MongoDB via Microsoft Azure to store dialog data. You can set up an instance at https://azure.microsoft.com/en-us/products/cosmos-db and obtain a "conncetion string" used to authenticate access to this database. You can easily use another MongoDB deployment for this as well, for example the free "shared" tier at https://www.mongodb.com/pricing.

Create a text file named `API_KEYS` in this folder and put your OpenAI API key (for access to GPT-3 models), Yelp API key and Cosmos connection string in it:

`export OPENAI_API_KEY=<your OpenAI API key>`

`export YELP_API_KEY=<your yelp.com API key>`

`export COSMOS_CONNECTION_STRING=<your Cosmos connection string>`

**Note that this file is in `.gitignore`. It is important that you never push your API keys to git.**

# Running the Code
You can read `Makefile` to see what the input arguments are.
1. Run `make genie-server`
This command will take some time to finish the first time you run it. It will download the necessary files and semantic parser model to ensure that the yelp database interface works properly.
<br>Tip: If you see Prediction worker had an error: spawn genienlp ENOENT, this means genienlp is not installed correctly.
If successful, the final message you see should be similar to this:
[I 230211 02:15:11 util:299] TransformerSeq2Seq has 139,420,416 parameters
Keep this command running.

1. If you want to run the agent locally, in a new terminal Run `make yelpbot`.
1. If you want to run the backend server, so that multiple users can send requests through a front-end, run `make start-backend`.

# Install front-end

Front-end website [here](https://github.com/stanford-oval/wikichat) (ovalchat , use branch `wip/restaurant-genie` for restaurant look). Follow instructions there to install it. To connect yelp-bot to this front-end, do:

1. do `make genie-server` in `genie-llm` .This brings up the Genie semantic parser. Keep this running, and;

2. run `make start-backendin` in `genie-llm` . This will bring up a connection for front-end to query at localhost:5001.

3. run `export NEXT_PUBLIC_CHAT_BACKEND=http://127.0.0.1:5001` followed by `yarn run dev` in ovalchat to bring up the front-end and test at localhost:3000
