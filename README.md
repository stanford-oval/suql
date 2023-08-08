# Installation

Note: this installation assumes there already exists a postgres database on the VM, all necessary python functions have been declared in the postgres database, and there is a server running (via `python reviews_server.py`).

1. Install dependencies in `requirements.txt`;

2. Run `python -m spacy download en_core_web_sm`;

3. Export your OpenAI key with `export OPENAI_API_KEY=[your OpenAI key]`;

4. Test with `python yelp_loop.py`.