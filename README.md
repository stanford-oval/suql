# Installation

Note: this installation assumes there already exists a postgres database on the VM, all necessary python functions have been declared in the postgres database, and there is a server running (via `python reviews_server.py`).

1. Install dependencies in `requirements.txt`;

2. Run `python -m spacy download en_core_web_sm`;

3. Export your OpenAI key with `export OPENAI_API_KEY=[your OpenAI key]`;

4. Test with `python yelp_loop.py`.

# Known issues

1. if you encounter error msgs similar to `PermissionError: [Errno 13] Permission denied: '/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4.2749b823-646b-45d7-9fcf-11414469d900.tmp'`. Refer to https://github.com/openai/tiktoken/issues/75. A likely solution is setting `TIKTOKEN_CACHE_DIR=""`.