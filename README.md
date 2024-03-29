<p align="center">
    <b>SUQL (Structured and Unstructured Query Language)</b>
    <br>
    <a href="https://arxiv.org/abs/2311.09818">
        <img src="https://img.shields.io/badge/cs.CL-2311.09818-b31b1b" alt="arXiv">
    </a>
    <a href="https://github.com/stanford-oval/suql/stargazers">
        <img src="https://img.shields.io/github/stars/stanford-oval/suql?style=social" alt="Github Stars">
    </a>
</p>
<p align="center">
    Conversational Search over Structured and Unstructured Data with LLMs
</p>
<p align="center">
    Online demo:
    <a href="https://yelpbot.genie.stanford.edu" target="_blank">
        https://yelpbot.genie.stanford.edu
    </a>
    <br>
</p>


# What is SUQL

SUQL stands for Structured and Unstructured Query Language. It augments SQL with several important free text primitives for a precise, succinct, and expressive representation. It can be used to build chatbots for relational data sources that contain both structured and unstructured information. Similar to how text-to-SQL has seen [great success](https://python.langchain.com/docs/use_cases/qa_structured/sql), SUQL can be uses as the semantic parsing target language for hybrid databases, for instance, for:

![An example restaurant relational database](figures/figure1.png)

Several important features:

- SUQL seamlessly integrates retrieval models, LLMs, and traditional SQL to deliver a clean, effective interface for hybrid data access;
    - It utilizes techniques inherent to each component: retrieval model and LM for unstructured data and relational SQL for structured data;
- Index of free text fields built with [faiss](https://github.com/facebookresearch/faiss), natively supporting all your favorite dense vector processing methods, e.g. product quantizer, HNSW, etc.;
- A series of important optimizations to minimize expensive LLM calls;
- Scalability to large databases with PostgreSQL;
- Support for general SQLs, e.g. JOINs, GROUP BYs.

## The answer function

One important component of SUQL is the `answer` function. `answer` function allows for constraints from free text to be easily combined with structured constraints. Here is one high-level example:

![An example for using SUQL](figures/figure2.png)

For more details, see our paper at https://arxiv.org/abs/2311.09818.

# Installation / Usage tutorial

## Installing [PostgreSQL database](https://www.postgresql.org/)

1. Follow the instruction there to install a postgreSQL database. For instance, if you are installing on Ubuntu, then follow section `PostgreSQL Apt Repository` at https://www.postgresql.org/download/linux/ubuntu/.

2. After that, the SUQL compiler needs to make use of python functions within postgreSQL. This is done via the `postgresql-plpython3` language. If you are using Ubuntu, simply run `sudo apt-get install postgresql-plpython3-15`.

3. Then, in your database's command line (incurred via `psql your_database_name`), do `CREATE EXTENSION plpython3u;`. This loads this language into the current db.

## Installing python dependencies of SUQL

1. Install dependencies in `requirements.txt`;

2. Run `python -m spacy download en_core_web_sm`;

### Entry point to the SUQL compiler

The entry point to the SUQL compiler is the following function from `sql_free_text_support/execute_free_text_sql.py`:

```
suql_execute()
```

## How to set up SUQL on your PostgreSQL database step-by-step.

Here is a rough breakdown of what you need to do to set up SUQL on your domain:

1. Set up OpenAI API key with `export OPENAI_API_KEY=[your OpenAI API key here]`

2. Write a semantic parser prompt, and substitute `prompts/parser_sql.prompt` with it. Include examples of how to use the `answer` function.

3. Set up an embedding server for the SUQL compiler to query. Go to `sql_free_text_support/faiss_embedding.py`, and modify the lines of the form `embedding_store.add(table_name="restaurants", primary_key_field_name="_id", free_text_field_name="popular_dishes", db_name="restaurants")` to match your database with its column names. For instance, this line specifies the SUQL compiler to set up an embedding server for the `restaurants` database, which has `_id` column as the unique row identifier, for the `popular_dishes` and `reviews` columns (these columns need to be of type `TEXT` or `TEXT[]`) under table `restaurants`. By default, this will be set up on port 8509, which is then called by `execute_free_text_sql`. In case you need to use another port, please change both addresses.

4. In the command line for your database, copy and paste all content under `custom_functions.sql`. This will define the `answer` and `summary` functions under your PostgreSQL database.

5. Set up the backend server for the `answer`, `summary` functions. As you probably noticed, the code in `custom_functions.sql` is just making queries to a server. This server can be instantiated by running `python free_text_fcns_server.py.py`.

6. There is a classifier to determine whether a user utterance requires database access, at this line: `llm_generate(template_file='prompts/if_db_classification.prompt', ...)`. This may or may not be applicable to your domain, and if it is, please modify the [corresponding prompt](https://github.com/stanford-oval/suql/blob/main/prompts/if_db_classification.prompt).

7. A note on PostgreSQL's permission issue. The user-facing parts of this system would only require **SELECT** privilege (it would be safe to grant only SELECT privilege for GPT-generated queries). This user is named `select_user` with password `select_user` in [this file](https://github.com/stanford-oval/suql/blob/main/postgresql_connection.py). You should change the default values for `user`, `password`, and `database` there to match your PostgreSQL set up. This user also appears once in [this file](https://github.com/stanford-oval/suql/blob/main/sql_free_text_support/execute_free_text_sql.py) of the SUQL compiler, so that intermediate tables created can be queried. Please also change it to match your user name.

8. Furthermore, various parts of the SUQL compiler require the permission to **create** a temporary table. You can search for `creator_role` under [this file](https://github.com/stanford-oval/suql/blob/main/sql_free_text_support/execute_free_text_sql.py). You would need to create a user with the privilege to **CREATE** tables under your database, and change `creator_role` to match that user and password. 

9. Test with `python yelp_loop.py`.

# Known issues

1. if you see error msgs similar to `PermissionError: [Errno 13] Permission denied: '/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4.2749b823-646b-45d7-9fcf-11414469d900.tmp'`. Refer to https://github.com/openai/tiktoken/issues/75. A likely solution is setting `TIKTOKEN_CACHE_DIR=""`.

2. A lot of times, Azure/OpenAI's chatGPT deployment's latency is unstable. We have experienced up to 10 minutes of latency for some inputs. These cases are rare (we estimate < 3% of cases), but they do happen from time to time. For those cases, if we cancel the request and re-issue them, then we typically can get a response in normal time. To counter this issue, we have implemented a max-wait-then-reissue functionality in our API calls. Under [this file](https://github.com/stanford-oval/genie-llm/blob/main/prompt_continuation.py), we have the following block:

```
if max_wait_time is None:
    max_wait_time = 0.005 * total_token + 1
```

This says that if a call to `llm_generate` does not set a `max_wait_time`, then it is dynamically calculated based on this linear function of `total_token`. This is imperfect, and we are erroring on the side of waiting longer (e.g., for an input with `1000` tokens, this would wait for 6 seconds, which might be too long). You can set a custom wait time, or disable this feature or together by setting `attempts = 0`.

3. The SUQL compiler right now uses the special character `^` when handling certain join-related optimizations. Please do not include this character `^` in your column names. (This restriction could be lifted in the future.)