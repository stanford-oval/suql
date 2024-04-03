# Overview

You should use this installation guide if you are planning on setting up a SUQL-powered agent. Downloading from source makes it easy to modify few-shot prompts and quickly set up a working agent. If you are planning on using SUQL in a larger codebase and only need an entry point to the SUQL compiler, we recommend using [install_pip.md](install_pip.md) instead.

To get started, run:
```
git clone https://github.com/stanford-oval/suql.git
```

# Installing [PostgreSQL database](https://www.postgresql.org/)

1. Follow the instruction there to install a postgreSQL database. For instance, if you are installing on Ubuntu, then follow section `PostgreSQL Apt Repository` at https://www.postgresql.org/download/linux/ubuntu/.

2. After that, the SUQL compiler needs to make use of python functions within postgreSQL. This is done via the `postgresql-plpython3` language. If you are using Ubuntu, simply run `sudo apt-get install postgresql-plpython3-15`.

3. Then, in your database's command line (incurred via `psql your_database_name`), do `CREATE EXTENSION plpython3u;`. This loads this language into the current db.

# Installing python dependencies of SUQL

1. Install dependencies via `conda env create -f conda_env.yml`;

2. Run `python -m spacy download en_core_web_sm`;

# How to set up SUQL on your PostgreSQL database step-by-step.

Here is a breakdown of what you need to do to set up SUQL on your domain:

## Set up PSQL for SUQL queries

1. In the command line for your database (e.g. `psql restaurants`), copy and paste all content under `custom_functions.sql`. This will define the necessary free text functions (including `answer` and `summary`) under your PostgreSQL database.
   - Note that these functions are making calls to a local address, by default 8500, defined in `src/suql/free_text_fcns_server.py` (see step 5 below). If you change this address make sure to also change the address below.

2. The user-facing parts of this system should only require **SELECT** privilege (it would be safe to grant only SELECT privilege for GPT-generated queries). By default, this user is named `select_user` with password `select_user` in `postgresql_connection.py`.
   - If you are ok with this user name + login, run the following code in your `psql` command line to create this role:
```sql
CREATE ROLE select_user WITH PASSWORD 'select_user';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user;
ALTER ROLE select_user LOGIN;
```
   - If you wish to choose another credential:
      - This user also appears once in `src/suql/sql_free_text_support/execute_free_text_sql.py` under function `suql_execute` of the SUQL compiler, so that intermediate tables created can be queried. Please also change it to match your user name.
      - This user also appears in `src/suql/faiss_embedding.py`, please make sure to also change there.

3. A few parts of the SUQL compiler require the permission to **CREATE** a temporary table. These temporary tables are immediately deleted after the compiler finishes processing a query. By default, this user is named `creator_role` with password `creator_role`. Similar to above:
   - If you are ok with this user name + login, run the following code in your `psql` command line to create this role:
```sql
CREATE ROLE creator_role WITH PASSWORD 'creator_role';
GRANT CREATE ON DATABASE <your-db-name> TO creator_role;
GRANT CREATE ON SCHEMA public TO creator_role;
ALTER ROLE creator_role LOGIN;
```
    
   - If you wish to choose another credential:
      - You would need to create a user with the privilege to **CREATE** tables under your database, and change `creator_role` to match that user and password.
      - This user appears once in `src/suql/sql_free_text_support/execute_free_text_sql.py` under function `suql_execute` of the SUQL compiler.

## Initialize two SUQL servers

4. Set up an embedding server for the SUQL compiler to query. Go to `src/suql/faiss_embedding.py`, and modify the lines
```python
embedding_store.add(
   table_name="restaurants",
   primary_key_field_name="_id",
   free_text_field_name="reviews",
   db_name="restaurants",
   user="select_user",
   password="select_user"
)
```
under `if __name__ == "__main__":` to match your database with its column names. 
   - For instance, this line instructs the SUQL compiler to set up an embedding server for the `restaurants` database, which has `_id` column as the unique row identifier, for the `popular_dishes` column (such column need to be of type `TEXT` or `TEXT[]`, or other fixed-length strings/list of strings) under table `restaurants`. This is executed with user privilege `user="select_user"` and `password="select_user"`;
   - By default, this will be set up on port 8501, which is then called by `src/suql/execute_free_text_sql.py`. In case you need to use another port, please change both addresses.

5. Set up the backend server for the `answer`, `summary` functions. In a separate terminal, first set up OpenAI API key with `export OPENAI_API_KEY=[your OpenAI API key here]`. Then, run `python src/suql/free_text_fcns_server.py`.
   - As you probably noticed, the code in `custom_functions.sql` is just making queries to this server, which handles the LLM API calls. If you changed the address in `custom_functions.sql`, then also update the address under `if __name__ == "__main__":`.

## Write 2 few-shot prompts

We are very close to a fully-working LLM-powered agent!

6. Write a semantic parser prompt that asks LLMs to generate SUQL. Look at `src/suql/prompts/parser_suql.prompt` for an example on how this is done on restaurants. 
   - You should change the schema declaration and few-shot examples to match your domain. Make sure to include examples on how to use the `answer` function;
   - Feel free to incorporate other prompting techniques from text2sql advances.

7. In many cases, it is ideal to limit what kind of user query would your agent respond to. There is a classifier to determine whether a user utterance requires database access. See `src/suql/prompts/if_db_classification.prompt` for an example on how this is done on restaurants.
   - If you decide to keep this, then modify the examples to match your domain;
   - If you decide to delete this, then simply set the line `enable_classifier=True` to be `enable_classifier=False`.

8. In a separate terminal from the two servers above, run `export OPENAI_API_KEY=[your OpenAI API key here]`. Test with `python src/suql/agent.py`. You should be able to interact with your agent on your CLI!

# Set up with Chainlit

Code to set up a front-end powered by [Chainlit](https://github.com/Chainlit/chainlit) is on `wip/chainlit`.