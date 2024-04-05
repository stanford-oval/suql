# Overview

You should use this installation guide if you are integrating SUQL into a larger project and would like to have access to a Python SUQL endpoint. If you instead would like to spin up an agent using SUQL, we recommend checking out [install_source.md](install_source.md)

To install as a package, run:

```bash
pip install suql
```

# Set up other pre-requisites

1. Run `python -m spacy download en_core_web_sm`;

2. Install `faiss` by `conda install faiss`.

# Entry point:

The main entry point to the SUQL compiler is simply:
```python
from suql import suql_execute
```
See documentation for its specific usage. Now, let's set up your database and 2 servers to enable SUQL access on your db!

# Installing [PostgreSQL database](https://www.postgresql.org/)

1. Follow the instruction there to install a postgreSQL database. For instance, if you are installing on Ubuntu, then follow section `PostgreSQL Apt Repository` at https://www.postgresql.org/download/linux/ubuntu/.

2. After that, the SUQL compiler needs to make use of python functions within postgreSQL. This is done via the `postgresql-plpython3` language. If you are using Ubuntu, simply run `sudo apt-get install postgresql-plpython3-15`.

3. Then, in your database's command line (incurred via `psql your_database_name`), do `CREATE EXTENSION plpython3u;`. This loads this language into the current db.

# Preparations for database and servers

## Set up PSQL for SUQL queries

1. In the command line for your database (e.g. `psql restaurants`), copy and paste all content under `custom_functions.sql`. This will define the necessary free text functions (including `answer` and `summary`) under your PostgreSQL database.
   - Note that these functions are making calls to a local address, by default 8500. If you change this address, then also modify the port number in Step 5 below.

2. The user-facing parts of this system should only require **SELECT** privilege (it would be safe to grant only SELECT privilege for GPT-generated queries). By default, this user is named `select_user` with password `select_user` in `postgresql_connection.py`.
   - If you are ok with this user name + login, run the following code in your `psql` command line to create this role:
```sql
CREATE ROLE select_user WITH PASSWORD 'select_user';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO select_user;
ALTER ROLE select_user LOGIN;
```
   - If you wish to choose another credential, modify the keyword arguments `select_username` and `select_userpswd` of `suql_execute`.

3. A few parts of the SUQL compiler require the permission to **CREATE** a temporary table. These temporary tables are immediately deleted after the compiler finishes processing a query. By default, this user is named `creator_role` with password `creator_role`. Similar to above:
   - If you are ok with this user name + login, run the following code in your `psql` command line to create this role:
```sql
CREATE ROLE creator_role WITH PASSWORD 'creator_role';
GRANT CREATE ON DATABASE <your-db-name> TO creator_role;
GRANT CREATE ON SCHEMA public TO creator_role;
ALTER ROLE creator_role LOGIN;
```
   - If you wish to choose another credential, please create a user with the privilege to **CREATE** tables under your database, and modify the keyword arguments `create_username` and `create_userpswd` of `suql_execute`.

## Initialize two SUQL servers

4. Set up an embedding server for the SUQL compiler to query. Write a Python script with the following content and execute it with `python`:
```python
from suql.faiss_embedding import MultipleEmbeddingStore
embedding_store = MultipleEmbeddingStore()
embedding_store.add(
   table_name="restaurants",
   primary_key_field_name="_id",
   free_text_field_name="reviews",
   db_name="restaurants",
   user="select_user",
   password="select_user"
)

host = "127.0.0.1"
port = 8501
embedding_store.start_embedding_server(host = host, port = port)
```

- The line `embedding_store.add` instructs the SUQL compiler to set up an embedding server for the `restaurants` database, which has `_id` column as the unique row identifier, for the `popular_dishes` column (such column need to be of type `TEXT` or `TEXT[]`, or other fixed-length strings/list of strings) under table `restaurants`. This is executed with user privilege `user="select_user"` and `password="select_user"`. 
    - Make sure to modify the keyword arguments `select_username` and `select_userpswd` if you changed this user in Step 2 above;
    - You can add more columns as needed using ``embedding_store.add()`;
    - This will be set up on port 8501, which matches the default keyword argument `embedding_server_address` in `suql_execute`. Make sure both addresses match if you modify it.

5. Set up the backend server for the `answer`, `summary` functions. In a separate terminal, first set up OpenAI API key with `export OPENAI_API_KEY=[your OpenAI API key here]`. Write the following content into a Python script and execute in that terminal:
```python
from suql.free_text_fcns_server import start_free_text_fncs_server

host = "127.0.0.1"
port = 8500
start_free_text_fncs_server(host = host, port = port)
```

# Test with the entry point

You should be good to go! In a separate terminal, run `export OPENAI_API_KEY=[your OpenAI API key here]`, and test with

```python
from suql import suql_execute
# e.g. suql = "SELECT * FROM restaurants WHERE answer(reviews, 'is this a family-friendly restaurant?') = 'Yes' AND rating = 4;"
suql = "Your favorite SUQL"
suql_execute(suql)
```