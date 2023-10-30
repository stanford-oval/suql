# Installation / Usage tutorial

## Installing [PostgreSQL database](https://www.postgresql.org/)

1. Follow the instruction there to install a postgreSQL database. For instance, if you are installing on Ubuntu, then follow section `PostgreSQL Apt Repository` at https://www.postgresql.org/download/linux/ubuntu/.

2. After that, the SUQL compiler needs to make use of python functions within postgreSQL. This is done via the `postgresql-plpython3` language. If you are using Ubuntu, simply run `sudo apt-get install postgresql-plpython3-15`. *TODO: pls contribute on how to do this on other platforms*.

3. Then, in your database's command line (incurred via `psql your_database_name`), do `CREATE EXTENSION plpython3u;`. This loads this language into the current db.

## Installing python dependencies of SUQL

1. Install dependencies in `requirements.txt`;

2. Run `python -m spacy download en_core_web_sm`;

## A breakdown of how SUQL works on a high level

### The `answer`` function

The main free text function introduced by SUQL is the `answer` function. This function can appear in both projection and filter clauses. The behavior and implementation when it appears in projections are different when it appears in filters.

- If `answer` appears as a projection field, e.g. `SELECT answer("Town/Village_Info", 'is this town known for its tin mining?') FROM "my_table" WHERE "Team" = 'Veor';`, then it is asking GPT the question `is this town known for its tin mining?` based on the values in the field `"Town/Village_Info"`. This prompt is under [this file](prompts/review_qa.prompt). *(named as `review_qa.prompt` because we used to ask a lot of questions about reviews in YelpBot).*

- If `answer` appears as filter, then it is more complicated. Consider `SELECT "Team" FROM "validation_table_20" WHERE answer("Town/Village_Info", 'is this town known for its tin mining?') = 'Yes';`. The semantics of this SQL is to say to select a row such that the `"Town/Village_Info"` value supports the fact that `this town is know for its tin mining`. The SUQL compiler has implemented some optimization to make this filtering possible. The high-level algorithm is the following:
    
    - First do a predicate-based parsing to understand what are the predicates involving structured columns and what are for free text columns. Always try to execute the ones on structured columns first (there is a general predicate-based ordering algorithm implemented there);

    - Then, when it comes to dealing with free text columns: First use an embedding model to find the closest match to the query string `'is this town known for its tin mining?'`, based on the text inside `"Town/Village_Info"``

    - For the top results, go through the results one by one and ask GPT the question whether  this usage is correct:

    `answer("Town/Village_Info", 'is this town known for its tin mining?') = 'Yes'`

The reason why we needed an embedding model is if we are working on restaurants for example, where we have > 1000 rows, it is infeasible to ask GPT all rows. Thus, we use the embedding model to fetch the top answers, which are then fed into the second step.

Now, this optimization is only useful if I am looking for only a certain number of rows with a `LIMIT`` clause. If the computation is in regards to all rows, then this optimization will not take effect and instead a questions to GPT will be asked for all rows. (Does this make sense?)

### The classifier concept

If a text enum field only has a few values, then we can directly put it in the semantic parser prompt, e.g., the `price` field under `parser_sql.prompt`.

However if there are a lot of choices, then a classifier is needed to conform a Enum value prediction to one of the given values. For instance, given a predicted SUQL query: `SELECT * FROM restaurants WHERE 'coffee' = ANY (cuisines)`, the classifier would classify `'coffee'` to match with `'coffee \& tea'` and `'cafe'`. The resulting SUQL would be: `SELECT * FROM restaurants WHERE 'coffee \& tea' = ANY (cuisines) OR 'cafe' = ANY (cuisines)`. A parser-based approach has been implemented in the SUQL compiler and should work generally with all databases. This classifier looks at each predicate involving structured columns and executes it against the db. If it returns no result, it will attempt to do a classification to match the predicted value to one of the values in the db. This is done via `sql_free_text_support/prompts/field_classification.prompt`.

### Entry point to the SUQL compiler

The entry point to the SUQL compiler, which does the aforementioned optimizations, is the following snippet under `yelp_loop.py`:

```
visitor = SelectVisitor()
root = parse_sql(second_sql)
visitor(root)
second_sql = RawStream()(root)
```

The `SelectVisitor` might modify the original sql (`second_sql` here) and return the caller a new one. It might create a new temporary table named `temp_*`, which the modified SQL will be executed on.

## How to set up SUQL on your PostgreSQL database step-by-step.

First of all, note that all codes under the `main` branch are originally used for the restaurant chatbot YelpBot. There were some hard coding to make this happen. For instance, in `yelp_loop.py`, there is still a step of a regex-based heuristics to add a `LIMIT 3` clause to a predicted SQL, if a `LIMIT` clause does not exist. This might no longer be needed in your use case. *(TODO: I will be adding more heuristics used in the main branch and summarize them here)*.

Here is a rough breakdown of what you need to do to set up SUQL on your domain:

0. Set up OpenAI API key and connection. This is done via the `prompt_continuation.py` file under the root folder. This file is using Azure's API. If you need to use Google/OpenAI's API, substitute it there. (An example of how this is done for OpenAI is done on the `hybridQA` branch.)

1. Write a semantic parser prompt, and substitute `prompts/parser_sql.prompt` with it. Include examples of how to use the `answer` function.

2. Set up an embedding server for the SUQL compiler to query. Go to `sql_free_text_support/embedding_support.py`, and modify the line `embedding_store.add("restaurants", "_id", "popular_dishes")` and `embedding_store.add("restaurants", "_id", "reviews")` to match your database with its column names. For instance, the current two lines there are saying that set up an embedding server for the `restaurants` database, which has `_id` column as the unique row identifier, for the `popular_dishes` and `reviews` columns (these columns need to be of type `TEXT` or `TEXT[]`).

3. In the command line for your database, copy and paste all content under `custom_functions.sql`. This will define the `answer` and `summary` functions under your PostgreSQL database.

4. Set up the backend server for the `answer`, `summary` functions. As you probably noticed, the code in `custom_functions.sql` is just making queries to a server. This server can be instantiated by running `python reviews_server.py`.

5. Test with `python yelp_loop.py`.

# Known issues

1. if you see error msgs similar to `PermissionError: [Errno 13] Permission denied: '/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4.2749b823-646b-45d7-9fcf-11414469d900.tmp'`. Refer to https://github.com/openai/tiktoken/issues/75. A likely solution is setting `TIKTOKEN_CACHE_DIR=""`.