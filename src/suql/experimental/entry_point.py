from suql.sql_free_text_support.execute_free_text_sql import suql_execute
import litellm

litellm.set_verbose = True

suql = "SELECT * FROM log_normal WHERE answer(content, 'Is there an error?') = 'No'"
table_w_ids = {"log_normal": "line_id"}
database = "postgres"
# Try better: llm_model_name="gpt-4o-mini"
answer = suql_execute(suql, table_w_ids, database)
print(answer)
