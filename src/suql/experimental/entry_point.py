from suql.sql_free_text_support.execute_free_text_sql import suql_execute
import litellm

# litellm.set_verbose = True

# suql = "SELECT * FROM log_normal WHERE answer(content, 'Does the log mention any system crashes or failures?') = 'Yes'"
# suql = "SELECT * FROM log_normal WHERE answer(content, 'Is there an error?') = 'Yes' or answer(content, 'Is the error serious?') = 'Yes' or answer(content, 'Is the error big?') = 'Yes'"
# suql = "SELECT * FROM log_normal WHERE is_relevant(content, 'What is happening?') OR is_relevant(content, 'What was wrong with the system?') OR answer(content, 'Is this line of log interesting?') = 'Yes'"
suql = "SELECT * FROM log_normal WHERE is_relevant(content, 'What was wrong with the system?')"

table_w_ids = {"log_normal": "line_id"}
database = "postgres"
# Try better: llm_model_name="gpt-4o-mini"
answer = suql_execute(suql, table_w_ids, database)
print(answer)
