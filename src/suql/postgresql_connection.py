import time

import psycopg2
import sqlparse


def execute_sql(
    sql_query,
    database,
    user="select_user",
    password="select_user",
    data=None,
    commit_in_lieu_fetch=False,
    no_print=False,
    unprotected=False
):
    start_time = time.time()

    if password == "":
        conn = psycopg2.connect(
            dbname=database,
            user=user,
            host="/var/run/postgresql",
            port="5432",
            options="-c statement_timeout=30000 -c client_encoding=UTF8",
        )
    else:
        conn = psycopg2.connect(
            database=database,
            user=user,
            password=password,
            host="127.0.0.1",
            port="5432",
            options="-c statement_timeout=30000 -c client_encoding=UTF8",
        )

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    cursor.execute("SET statement_timeout = 30000")  # Set timeout to 60 seconds
    conn.commit()

    def sql_unprotected():
        if not no_print:
            print("executing SQL {}".format(sql_query))
        # Execute the SQL query
        if data:
            cursor.execute(sql_query, data)
        else:
            cursor.execute(sql_query)

        # Fetch all the results
        if commit_in_lieu_fetch:
            conn.commit()
            results = []
            column_names = []
        else:
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
        return results, column_names

    try:
        results, column_names = sql_unprotected()

    except psycopg2.Error as e:
        print("Error executing SQL query:", e)
        if unprotected:
            raise e
        end_time = time.time()
        elapsed_time = end_time - start_time
        return [], [], elapsed_time

    # Close the cursor and connection
    cursor.close()
    conn.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return list(results), column_names, elapsed_time


def execute_sql_with_column_info(
    sql_query,
    database,
    user="select_user",
    password="select_user",
    unprotected=False,
):
    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host="127.0.0.1",
        port="5432",
        options="-c statement_timeout=30000 -c client_encoding=UTF8",
    )

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    cursor.execute("SET statement_timeout = 30000")  # Set timeout to 60 seconds
    conn.commit()

    def sql_unprotected():
        # Execute the SQL query
        cursor.execute(sql_query)

        # Fetch all the results
        results = cursor.fetchall()

        column_names = [desc[0] for desc in cursor.description]
        column_type_oids = [desc[1] for desc in cursor.description]

        type_map = {}
        cursor.execute(
            "SELECT oid, typname FROM pg_type WHERE oid = ANY(%s);",
            ([desc[1] for desc in cursor.description],),
        )
        for oid, typname in cursor.fetchall():
            if typname.startswith("_"):
                type_map[oid] = typname[1:] + "[]"
            else:
                type_map[oid] = typname

        column_types = [type_map[oid] for oid in column_type_oids]
        column_info = list(zip(column_names, column_types))
        
        return results, column_info

    try:
        results, column_info = sql_unprotected()
    except psycopg2.Error as e:
        print("Error executing SQL query:", e)
        if unprotected:
            raise e
        return [], []

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return list(results), column_info

def split_sql_statements(query):
    def strip_trailing_comments(stmt):
        idx = len(stmt.tokens) - 1
        while idx >= 0:
            tok = stmt.tokens[idx]
            if tok.is_whitespace or sqlparse.utils.imt(tok, i=sqlparse.sql.Comment, t=sqlparse.tokens.Comment):
                stmt.tokens[idx] = sqlparse.sql.Token(sqlparse.tokens.Whitespace, " ")
            else:
                break
            idx -= 1
        return stmt

    def strip_trailing_semicolon(stmt):
        idx = len(stmt.tokens) - 1
        while idx >= 0:
            tok = stmt.tokens[idx]
            # we expect that trailing comments already are removed
            if not tok.is_whitespace:
                if sqlparse.utils.imt(tok, t=sqlparse.tokens.Punctuation) and tok.value == ";":
                    stmt.tokens[idx] = sqlparse.sql.Token(sqlparse.tokens.Whitespace, " ")
                break
            idx -= 1
        return stmt

    def is_empty_statement(stmt):
        # copy statement object. `copy.deepcopy` fails to do this, so just re-parse it
        st = sqlparse.engine.FilterStack()
        st.stmtprocess.append(sqlparse.filters.StripCommentsFilter())
        stmt = next(st.run(str(stmt)), None)
        if stmt is None:
            return True

        return str(stmt).strip() == ""

    stack = sqlparse.engine.FilterStack()

    result = [stmt for stmt in stack.run(query)]
    result = [strip_trailing_comments(stmt) for stmt in result]
    result = [strip_trailing_semicolon(stmt) for stmt in result]
    result = [str(stmt).strip() for stmt in result if not is_empty_statement(stmt)]

    if len(result) > 0:
        return result

    return [""]  # if all statements were empty - return a single empty statement

def query_is_select_no_limit(query):
    limit_keywords = ["LIMIT", "OFFSET"]
    
    def find_last_keyword_idx(parsed_query):
        for i in reversed(range(len(parsed_query.tokens))):
            if parsed_query.tokens[i].ttype in sqlparse.tokens.Keyword:
                return i
        return -1
    
    parsed_query = sqlparse.parse(query)[0]
    last_keyword_idx = find_last_keyword_idx(parsed_query)
    # Either invalid query or query that is not select
    if last_keyword_idx == -1 or parsed_query.tokens[0].value.upper() != "SELECT":
        return False

    no_limit = parsed_query.tokens[last_keyword_idx].value.upper() not in limit_keywords

    return no_limit

def add_limit_to_query(
    query,
    limit_query = " LIMIT 1000"
):
    parsed_query = sqlparse.parse(query)[0]
    limit_tokens = sqlparse.parse(limit_query)[0].tokens
    length = len(parsed_query.tokens)
    if parsed_query.tokens[length - 1].ttype == sqlparse.tokens.Punctuation:
        parsed_query.tokens[length - 1 : length - 1] = limit_tokens
    else:
        parsed_query.tokens += limit_tokens

    return str(parsed_query)

def apply_auto_limit(
    query_text,
    limit_query = " LIMIT 1000"
):
    def combine_sql_statements(queries):
        return ";\n".join(queries)
    
    queries = split_sql_statements(query_text)
    res = []
    for query in queries:
        if query_is_select_no_limit(query):
            query = add_limit_to_query(query, limit_query=limit_query)
        res.append(query)
    
    return combine_sql_statements(res)

if __name__ == "__main__":
    print(apply_auto_limit("SELECT * FROM restaurants LIMIT 1;"))
    print(apply_auto_limit("SELECT * FROM restaurants;"))