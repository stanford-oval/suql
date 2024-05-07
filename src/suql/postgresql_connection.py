import time

import psycopg2


def execute_sql(
    sql_query,
    database,
    user="select_user",
    password="select_user",
    data=None,
    commit_in_lieu_fetch=False,
    no_print=False,
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

    try:
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

    except psycopg2.Error as e:
        print("Error executing SQL query:", e)
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

    try:
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

    except psycopg2.Error as e:
        print("Error executing SQL query:", e)
        if unprotected:
            raise e
        return [], []

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return list(results), column_info


if __name__ == "__main__":
    print(execute_sql("SELECT * FROM restaurants LIMIT 1;"))