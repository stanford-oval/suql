import psycopg2
import time
import re

def execute_sql(sql_query):
    if not ("LIMIT" in sql_query):
        sql_query = re.sub(r';$', ' LIMIT 5;', sql_query, flags=re.MULTILINE)
    
    start_time = time.time()

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        database="restaurants",
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432"
    )

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    try:
        print("executing SQL {}".format(sql_query))
        # Execute the SQL query
        cursor.execute(sql_query)

        # Fetch all the results
        results = cursor.fetchall()
        
        column_names = [desc[0] for desc in cursor.description]


    except psycopg2.Error as e:
        print("Error executing SQL query:", e)
        return [], []

    # Close the cursor and connection
    cursor.close()
    conn.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return list(results), column_names, elapsed_time

if __name__ == "__main__":
    print(execute_sql("SELECT * FROM restaurants LIMIT 1"))