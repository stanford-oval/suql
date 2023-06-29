import psycopg2

def execute_sql(sql_query):

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
    return list(results), column_names