import psycopg2
import time

def execute_sql(sql_query):
    
    start_time = time.time()

    # Establish a connection to the PostgreSQL database
    conn = psycopg2.connect(
        database="restaurants",
        user="yelpbot_user",
        password="yelpbot_user",
        host="127.0.0.1",
        port="5432",
        options='-c statement_timeout=45000'
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        return [], [], elapsed_time

    # Close the cursor and connection
    cursor.close()
    conn.close()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return list(results), column_names, elapsed_time

if __name__ == "__main__":
    print(execute_sql("SELECT * FROM restaurants LIMIT 1"))