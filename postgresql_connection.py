import psycopg2
import time

def execute_sql(sql_query):
    
    start_time = time.time()

    # Establish a connection to the PostgreSQL database
    # TODO: not sure if timeout this way is actually working
    conn = psycopg2.connect(
        database="restaurants",
        user="yelpbot_user",
        password="yelpbot_user",
        host="127.0.0.1",
        port="5432",
        options='-c statement_timeout=60000 -c client_encoding=UTF8'
    )

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    
    cursor.execute("SET statement_timeout = 60000")  # Set timeout to 60 seconds
    conn.commit()

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
    print(execute_sql("SELECT * FROM restaurants LIMIT 1;"))
    print(execute_sql("SELECT reviews FROM restaurants WHERE name ILIKE 'Bistronomie by Baumé' LIMIT 1;"))
    print(execute_sql("SELECT *, summary(reviews) FROM restaurants WHERE 'chef\'s table' = ANY (popular_dishes) AND location = 'Palo Alto' LIMIT 1;"))