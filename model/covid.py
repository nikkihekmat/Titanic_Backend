import psycopg2

# Function to create the covid_data table


def create_covid_data_table():
    conn = psycopg2.connect(
        dbname="your_database_name",
        user="your_database_user",
        password="your_database_password",
        host="your_database_host"
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS covid_data (
            country TEXT PRIMARY KEY,
            total_cases INTEGER,
            total_deaths INTEGER
        )
    """)
    conn.commit()
    conn.close()

# Call the function to create the table


if __name__ == "__main__":
    create_covid_data_table()
