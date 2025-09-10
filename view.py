import sqlite3

def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create user table if not exists
def create_user_table(conn):
    sql_create_users_table = """
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_users_table)
    except sqlite3.Error as e:
        print(e)

# Create database connection and table
conn = create_connection()
create_user_table(conn)
cnn.close()





print("user table")

conn=sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('Select * from user')
rows=cursor.fetchall()
if not rows:
	print("No users existed")
for i in rows:
	print(i)
conn.commit()
conn.close()