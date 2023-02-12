import sqlite3
import pandas as pd

# def csv_to_sqlite():
#     df = pd.read_csv('train.csv')
#     columns = tuple(df.columns)
#     con = sqlite3.connect("database.db")
#     cur = con.cursor()
#     cur.execute(f"CREATE TABLE t {columns};") # use your column names here
#     with open('train.csv','r') as fin: # `with` statement available in 2.5+
#         dr = csv.DictReader(fin) # comma is default delimiter
#         to_db = [(i['state'], i['account_length'],i['area_code'],
#                   i['international_plan'], i['voice_mail_plan'], i['number_vmail_messages'],
#                   i['total_day_minutes'], i['total_day_calls'],
#                   i['total_day_charge'], i['total_eve_minutes'],
#                   i['total_eve_calls'], i['total_eve_charge'],
#                   i['total_night_minutes'], i['total_night_calls'],
#                   i['total_night_charge'], i['total_intl_minutes'],
#                   i['total_intl_calls'], i['total_intl_charge'],
#                   i['number_customer_service_calls'], i['churn']) for i in (dr)]
#
#     cur.executemany(f"INSERT INTO t {columns} VALUES (?, ?, ?, ?, ?,"
#                     f" ?, ?, ?, ?, ?,"
#                     f"?, ?, ?, ?, ?, "
#                     f"?, ?, ?, ?, ?);", to_db)
#     con.commit()
#     con.close()


# Read sqlite query results into a pandas DataFrame

# import MySQLdb
#
# mydb = MySQLdb.connect(
#   host="127.0.0.1",
#   user="babayaga",
#   password="babayaga",
#   port= 3306
# )
#
# mycursor = mydb.cursor()
#
# mycursor.execute("CREATE DATABASE mydatabase")
con = sqlite3.connect("database.db")
pd.read_csv('train.csv').to_sql('train', con, if_exists='append', index=False)
df = pd.read_sql_query("SELECT * from 'train'", con)
print(df.head())
con.close()

from sqlalchemy import create_engine
import pandas as pd

# db_connection_str = 'mysql+pymysql://babayaga:babayaga@mysql_host/mysql_db'
# db_connection = create_engine(db_connection_str)
# pd.read_csv('train.csv').to_sql('train-mysql', db_connection, if_exists='append', index=False)
# df = pd.read_sql('SELECT * FROM train-mysql', con=db_connection)
# print(columns)
