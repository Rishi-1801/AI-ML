import sqlite3

# Connect to sqlite
connection=sqlite3.connect("student.db")

# create  cursor obj to insert record,create table
cursor=connection.cursor()

#create the table
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)"""

cursor.execute(table_info)  # here actually the table is created

# Insert the rows
cursor.execute('''Insert Into STUDENT values('Bunny','Gaming','A',90)''')
cursor.execute('''Insert Into STUDENT values('Trinay','Data Science','A',85)''')
cursor.execute('''Insert Into STUDENT values('Rishi','AIML','A',90)''')
cursor.execute('''Insert Into STUDENT values('Guru','Electronics','B',92)''')
cursor.execute('''Insert Into STUDENT values('Akash','DEVOPS','B',80)''')

# Print the all the records
print("The inserted records are:")
data=cursor.execute('''Select * from STUDENT''')

for row in data:
    print(row)

# Commit your changes and close the connection with database
connection.commit()
connection.close()