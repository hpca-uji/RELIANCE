import mysql.connector
import os

# Function that returns a connection to the DB
def connectToDB():
    connection_config_dict = {
        'user': os.environ['DB_USER'],
        'password': os.environ['DB_PWD'],
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': int(os.environ.get('DB_PORT', '3306')),
        'database': os.environ.get('DB_NAME', 'app_db')
    }
    connection = None
    try:
        connection = mysql.connector.connect(**connection_config_dict)
    except mysql.connector.Error as e:
        print("Error while connecting to MySQL", e)

    return connection
# Function that inserts a prediction into the DB
def insertInference(id, predictions, receivedtime, processedtime, acquisitiontime=None):
    connection = connectToDB()
    if connection.is_connected():
        try:
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO ChestPredictions (id, predictions, acquisitiontime, receivedtime, processedtime) 
                                    VALUES (%s, %s, %s, %s, %s) """
            chestPrediction = (id, predictions, acquisitiontime, receivedtime, processedtime)
            cursor.execute(mySql_insert_query, chestPrediction)
            connection.commit()
            print("Chest prediction inserted successfully into ChestPredictions table")
        except mysql.connector.Error as error:
            print("Failed to insert into ChestPredictions table {}".format(error))
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")
