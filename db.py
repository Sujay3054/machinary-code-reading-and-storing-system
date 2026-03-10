import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class MySQLClient:
    def __init__(self):
        try:
            # Get connection details from environment variables
            self.connection = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DATABASE"),
                port=int(os.getenv("MYSQL_PORT", 3306))
            )
            self.cursor = self.connection.cursor()
            self.create_table()
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            self.connection = None

    def create_table(self):
        # SQL to create the table if it does not exist
        create_query = """
        CREATE TABLE IF NOT EXISTS ocr_texts (
            record_id TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            x1 FLOAT,
            y1 FLOAT,
            x2 FLOAT,
            y2 FLOAT,
            recognized_text TEXT,
            confidence FLOAT
        );
        """
        self.cursor.execute(create_query)
        self.connection.commit()

    def insert_result(self, bbox, text, confidence):
        # SQL to insert OCR results into the database
        insert_query = """
        INSERT INTO ocr_texts (x1, y1, x2, y2, recognized_text, confidence)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        top_left = bbox[0]
        bottom_right = bbox[2]
        self.cursor.execute(insert_query, (
            top_left[0], top_left[1],
            bottom_right[0], bottom_right[1],
            text, confidence
        ))
        self.connection.commit()

    def close(self):
        # Close database connection
        if self.connection:
            self.cursor.close()
            self.connection.close()

