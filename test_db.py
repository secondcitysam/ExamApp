import MySQLdb, os
from dotenv import load_dotenv
load_dotenv()  # Load values from .env file

try:
    connection = MySQLdb.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        passwd=os.getenv('MYSQL_PASSWORD'),
        db=os.getenv('MYSQL_DB'),
        port=int(os.getenv('MYSQL_PORT', 3306)),
        use_unicode=True,
        charset='utf8mb4'
    )

    print("✅ Connected successfully to Railway MySQL!")
except Exception as e:
    print("❌ Connection failed:", e)
