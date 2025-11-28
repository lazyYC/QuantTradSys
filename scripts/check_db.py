import sqlite3
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python check_db.py <db_path> <query>")
    sys.exit(1)

db_path = Path(sys.argv[1])
query = sys.argv[2]

if not db_path.exists():
    print(f"DB not found at {db_path}")
    sys.exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(query)
    for row in cursor:
        print(row)
    conn.close()
except Exception as e:
    print(f"Error: {e}")
