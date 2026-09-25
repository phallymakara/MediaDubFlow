import sqlite3

conn = sqlite3.connect("mediadubflow.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()

for (table_name,) in tables:
    print(f"Table: {table_name}")
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    for col in cols:
        nullable = "NOT NULL" if col[3] else "NULL"
        print(f"  {col[1]:<35} {col[2]:<20} {nullable}")
    print()

conn.close()
print("Verification complete.")
