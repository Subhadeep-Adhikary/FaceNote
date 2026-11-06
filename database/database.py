import sqlite3
from pathlib import Path
import numpy as np

DB_PATH = Path(__file__).resolve().parent / "faces.db"

def get_connection():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys= ON")
    return conn

def initialise_database():
    conn=get_connection()
    cur=conn.cursor()

    cur.execute('''
CREATE TABLE IF NOT EXISTS
persons(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
''')

    cur.execute('''
CREATE TABLE IF NOT EXISTS
face_embaddings(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      person_id INTEGER NOT NULL,
      embadding BLOB NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
      )
''')
    conn.commit()

    # Earlier versions created this table with a foreign key to the
    # nonexistent singular table "person". SQLite permits that schema but
    # fails when an embedding is inserted. Rebuild it while preserving rows.
    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(face_embaddings)"
    ).fetchall()
    if any(row[2] != "persons" for row in foreign_keys):
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN")
            conn.execute("""
                CREATE TABLE face_embaddings_fixed (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    embadding BLOB NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                INSERT INTO face_embaddings_fixed (id, person_id, embadding, created_at)
                SELECT id, person_id, embadding, created_at FROM face_embaddings
            """)
            conn.execute("DROP TABLE face_embaddings")
            conn.execute(
                "ALTER TABLE face_embaddings_fixed RENAME TO face_embaddings"
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    conn.close()
    print("DB initialised")

def add_person(name,embaddings):
    conn=get_connection()
    cur=conn.cursor()

    cur.execute(
        "INSERT INTO persons (name) VALUES (?)",
        (name,) 
    )
    person_id=cur.lastrowid

    for embadding in embaddings:
        embadding_blob=np.asarray(
            embadding,
            dtype=np.float32
        ).tobytes()

        cur.execute(
            """
            INSERT INTO face_embaddings
            (person_id,embadding)
            VALUES(?,?)
            """,(person_id,embadding_blob)
        )

    conn.commit()
    conn.close()

    return person_id

def get_all_embaddings():
    conn=get_connection()
    cur=conn.cursor()

    cur.execute("""
            SELECT
            persons.id,
            persons.name,
            face_embaddings.embadding
            FROM persons 
            JOIN face_embaddings
            ON persons.id=face_embaddings.person_id
   """)

    rows=cur.fetchall()
    conn.close()

    results=[]

    for person_id,name,blob in rows:
        embadding=np.frombuffer(
            blob,
            dtype=np.float32
        ).copy()

        results.append({
            "id":person_id,
            "name":name,
            "embedding":embadding
        })

    return results

def delete_person(person_id):
    conn=get_connection()
    conn.execute("DELETE FROM persons WHERE id= ?",
                 (person_id,)
                )
    conn.commit()
    conn.close()
