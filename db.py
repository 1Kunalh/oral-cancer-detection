import sqlite3
from pathlib import Path

DB_PATH = Path("oral_cancer.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT UNIQUE,
        patient_id TEXT,
        filename TEXT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL,
        review_status TEXT DEFAULT 'Pending',
        doctor_notes TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def add_case(case_id, patient_id, filename, image_path, prediction, confidence):
    conn = get_connection()
    conn.execute("""
        INSERT INTO cases (case_id, patient_id, filename, image_path, prediction, confidence, review_status)
        VALUES (?, ?, ?, ?, ?, ?, 'Pending')
    """, (case_id, patient_id, filename, image_path, prediction, confidence))
    conn.commit()
    conn.close()

def get_cases():
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM cases ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_review(case_id, review_status, doctor_notes):
    conn = get_connection()
    conn.execute("""
        UPDATE cases
        SET review_status = ?, doctor_notes = ?
        WHERE case_id = ?
    """, (review_status, doctor_notes, case_id))
    conn.commit()
    conn.close()