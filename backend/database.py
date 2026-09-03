
import sqlite3, hashlib, secrets, os
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "healthcare.db"
if os.getenv("VERCEL"):
    DEFAULT_DB = Path("/tmp/healthcare.db")
DB = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DB)))

def conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=30000")
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db():
    c = conn()

    c.execute("""CREATE TABLE IF NOT EXISTS patient_accounts(
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL)""")

    c.execute("""CREATE TABLE IF NOT EXISTS assessments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        age REAL,
        gender INTEGER,
        height REAL,
        weight REAL,
        ap_hi REAL,
        ap_lo REAL,
        cholesterol INTEGER,
        gluc INTEGER,
        smoke INTEGER,
        alco INTEGER,
        active INTEGER,
        family_history INTEGER,
        ecg_status TEXT,
        bmi REAL,
        risk_probability REAL,
        risk_level TEXT,
        model TEXT,
        created_at TEXT NOT NULL)""")

    # Safe migration for your existing database
    columns = [
        row["name"]
        for row in c.execute("PRAGMA table_info(assessments)").fetchall()
    ]

    if "model" not in columns:
        c.execute("ALTER TABLE assessments ADD COLUMN model TEXT")

    c.execute("""CREATE TABLE IF NOT EXISTS assistant_jobs(
        id TEXT PRIMARY KEY,
        patient_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT,
        status TEXT NOT NULL,
        source TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT
    )""")

    c.commit()
    c.close()

def create_assistant_job(job_id, patient_id, question):
    c = conn()
    try:
        c.execute("""INSERT INTO assistant_jobs
        (id,patient_id,question,status,created_at)
        VALUES(?,?,?,?,?)""", (job_id, patient_id, question, "processing", datetime.now().isoformat()))
        c.commit()
    finally:
        c.close()

def complete_assistant_job(job_id, answer, source):
    c = conn()
    try:
        c.execute("""UPDATE assistant_jobs
        SET answer=?, status='completed', source=?, completed_at=? WHERE id=?""",
        (answer, source, datetime.now().isoformat(), job_id))
        c.commit()
    finally:
        c.close()

def get_assistant_job(job_id, patient_id):
    c = conn()
    try:
        row = c.execute("SELECT * FROM assistant_jobs WHERE id=? AND patient_id=?", (job_id, patient_id)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()

def get_pending_assistant_jobs(patient_id):
    c = conn()
    try:
        rows = c.execute("""SELECT * FROM assistant_jobs
            WHERE patient_id=? AND status='processing'
            ORDER BY created_at ASC""", (patient_id,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        c.close()

def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200000)
    return salt.hex() + ":" + digest.hex()

def verify_password(password, stored):
    salt, digest = stored.split(":")
    calc = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 200000
    ).hex()
    return secrets.compare_digest(calc, digest)

def create_patient(name, email, phone, password):
    c = conn()
    try:
        cur = c.execute(
            """INSERT INTO patient_accounts
            (full_name,email,phone,password_hash,created_at)
            VALUES (?,?,?,?,?)""",
            (name.strip(), email.lower().strip(), phone.strip(),
             hash_password(password), datetime.now().isoformat())
        )
        c.commit()
        return cur.lastrowid
    finally:
        c.close()

def get_login(email, password):
    c = conn()
    try:
        r = c.execute(
            "SELECT * FROM patient_accounts WHERE email=?",
            (email.lower().strip(),)
        ).fetchone()
        if r and verify_password(password, r["password_hash"]):
            return {
                "patient_id": r["patient_id"],
                "full_name": r["full_name"],
                "email": r["email"]
            }
        return None
    finally:
        c.close()

def save_assessment(patient_id, p, bmi, probability, risk, model_name):
    c = conn()

    try:
        c.execute("""INSERT INTO assessments(
            patient_id,age,gender,height,weight,ap_hi,ap_lo,
            cholesterol,gluc,smoke,alco,active,family_history,
            ecg_status,bmi,risk_probability,risk_level,model,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                patient_id,
                p.age,
                p.gender,
                p.height,
                p.weight,
                p.ap_hi,
                p.ap_lo,
                p.cholesterol,
                p.gluc,
                p.smoke,
                p.alco,
                p.active,
                p.family_history,
                p.ecg_status,
                bmi,
                probability,
                risk,
                model_name,
                datetime.now().isoformat()
            ))

        c.commit()

    finally:
        c.close()

def get_assessments(patient_id):
    c = conn()
    try:
        rows = c.execute(
            """SELECT * FROM assessments WHERE patient_id=?
               ORDER BY id DESC""", (patient_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()
