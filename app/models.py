"""
Data models for DrawScan AI.
Uses SQLite for the prototype via simple dataclasses + sqlite3.
"""

import hashlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "data/drawscan.db")


class SLATier(str, Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


SLA_LIMITS = {
    SLATier.STARTER: {
        "label": "Starter",
        "max_drawings_per_month": 10,
        "priority_processing": False,
        "api_access": False,
        "report_formats": ["xlsx"],
        "custom_templates": False,
        "color": "#7A7A7A",
    },
    SLATier.PROFESSIONAL: {
        "label": "Professional",
        "max_drawings_per_month": 50,
        "priority_processing": True,
        "api_access": True,
        "report_formats": ["xlsx", "docx", "pdf"],
        "custom_templates": False,
        "color": "#3769F9",
    },
    SLATier.ENTERPRISE: {
        "label": "Enterprise",
        "max_drawings_per_month": 999999,
        "priority_processing": True,
        "api_access": True,
        "report_formats": ["xlsx", "docx", "pdf"],
        "custom_templates": True,
        "color": "#61CE70",
    },
}


class ExtractionStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    OCR = "ocr"
    EXTRACTING = "extracting"
    REVIEW = "review"
    COMPLETE = "complete"
    FAILED = "failed"


def get_db():
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    return db


class db_session:
    """Context manager for SQLite connections. Ensures close on exit."""
    def __init__(self):
        self.db = None
    def __enter__(self):
        self.db = get_db()
        return self.db
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db:
            if exc_type is None:
                self.db.commit()
            self.db.close()
        return False


MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024
EXTRACTION_TIMEOUT_SECONDS = int(os.getenv("EXTRACTION_TIMEOUT", "180"))


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    check, _ = hash_password(password, salt)
    return check == hashed


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            sla_tier TEXT NOT NULL DEFAULT 'starter',
            is_admin INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS extractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploading',
            upload_path TEXT,
            image_path TEXT,
            extraction_json TEXT,
            report_path TEXT,
            accuracy_score REAL,
            processing_time_ms INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Seed admin user if none exists
    existing = db.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
    if existing == 0:
        pw_hash, pw_salt = hash_password("admin123")
        now = datetime.utcnow().isoformat()
        db.execute(
            "INSERT INTO users (email, name, password_hash, password_salt, sla_tier, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("admin@unitedrubber.com", "Admin", pw_hash, pw_salt, "enterprise", 1, now),
        )
        # Seed a demo user
        pw_hash2, pw_salt2 = hash_password("demo123")
        db.execute(
            "INSERT INTO users (email, name, password_hash, password_salt, sla_tier, is_admin, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("demo@unitedrubber.com", "Abhinav Sharma", pw_hash2, pw_salt2, "professional", 0, now),
        )
        db.commit()
    db.close()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    db = get_db()
    now = datetime.utcnow()
    expires = now + timedelta(days=7)
    db.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), expires.isoformat()),
    )
    db.execute("UPDATE users SET last_login = ? WHERE id = ?", (now.isoformat(), user_id))
    db.commit()
    db.close()
    return token


def get_user_by_session(token: str) -> Optional[dict]:
    if not token:
        return None
    db = get_db()
    row = db.execute(
        """SELECT u.* FROM users u
           JOIN sessions s ON s.user_id = u.id
           WHERE s.token = ? AND s.expires_at > ? AND u.is_active = 1""",
        (token, datetime.utcnow().isoformat()),
    ).fetchone()
    db.close()
    if row:
        return dict(row)
    return None


def authenticate(email: str, password: str) -> Optional[dict]:
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,)).fetchone()
    db.close()
    if user and verify_password(password, user["password_hash"], user["password_salt"]):
        return dict(user)
    return None


def get_user_extractions(user_id: int, limit: int = 20) -> list[dict]:
    db = get_db()
    rows = db.execute(
        "SELECT * FROM extractions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_user_monthly_usage(user_id: int) -> int:
    db = get_db()
    first_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0).isoformat()
    row = db.execute(
        "SELECT COUNT(*) as c FROM extractions WHERE user_id = ? AND created_at >= ?",
        (user_id, first_of_month),
    ).fetchone()
    db.close()
    return row["c"]


def create_extraction(user_id: int, filename: str, upload_path: str) -> int:
    db = get_db()
    now = datetime.utcnow().isoformat()
    cursor = db.execute(
        "INSERT INTO extractions (user_id, filename, status, upload_path, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, filename, "processing", upload_path, now),
    )
    db.commit()
    extraction_id = cursor.lastrowid
    db.close()
    return extraction_id


def update_extraction(extraction_id: int, **kwargs):
    db = get_db()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [extraction_id]
    db.execute(f"UPDATE extractions SET {sets} WHERE id = ?", vals)
    db.commit()
    db.close()


def get_extraction(extraction_id: int) -> Optional[dict]:
    db = get_db()
    row = db.execute("SELECT * FROM extractions WHERE id = ?", (extraction_id,)).fetchone()
    db.close()
    return dict(row) if row else None


def get_all_users() -> list[dict]:
    db = get_db()
    rows = db.execute("SELECT id, email, name, sla_tier, is_admin, is_active, created_at, last_login FROM users ORDER BY created_at DESC").fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_all_extractions(limit: int = 50) -> list[dict]:
    db = get_db()
    rows = db.execute(
        """SELECT e.*, u.name as user_name, u.email as user_email
           FROM extractions e JOIN users u ON e.user_id = u.id
           ORDER BY e.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def get_stats(user_id: int = None) -> dict:
    db = get_db()
    where = "WHERE user_id = ?" if user_id else ""
    params = (user_id,) if user_id else ()

    total = db.execute(f"SELECT COUNT(*) as c FROM extractions {where}", params).fetchone()["c"]
    complete = db.execute(
        f"SELECT COUNT(*) as c FROM extractions {where} {'AND' if where else 'WHERE'} status = 'complete'",
        params,
    ).fetchone()["c"]
    avg_time = db.execute(
        f"SELECT AVG(processing_time_ms) as a FROM extractions {where} {'AND' if where else 'WHERE'} processing_time_ms IS NOT NULL",
        params,
    ).fetchone()["a"]
    avg_accuracy = db.execute(
        f"SELECT AVG(accuracy_score) as a FROM extractions {where} {'AND' if where else 'WHERE'} accuracy_score IS NOT NULL",
        params,
    ).fetchone()["a"]

    db.close()
    return {
        "total_drawings": total,
        "reports_generated": complete,
        "avg_processing_time": round(avg_time / 1000, 1) if avg_time else 0,
        "avg_accuracy": round(avg_accuracy, 1) if avg_accuracy else 0,
    }
