"""
SQLite Database manager for Marbl Meat QC Pipeline.

Stores:
  - Batches (Module 1: Intake & Quality Assessment)
  - Visual inspection records (Modules 2 & 3: Freshness, Size, Visible Fat Coverage)
  - Staged Drip Loss records (Future evaluation)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "marbl.db"


def get_db_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory enabled."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def init_db():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Batches table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS batches (
        batch_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        meat_type TEXT NOT NULL,
        initial_ph REAL NOT NULL,
        receiving_temp_c REAL NOT NULL,
        time_since_slaughter_hrs REAL NOT NULL,
        transport_temp_avg_c REAL NOT NULL,
        transport_duration_hrs REAL NOT NULL,
        transport_excursion INTEGER NOT NULL DEFAULT 0,
        packaging_type TEXT NOT NULL,
        initial_weight_val REAL NOT NULL,
        initial_weight_unit TEXT NOT NULL DEFAULT 'kg',
        shelf_life_days REAL NOT NULL,
        shelf_life_min_days REAL NOT NULL,
        shelf_life_max_days REAL NOT NULL,
        confidence_score REAL NOT NULL,
        recommended_temp_c TEXT NOT NULL,
        warnings_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'intake_complete'
    );
    """)

    # 2. Visual inspection records linked to batch
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inspection_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        decision TEXT NOT NULL,
        good_confidence REAL,
        spoiled_confidence REAL,
        size_category TEXT,
        longest_dimension_mm REAL,
        piece_count_estimate INTEGER,
        fat_coverage_pct REAL,
        lean_coverage_pct REAL,
        fat_category TEXT,
        segmentation_quality REAL,
        FOREIGN KEY (batch_id) REFERENCES batches (batch_id) ON DELETE CASCADE
    );
    """)

    # 3. Drip loss tests (staged for future development)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drip_loss_staged (
        batch_id TEXT PRIMARY KEY,
        initial_weight_val REAL NOT NULL,
        initial_weight_unit TEXT NOT NULL,
        final_weight_val REAL,
        final_weight_unit TEXT,
        storage_duration_hrs REAL,
        storage_temp_c REAL,
        drip_loss_pct REAL,
        test_status TEXT NOT NULL DEFAULT 'in_development',
        notes TEXT,
        FOREIGN KEY (batch_id) REFERENCES batches (batch_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()


def generate_batch_id() -> str:
    """Generate a clean, standardized Batch ID like MB-20261102-8F2A."""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"MB-{date_str}-{short_uuid}"


def create_batch(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new batch and its staged drip loss record."""
    batch_id = batch_data.get("batch_id") or generate_batch_id()
    created_at = batch_data.get("created_at") or datetime.now().isoformat()
    warnings_json = json.dumps(batch_data.get("warnings", []))

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO batches (
        batch_id, created_at, meat_type, initial_ph, receiving_temp_c,
        time_since_slaughter_hrs, transport_temp_avg_c, transport_duration_hrs,
        transport_excursion, packaging_type, initial_weight_val, initial_weight_unit,
        shelf_life_days, shelf_life_min_days, shelf_life_max_days,
        confidence_score, recommended_temp_c, warnings_json, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id,
        created_at,
        batch_data["meat_type"],
        float(batch_data["initial_ph"]),
        float(batch_data["receiving_temp_c"]),
        float(batch_data.get("time_since_slaughter_hrs", 24.0)),
        float(batch_data.get("transport_temp_avg_c", 2.0)),
        float(batch_data.get("transport_duration_hrs", 4.0)),
        1 if batch_data.get("transport_excursion") else 0,
        batch_data["packaging_type"],
        float(batch_data["initial_weight_val"]),
        batch_data.get("initial_weight_unit", "kg"),
        float(batch_data["shelf_life_days"]),
        float(batch_data["shelf_life_min_days"]),
        float(batch_data["shelf_life_max_days"]),
        float(batch_data.get("confidence_score", 0.85)),
        batch_data.get("recommended_temp_c", "0°C to +2°C"),
        warnings_json,
        batch_data.get("status", "intake_complete"),
    ))

    # Stage drip loss record
    cursor.execute("""
    INSERT INTO drip_loss_staged (
        batch_id, initial_weight_val, initial_weight_unit, test_status, notes
    ) VALUES (?, ?, ?, 'in_development', 'Initial weight recorded. Final weight pending standardized 48h storage test.')
    """, (
        batch_id,
        float(batch_data["initial_weight_val"]),
        batch_data.get("initial_weight_unit", "kg"),
    ))

    conn.commit()
    conn.close()

    return get_batch(batch_id)


def get_batch(batch_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full batch record including warnings and inspection counts."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    batch = dict(row)
    try:
        batch["warnings"] = json.loads(batch["warnings_json"])
    except Exception:
        batch["warnings"] = []

    # Get count of inspections
    cursor.execute("""
    SELECT 
        COUNT(*) as total_inspections,
        SUM(CASE WHEN decision = 'discard' THEN 1 ELSE 0 END) as discarded_count,
        SUM(CASE WHEN decision != 'discard' THEN 1 ELSE 0 END) as passed_count,
        AVG(fat_coverage_pct) as avg_fat_pct
    FROM inspection_records WHERE batch_id = ?
    """, (batch_id,))
    stats = cursor.fetchone()
    if stats:
        batch["total_inspections"] = stats["total_inspections"] or 0
        batch["discarded_count"] = stats["discarded_count"] or 0
        batch["passed_count"] = stats["passed_count"] or 0
        batch["avg_fat_pct"] = round(stats["avg_fat_pct"], 1) if stats["avg_fat_pct"] is not None else None

    # Staged drip loss info
    cursor.execute("SELECT * FROM drip_loss_staged WHERE batch_id = ?", (batch_id,))
    drip = cursor.fetchone()
    batch["drip_loss_staged"] = dict(drip) if drip else None

    conn.close()
    return batch


def list_batches(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent batches in reverse chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT b.*, 
           COUNT(r.id) as inspection_count
    FROM batches b
    LEFT JOIN inspection_records r ON b.batch_id = r.batch_id
    GROUP BY b.batch_id
    ORDER BY b.created_at DESC
    LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    results = []
    for r in rows:
        item = dict(r)
        try:
            item["warnings"] = json.loads(item["warnings_json"])
        except Exception:
            item["warnings"] = []
        results.append(item)

    conn.close()
    return results


def add_inspection_record(batch_id: str, record_data: Dict[str, Any]) -> int:
    """Link an AI visual inspection result to a Batch ID."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO inspection_records (
        batch_id, filename, timestamp, decision, good_confidence, spoiled_confidence,
        size_category, longest_dimension_mm, piece_count_estimate,
        fat_coverage_pct, lean_coverage_pct, fat_category, segmentation_quality
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        batch_id,
        record_data.get("filename", "unknown"),
        datetime.now().isoformat(),
        record_data.get("decision", "unknown"),
        record_data.get("good_confidence"),
        record_data.get("spoiled_confidence"),
        record_data.get("size_category"),
        record_data.get("longest_dimension_mm"),
        record_data.get("piece_count_estimate"),
        record_data.get("fat_coverage_pct"),
        record_data.get("lean_coverage_pct"),
        record_data.get("fat_category"),
        record_data.get("segmentation_quality"),
    ))

    rec_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rec_id


def get_batch_inspection_records(batch_id: str) -> List[Dict[str, Any]]:
    """Get all inspection records for a batch."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM inspection_records WHERE batch_id = ? ORDER BY id ASC
    """, (batch_id,))

    rows = cursor.fetchall()
    records = [dict(r) for r in rows]
    conn.close()
    return records

