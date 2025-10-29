# violation.py
import sqlite3
from datetime import datetime

DB = "violations.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        camera_id TEXT,
        source TEXT,
        detected_str TEXT,
        helmet INTEGER, vest INTEGER, gloves INTEGER, goggles INTEGER, shoes INTEGER,
        violations_count INTEGER,
        hour INTEGER,
        time_of_day TEXT,
        risk_label TEXT,
        predicted_risk TEXT
    )
    """)
    conn.commit()
    conn.close()

def parse_detected(s):
    defaults = {"Helmet":0,"Vest":0,"Gloves":0,"Goggles":0,"Shoes":0}
    try:
        parts = [p.strip() for p in str(s).split(",")]
        for p in parts:
            if ":" in p:
                k, v = p.split(":")
                k = k.strip()
                v = int(v.strip())
                if k in defaults:
                    defaults[k] = v
    except:
        pass
    return defaults

def append_detection(detected_str, camera_id=None, source="webcam", predicted_risk=None):
    counts = parse_detected(detected_str)
    violations_count = sum(1 for k in ["Helmet","Vest","Gloves","Goggles","Shoes"] if counts.get(k,0) == 0)
    hour = datetime.now().hour
    tod = "morning" if hour < 12 else ("afternoon" if hour < 18 else "night")

    conn = sqlite3.connect(DB, timeout=10)
    c = conn.cursor()
    c.execute("""
      INSERT INTO violations (timestamp, camera_id, source, detected_str, helmet, vest, gloves, goggles, shoes, violations_count, hour, time_of_day, risk_label, predicted_risk)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        camera_id,
        source,
        detected_str,
        counts.get("Helmet",0),
        counts.get("Vest",0),
        counts.get("Gloves",0),
        counts.get("Goggles",0),
        counts.get("Shoes",0),
        violations_count,
        hour,
        tod,
        ("Low" if violations_count==0 else "Medium" if violations_count<=2 else "High"),
        str(predicted_risk) if predicted_risk is not None else None
    ))
    conn.commit()
    conn.close()
