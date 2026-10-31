import sqlite3
from datetime import datetime
from config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            track_id INTEGER,
            species TEXT,
            confidence REAL,
            bbox TEXT,
            threat_level TEXT,
            threat_score REAL,
            reasons TEXT,
            is_synced INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            log_type TEXT,
            message TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_detection(track_id, species, confidence, bbox, threat_level, threat_score, reasons):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO detections (timestamp, track_id, species, confidence, bbox, threat_level, threat_score, reasons, is_synced)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    ''', (timestamp, track_id, species, float(confidence), str(bbox), threat_level, float(threat_score), str(reasons)))
    conn.commit()
    conn.close()

def get_detections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def sync_offline_queue():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE detections SET is_synced = 1 WHERE is_synced = 0")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count