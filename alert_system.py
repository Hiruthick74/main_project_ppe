"alert system.py"
import requests
from pygame import mixer
import time
import csv
from datetime import datetime
import os

# Telegram config
BOT_TOKEN = "7283109516:AAHjpO0uS8prIQidrnfrufX0HcSb_h8v130"
CHAT_ID = "5510984298"

# Sound file
ALERT_SOUND = r"C:\\Users\\rselv\\Music\\ppe sound.mp3"

# Cooldown (10 seconds)
LAST_ALERT_TIME = 0
COOLDOWN = 10  # seconds

# Log file
LOG_FILE = "violations.csv"

# Initialize sound once
mixer.init()

# Create log file if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected PPE", "Missing PPE"])


def send_telegram_alert(message):
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        resp = requests.post(url, data=data, timeout=5)
        print(f"📨 Telegram Response: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Telegram Error: {e}")


def log_violation(ppe_count, missing_items):
    """Save violations to CSV log"""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"Helmet:{ppe_count['Helmet']}, Vest:{ppe_count['Vest']}, Gloves:{ppe_count['Gloves']}, Goggles:{ppe_count['Goggles']}, Shoes:{ppe_count['Shoes']}",
            ", ".join(missing_items)
        ])


def trigger_alert(ppe_count):
    """Trigger sound + Telegram alert with cooldown"""
    global LAST_ALERT_TIME
    now = time.time()

    helmet = ppe_count.get("Helmet", 0)
    vest = ppe_count.get("Vest", 0)
    gloves = ppe_count.get("Gloves", 0)
    goggles = ppe_count.get("Goggles", 0)
    shoes = ppe_count.get("Shoes", 0)

    missing_items = []
    if helmet == 0: missing_items.append("Helmet")
    if vest == 0: missing_items.append("Vest")
    if gloves == 0: missing_items.append("Gloves")
    if goggles == 0: missing_items.append("Goggles")
    if shoes == 0: missing_items.append("Shoes")

    if missing_items and (now - LAST_ALERT_TIME > COOLDOWN):
        LAST_ALERT_TIME = now
        missing_text = ", ".join(missing_items)

        alert_msg = (
            f"⚠️ ALERT: Missing PPE Detected!\n"
            f"Detected → Helmet:{helmet}, Vest:{vest}, Gloves:{gloves}, Goggles:{goggles}, Shoes:{shoes}\n"
            f"❌ Missing → {missing_text}"
        )

        # Play sound alert
        mixer.music.load(ALERT_SOUND)
        mixer.music.play()

        # Telegram + Log
        send_telegram_alert(alert_msg)
        log_violation(ppe_count, missing_items)
