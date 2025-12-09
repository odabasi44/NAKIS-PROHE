import os
import json
from datetime import datetime
from flask import session

SETTINGS_FILE = "settings.json"
PREMIUM_FILE = "users.json"

TIER_RESTRICTIONS = {
    "free": [], 
    "starter": [],
    "pro": [],
    "unlimited": []
}

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {} # Varsayılan ayarlar config'den de gelebilir ama şimdilik boş dönelim
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE):
        return []
    try:
        with open(PREMIUM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_premium_users(data):
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_user_data_by_email(email):
    users = load_premium_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None

def check_user_status(email, tool, subtool):
    settings = load_settings()
    user_tier = "free"
    user_data = None

    if email != "guest":
        user_data = get_user_data_by_email(email)
        if user_data:
            try:
                # Tarih kontrolü
                if datetime.strptime(user_data.get("end_date"), "%Y-%m-%d") >= datetime.now():
                    user_tier = user_data.get("tier", "free")
            except:
                pass

    # Limit Kontrolü
    try:
        tool_limits = settings["limits"][tool][subtool][user_tier]
    except:
        tool_limits = 5 # Varsayılan limit

    current = 0
    if user_tier == "free":
        if "free_usage" not in session: session["free_usage"] = {}
        if tool not in session["free_usage"]: session["free_usage"][tool] = {}
        current = session["free_usage"][tool].get(subtool, 0)
    else:
        current = user_data.get("usage_stats", {}).get(subtool, 0)

    left = max(0, tool_limits - current)
    return {
        "allowed": current < tool_limits, 
        "reason": "limit" if current >= tool_limits else None, 
        "left": left, 
        "premium": session.get("is_premium", False)
    }

def increase_usage(email, tool, subtool):
    if email != "guest":
        users = load_premium_users()
        found = False
        for u in users:
            if u["email"].lower() == email.lower():
                if "usage_stats" not in u: u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                found = True
                break
        if found:
            save_premium_users(users)
            return

    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1
