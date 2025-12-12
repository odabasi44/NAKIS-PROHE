import os
import json
from datetime import datetime
from flask import session
from app.models import User
from app.extensions import db

SETTINGS_FILE = "settings.json"

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_settings_to_file(data):
    """Ayarları settings.json dosyasına kaydeder."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ayarlar kaydedilemedi: {e}")
        return False

# --- YENİ VERİTABANI FONKSİYONLARI ---

def get_user_data_by_email(email):
    """Veritabanından kullanıcıyı çeker."""
    user = User.query.filter_by(email=email).first()
    if user:
        return {
            "email": user.email,
            "tier": user.tier,
            "end_date": user.end_date.strftime("%Y-%m-%d") if user.end_date else None,
            "usage_stats": user.get_usage()
        }
    return None

def check_user_status(email, tool, subtool):
    """Kullanıcının limitini kontrol eder."""
    settings = load_settings()
    user_tier = "free"
    current_usage = 0
    
    # 1. Kayıtlı Kullanıcı Kontrolü
    if email != "guest":
        user = User.query.filter_by(email=email).first()
        if user:
            try:
                # Üyelik süresi dolmuş mu?
                if user.end_date and user.end_date >= datetime.now().date():
                    user_tier = user.tier
                    current_usage = user.get_usage().get(subtool, 0)
                else:
                    user_tier = "free"
            except:
                pass

    # 2. Misafir veya Süresi Dolmuş Kullanıcı
    if user_tier == "free":
        if "free_usage" not in session: session["free_usage"] = {}
        if tool not in session["free_usage"]: session["free_usage"][tool] = {}
        current_usage = session["free_usage"][tool].get(subtool, 0)

    # 3. Limitleri Ayarlardan Çek
    try:
        # settings.json: limits -> vector -> default -> free
        tool_limits = settings["limits"][tool][subtool][user_tier]
    except:
        tool_limits = 5 # Varsayılan limit

    # Admin ise limit yok
    if session.get("user_tier") == "unlimited" or user_tier == "unlimited":
        tool_limits = 999999

    left = max(0, tool_limits - current_usage)
    
    return {
        "allowed": current_usage < tool_limits,
        "reason": "limit" if current_usage >= tool_limits else None,
        "left": left,
        "premium": session.get("is_premium", False)
    }

def increase_usage(email, tool, subtool):
    """Kullanım sayısını artırır."""
    
    # 1. Kayıtlı Kullanıcı
    if email != "guest":
        user = User.query.filter_by(email=email).first()
        if user:
            user.increase_usage(subtool)
            db.session.commit()
            return

    # 2. Misafir Kullanıcı
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    
    current = session["free_usage"][tool].get(subtool, 0)
    session["free_usage"][tool][subtool] = current + 1
    session.modified = True
