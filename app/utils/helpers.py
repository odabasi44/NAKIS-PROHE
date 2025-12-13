import os
import json
from datetime import datetime
from flask import session
from app.models import User
from app.extensions import db
from sqlalchemy import func
from sqlalchemy.exc import OperationalError

SETTINGS_FILE = "settings.json"

# Kullanıcılar artık veritabanında olduğu için PREMIUM_FILE gerek yok.
# Sadece genel site ayarları (limitler vb.) için settings.json kullanıyoruz.

def load_settings():
    """
    Öncelik: DB (AppSetting key='settings') -> settings.json -> {}.
    Not: Coolify gibi ortamlarda container filesystem ephemeral olabildiği için DB tercih edilir.
    """
    try:
        from app.models import AppSetting
        row = AppSetting.query.filter_by(key="settings").first()
        if row:
            return row.get_value()
    except OperationalError:
        # DB henüz hazır değilse
        pass
    except Exception:
        pass

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(settings_dict):
    """Ayarları DB’ye kaydeder; ayrıca local geliştirme için settings.json da güncellenir."""
    settings_dict = settings_dict or {}
    try:
        from app.models import AppSetting
        row = AppSetting.query.filter_by(key="settings").first()
        if not row:
            row = AppSetting(key="settings")
            db.session.add(row)
        row.set_value(settings_dict)
        db.session.commit()
    except Exception:
        db.session.rollback()

    # Local fallback file (deploy'da persist etmeyebilir ama zararı yok)
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

# --- YENİ VERİTABANI FONKSİYONLARI ---

def get_user_data_by_email(email):
    """Veritabanından kullanıcıyı çeker."""
    if not email:
        return None

    email_norm = str(email).strip().lower()

    # Büyük/küçük harf duyarsız arama
    user = User.query.filter(func.lower(User.email) == email_norm).first()
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
    # Admin oturumu: sınırsız say
    if session.get("admin_logged"):
        return {
            "allowed": True,
            "reason": None,
            "left": 99999,
            "limit": 99999,
            "tier": "unlimited",
            "premium": True
        }

    user_tier = "free"
    current_usage = 0
    email_norm = "guest"
    if email and email != "guest":
        email_norm = str(email).strip().lower()
    
    # 1. Kayıtlı Kullanıcı Kontrolü
    if email_norm != "guest":
        user = User.query.filter(func.lower(User.email) == email_norm).first()
        if user:
            try:
                # Üyelik süresi dolmuş mu?
                if user.end_date and user.end_date >= datetime.now().date():
                    user_tier = user.tier
                    current_usage = user.get_usage().get(subtool, 0)
                else:
                    # Süresi dolmuşsa free gibi davran ama istatistiğini çek
                    user_tier = "free" 
                    # Süresi dolan kullanıcıyı session tarafında "free" olarak değerlendir.
                    session["is_premium"] = False
                    session["user_tier"] = "free"
            except:
                pass
        else:
            # DB'de user yok ama session premium olabilir (ör: admin panel login ile)
            if session.get("is_premium") and session.get("user_tier") in ("starter", "pro", "unlimited"):
                user_tier = session.get("user_tier")

    # 2. Misafir veya Süresi Dolmuş Kullanıcı (Session Kullanır)
    if user_tier == "free":
        if "free_usage" not in session: session["free_usage"] = {}
        if tool not in session["free_usage"]: session["free_usage"][tool] = {}
        current_usage = session["free_usage"][tool].get(subtool, 0)

    # 3. Limitleri Ayarlardan Çek
    try:
        # settings.json yapısına göre: limits -> image -> remove_bg -> free
        tool_limits = settings["limits"][tool][subtool][user_tier]
    except:
        tool_limits = 5 # Ayar bulunamazsa varsayılan limit

    left = max(0, tool_limits - current_usage)
    
    return {
        "allowed": current_usage < tool_limits,
        "reason": "limit" if current_usage >= tool_limits else None,
        "left": left,
        "limit": tool_limits,
        "tier": user_tier,
        "premium": session.get("is_premium", False)
    }

def increase_usage(email, tool, subtool):
    """Kullanım sayısını artırır."""
    email_norm = "guest"
    if email and email != "guest":
        email_norm = str(email).strip().lower()
    
    # 1. Kayıtlı Kullanıcı
    if email_norm != "guest":
        user = User.query.filter(func.lower(User.email) == email_norm).first()
        if user:
            user.increase_usage(subtool)
            try:
                db.session.commit() # Değişikliği veritabanına kaydet
            except Exception:
                db.session.rollback()
            # Usage event log (admin raporu / son işlemler)
            try:
                from app.models import UsageEvent
                db.session.add(UsageEvent(user_email=email_norm, tool=tool, subtool=subtool))
                db.session.commit()
            except Exception:
                db.session.rollback()
            return

    # 2. Misafir Kullanıcı (Session)
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    
    current = session["free_usage"][tool].get(subtool, 0)
    session["free_usage"][tool][subtool] = current + 1
    session.modified = True # Flask session'ın güncellendiğini anlasın

    # Guest event (ops sayısı için opsiyonel; burada kaydetmiyoruz)
