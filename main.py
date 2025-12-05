import os
import io
import json
import base64
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from PyPDF2 import PdfMerger
from flask import Flask, request, jsonify, render_template, session, redirect, send_file
from flask_cors import CORS
import onnxruntime as ort

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ============================================================
# SETTINGS SYSTEM
# ============================================================
def load_settings():
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ============================================================
# PREMIUM USER SYSTEM & TIERS
# ============================================================
PREMIUM_FILE = "premium_users.json"

# Varsayılan Paket Yetkileri (Hangi paket neye erişemez)
# Bu listede olanlar o pakette "KİLİTLİ" görünür.
TIER_RESTRICTIONS = {
    "free": ["pdf_merge", "pdf_split", "vector", "word2pdf", "resize"],
    "starter": ["vector", "pdf_split", "word2pdf"], # Başlangıç: Vektör ve İleri PDF yok
    "pro": [],      # Pro: Her şey açık (belki API kapalı olabilir)
    "unlimited": [] # Sınırsız: Her şey açık
}

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE):
        return []
    with open(PREMIUM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_premium_users(data):
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ============================================================
# USER LOGIN (YENİLENMİŞ)
# ============================================================
@app.route("/user_login", methods=["POST"])
def user_login():
    data = request.get_json()
    email = data.get("email", "").strip()
    
    # 1. Admin Kontrolü
    settings = load_settings()
    if settings.get("admin") and email == settings["admin"].get("email"):
        session["user_email"] = email
        session["user_role"] = "admin"
        session["user_tier"] = "unlimited" # Admin her yeri görür
        return jsonify({"status": "admin"})

    # 2. Premium Kullanıcı Kontrolü
    users = load_premium_users()
    user = next((u for u in users if u["email"] == email), None)

    if user:
        # Süre kontrolü
        end_date = datetime.strptime(user["end"], "%Y-%m-%d")
        if end_date >= datetime.now():
            session["user_email"] = email
            session["user_role"] = "premium"
            # Kullanıcının paketi yoksa varsayılan 'starter' olsun
            session["user_tier"] = user.get("tier", "starter") 
            return jsonify({"status": "premium", "tier": session["user_tier"]})
    
    # Kullanıcı yoksa veya süresi dolmuşsa
    return jsonify({"status": "error", "message": "Kullanıcı bulunamadı veya süresi dolmuş."}), 401

@app.route("/user_logout")
def user_logout():
    session.clear()
    return redirect("/")

# ============================================================
# PREMIUM / LIMIT CHECKER
# ============================================================
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()

    # Kullanıcıyı bul
    premium_user = next((u for u in users if u["email"] == email), None)
    
    # Paket Kontrolü (Tier Check)
    user_tier = "free"
    if premium_user:
        end_date = datetime.strptime(premium_user["end"], "%Y-%m-%d")
        if end_date >= datetime.now():
            user_tier = premium_user.get("tier", "starter")

    # Eğer araç bu pakette kısıtlıysa engelle
    # tool_key örnek: 'vector' veya 'pdf_merge'
    tool_key = subtool if subtool else tool 
    if tool_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier}

    # Limit Ayarları
    if tool == "pdf":
        limits = settings["limits"]["pdf"][subtool]
    elif tool == "image":
        limits = settings["limits"]["image"][subtool]
    elif tool == "vector":
        limits = settings["limits"]["vector"]
    else:
        return {"allowed": False, "reason": "invalid_tool"}

    # PREMIUM LİMİT KONTROLÜ
    if premium_user and user_tier != "free":
        left = limits["premium"] - premium_user["usage"]
        # Unlimited pakette limit düşmez veya çok yüksektir, basitlik için premium limiti kullanıyoruz
        if user_tier == "unlimited":
            return {"allowed": True, "premium": True, "tier": user_tier, "left": 9999}
            
        if left <= 0:
            return {"allowed": False, "reason": "premium_limit_full", "left": 0}
        return {"allowed": True, "premium": True, "tier": user_tier, "left": left}

    # FREE LİMİT KONTROLÜ
    if "free_usage" not in session:
        session["free_usage"] = {}
    if tool not in session["free_usage"]:
        session["free_usage"][tool] = {}
    if subtool not in session["free_usage"][tool]:
        session["free_usage"][tool][subtool] = 0

    usage = session["free_usage"][tool][subtool]
    left = limits["free"] - usage

    if left <= 0:
        return {"allowed": False, "reason": "free_limit_full", "left": 0}

    return {"allowed": True, "premium": False, "tier": "free", "left": left}


def increase_usage(email, tool, subtool):
    users = load_premium_users()
    for u in users:
        if u["email"] == email:
            u["usage"] += 1
            save_premium_users(users)
            return
    session["free_usage"][tool][subtool] += 1

# ============================================================
# API ROUTES (PDF, IMG, ETC.)
# ============================================================
# ... (PDF Merge, Remove BG ve Vector kodları önceki gibi buraya gelecek) ...
# ... Kod tasarrufu için burayı kısa tutuyorum, önceki main.py içeriği ile aynı ...
# ... Sadece check_user_status fonksiyonu güncellendi ...

# (ÖNEMLİ: Bu kısım önceki main.py'deki API fonksiyonlarının aynısı olmalı, 
# sadece check_user_status çağrısı yeni mantığı kullanacak)

@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    email = session.get("user_email", "guest") # Session'dan alıyoruz artık
    status = check_user_status(email, "image", "remove_bg")
    
    if not status["allowed"]:
        return jsonify({"success": False, "reason": status["reason"]}), 403

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Resim yok"}), 400

    # ... İşlem kodları ...
    # Fake response for demo context:
    return jsonify({"success": True, "file": "base64...", "filename": "demo.png"})

@app.route("/api/check_tool_status/<tool>/<subtool>")
def api_check_tool(tool, subtool):
    email = session.get("user_email", "guest")
    return jsonify(check_user_status(email, tool, subtool))

# ============================================================
# PAGE ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin_login")
def admin_login_page():
    return render_template("admin_login.html")

@app.route("/admin")
def admin_panel():
    if session.get("user_role") != "admin":
        return redirect("/admin_login")
    return render_template("admin.html")

@app.route("/remove-bg")
def remove_bg_page():
    return render_template("background_remove.html")

@app.route("/pdf/merge")
def pdf_merge_page():
    return render_template("pdf_merge.html")

@app.route("/vektor")
def vektor_page():
    return render_template("vector.html")

# ... Diğer route'lar ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
