import os
import json
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_cors import CORS

import cv2
import numpy as np
from PIL import Image
import io
import base64

# ============================================================
# FLASK BAŞLANGIÇ
# ============================================================

app = Flask(__name__, static_folder="static")
app.secret_key = "SUPER_SECRET_KEY_123"
CORS(app)

DB_PATH = "database.db"

# ============================================================
# SETTINGS.JSON OKU (Admin Bilgileri ve parametreler)
# ============================================================

with open("settings.json", "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

FREE_LIMIT = SETTINGS["free_usage_limit"]
PREMIUM_DAYS = SETTINGS["premium_duration_days"]
ADMIN_EMAIL = SETTINGS["admin_email"]
ADMIN_PASSWORD = SETTINGS["admin_password"]
WHATSAPP = SETTINGS["whatsapp_number"]

# ============================================================
# DATABASE OLUŞTUR
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            total_usage INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS guests (
            client_id TEXT PRIMARY KEY,
            usage_count INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ============================================================
# DB YARDIMCILAR
# ============================================================

def get_client_id():
    ip = request.remote_addr or "0.0.0.0"
    agent = request.headers.get("User-Agent", "")
    return f"{ip}_{hash(agent)}"

def get_user(email):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT email, start_date, end_date, total_usage FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    return row

def add_user(email):
    now = datetime.now()
    end = now + timedelta(days=PREMIUM_DAYS)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO users (email, start_date, end_date, total_usage) VALUES (?, ?, ?, ?)",
        (email, now.isoformat(), end.isoformat(), 0))
    conn.commit()
    conn.close()

def increment_user_usage(email):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET total_usage = total_usage + 1 WHERE email=?", (email,))
    conn.commit()
    conn.close()

def get_guest_usage(client_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT usage_count FROM guests WHERE client_id=?", (client_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_guest_usage(client_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO guests (client_id, usage_count)
        VALUES (?, 1)
        ON CONFLICT(client_id) DO UPDATE SET usage_count = usage_count + 1
    """, (client_id,))
    conn.commit()
    conn.close()

def is_user_premium(email):
    user = get_user(email)
    if not user:
        return False
    end_date = datetime.fromisoformat(user[2])
    return end_date > datetime.now()

# ============================================================
# GİRİŞ / ÇIKIŞ / STATUS API
# ============================================================

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    user = get_user(email)

    if user and is_user_premium(email):
        session["email"] = email
        return jsonify({
            "success": True,
            "premium": True,
            "end_date": user[2]
        })

    return jsonify({"success": False, "message": "Bu mail için premium üyelik yok."})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/user_status")
def user_status():
    if "email" in session:
        email = session["email"]
        premium = is_user_premium(email)
        user = get_user(email)

        return jsonify({
            "logged_in": True,
            "email": email,
            "premium": premium,
            "end_date": user[2]
        })

    client_id = get_client_id()
    usage = get_guest_usage(client_id)
    return jsonify({
        "logged_in": False,
        "premium": False,
        "guest_usage": usage,
        "remaining": max(0, FREE_LIMIT - usage)
    })

# ============================================================
# KULLANIM İZNİ VE KULLANIM KAYDI
# ============================================================

def check_usage_permission():
    # Premium kullanıcı
    if "email" in session:
        email = session["email"]
        if is_user_premium(email):
            return {"allowed": True, "premium": True}

        # Premium bitmiş → guest olarak devam
        client_id = get_client_id()
        usage = get_guest_usage(client_id)
        if usage < FREE_LIMIT:
            return {"allowed": True, "premium": False, "remaining": FREE_LIMIT - usage}
        return {"allowed": False, "premium": False, "remaining": 0}

    # Guest kullanıcı
    client_id = get_client_id()
    usage = get_guest_usage(client_id)
    if usage < FREE_LIMIT:
        return {"allowed": True, "premium": False, "remaining": FREE_LIMIT - usage}

    return {"allowed": False, "premium": False, "remaining": 0}

def register_usage():
    if "email" in session and is_user_premium(session["email"]):
        increment_user_usage(session["email"])
    else:
        client_id = get_client_id()
        increment_guest_usage(client_id)

# ============================================================
# ADMIN DECORATOR + ADMIN GİRİŞ
# ============================================================

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return jsonify({"success": False, "error": "not_admin"}), 403
        return f(*args, **kwargs)
    return wrapper

@app.route("/admin_login", methods=["POST"])
def admin_login():
    data = request.get_json()
    if data.get("email") == ADMIN_EMAIL and data.get("password") == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Yanlış email veya şifre"})

@app.route("/admin_logout", methods=["POST"])
@admin_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return jsonify({"success": True})

# ============================================================
# ADMIN PANEL SERVE
# ============================================================

@app.route("/control_panel")
def control_panel():
    if not session.get("admin_logged_in"):
        return send_from_directory("templates", "admin_login.html")
    return send_from_directory("templates", "admin.html")

# ============================================================
# ADMIN API'LERİ
# ============================================================

@app.route("/admin_users")
@admin_required
def admin_users():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT email,start_date,end_date,total_usage FROM users")
    rows = cur.fetchall()
    conn.close()

    users = []
    for r in rows:
        users.append({
            "email": r[0],
            "start_date": r[1],
            "end_date": r[2],
            "total_usage": r[3]
        })

    return jsonify({"success": True, "users": users})

@app.route("/admin_add_user", methods=["POST"])
@admin_required
def admin_add_user():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        days = int(data.get("days") or SETTINGS["premium_duration_days"])

        if not email:
            return jsonify({"success": False, "message": "Email alanı boş olamaz."}), 400

        now = datetime.now()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Mevcut kullanıcıyı kontrol et
        cur.execute("SELECT end_date FROM users WHERE email=?", (email,))
        row = cur.fetchone()

        if row:
            end_date = datetime.fromisoformat(row[0])
            if end_date > now:
                conn.close()
                return jsonify({
                    "success": False,
                    "message": "Bu kullanıcının zaten aktif bir premium üyeliği var."
                }), 400

        # Aktif değilse (yoksa veya süresi bitmişse) yeni süre tanımla
        new_end = now + timedelta(days=days)

        cur.execute("""
            INSERT OR REPLACE INTO users (email, start_date, end_date, total_usage)
            VALUES (?, ?, ?, COALESCE((SELECT total_usage FROM users WHERE email=?), 0))
        """, (email, now.isoformat(), new_end.isoformat(), email))
        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except Exception as e:
        print("admin_add_user error:", e)
        return jsonify({"success": False, "message": str(e)}), 500



@app.route("/admin_delete_user", methods=["POST"])
@admin_required
def admin_delete_user():
    email = request.json.get("email")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE email=?", (email,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/admin_reset_usage", methods=["POST"])
@admin_required
def admin_reset_usage():
    email = request.json.get("email")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE users SET total_usage=0 WHERE email=?", (email,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route("/admin_extend_user", methods=["POST"])
@admin_required
def admin_extend_user():
    email = request.json.get("email")
    days = int(request.json.get("days", 30))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT end_date FROM users WHERE email=?", (email,))
    row = cur.fetchone()

    current_end = datetime.fromisoformat(row[0])
    new_end = current_end + timedelta(days=days)

    cur.execute("UPDATE users SET end_date=? WHERE email=?", (new_end.isoformat(), email))
    conn.commit()
    conn.close()

    return jsonify({"success": True})

# ============================================================
# VEKTÖRLEŞTİRME FONKSİYONLARI
# ============================================================

def cartoon_vectorize(image_rgb, k=4):
    img_color = cv2.medianBlur(image_rgb, 7)
    Z = np.float32(img_color.reshape((-1, 3)))

    _, labels, centers = cv2.kmeans(
        Z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape(img_color.shape)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 140)
    edges = cv2.dilate(edges, None)
    edges = 255 - edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    cartoon = cv2.bitwise_and(reduced, edges_colored)
    return cartoon

def png_encode(arr):
    pil = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# ============================================================
# VEKTÖR API
# ============================================================

@app.route("/vectorize_style", methods=["POST"])
def vectorize_style():
    # Kullanım limiti kontrolü
    permission = check_usage_permission()
    if not permission["allowed"]:
        return jsonify({
            "success": False,
            "error": "limit_reached",
            "message": "Ücretsiz hakkınız doldu.",
            "whatsapp": f"https://wa.me/{WHATSAPP}?text=Aylık+üyelik+istiyorum"
        }), 403

    file = request.files["image"]
    style = request.form.get("style", "cartoon")
    colors = int(request.form.get("colors", 4))
    mode = request.form.get("mode", "color")  # "color" veya "bw"

    img = Image.open(io.BytesIO(file.read()))
    arr = np.array(img.convert("RGB"))

    # Şimdilik ana stilimiz cartoon
    if style == "cartoon":
        out = cartoon_vectorize(arr, k=colors)
    else:
        out = arr

    # Siyah beyaz isteniyorsa:
    if mode == "bw":
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Kullanım hakkını kaydet
    register_usage()

    return jsonify({"success": True, "image_data": png_encode(out)})

# ============================================================
# ANA SAYFA
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")

# ============================================================
# ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)



