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

# Progress store for vectorization
PROGRESS = {}

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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS packages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            display_name TEXT,
            description TEXT,
            price_monthly TEXT,
            price_quarterly TEXT,
            price_yearly TEXT
        )
    """)

    # Varsayılan paketleri ekle (tablo boşsa)
    cur.execute("SELECT COUNT(*) FROM packages")
    count = cur.fetchone()[0]
    if count == 0:
        default_pkgs = [
            (
                "basic",
                "Temel Paket",
                "Yeni başlayanlar için temel vektör paketi.",
                "199₺",
                "499₺",
                "1499₺",
            ),
            (
                "standard",
                "Orta Paket",
                "Düzenli tasarım ihtiyacı olanlar için.",
                "299₺",
                "799₺",
                "1999₺",
            ),
            (
                "premium",
                "Premium Paket",
                "Ajanslar ve yoğun kullanıcılar için gelişmiş paket.",
                "399₺",
                "999₺",
                "2499₺",
            ),
        ]
        cur.executemany(
            """
            INSERT INTO packages (slug, display_name, description, price_monthly, price_quarterly, price_yearly)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            default_pkgs,
        )

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
# PROGRESS HELPER
# ============================================================

def _progress_key():
    return session.get("email") or get_client_id()

def set_progress(value: int):
    try:
        key = _progress_key()
        PROGRESS[key] = int(value)
    except Exception:
        pass

def get_progress_value() -> int:
    try:
        key = _progress_key()
        return int(PROGRESS.get(key, 0))
    except Exception:
        return 0

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
            "end_date": user[2],
            "total_usage": user[3] or 0
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
            "end_date": user[2],
            "total_usage": user[3] or 0
        })

    client_id = get_client_id()
    usage = get_guest_usage(client_id)
    return jsonify({
        "logged_in": False,
        "premium": False,
        "guest_usage": usage,
        "remaining": max(0, FREE_LIMIT - usage)
    })

@app.route("/packages")
def get_packages():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT slug, display_name, description, price_monthly, price_quarterly, price_yearly FROM packages")
    rows = cur.fetchall()
    conn.close()
    packages = []
    for r in rows:
        packages.append({
            "slug": r[0],
            "display_name": r[1],
            "description": r[2],
            "price_monthly": r[3],
            "price_quarterly": r[4],
            "price_yearly": r[5],
        })
    return jsonify({"success": True, "packages": packages, "whatsapp": WHATSAPP})

@app.route("/vectorize_progress")
def vectorize_progress_status():
    return jsonify({"progress": get_progress_value()})

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
    email = data.get("email")
    password = data.get("password")

    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session["admin_logged_in"] = True
        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Yanlış email veya şifre"})

@app.route("/admin_logout", methods=["POST"])
@admin_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return jsonify({"success": True})

# ============================================================
# ADMIN PANEL SAYFASI
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
    cur.execute("SELECT email, start_date, end_date, total_usage FROM users")
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
    """Yeni premium kullanıcı ekler.
       Eğer kullanıcının aktif premiumu varsa hata döner."""
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

        # Yeni süreyi ayarla
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
    """Kullanıcı siler."""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()

        if not email:
            return jsonify({"success": False, "message": "Email alanı boş."}), 400

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email=?", (email,))
        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except Exception as e:
        print("admin_delete_user error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/admin_reset_usage", methods=["POST"])
@admin_required
def admin_reset_usage():
    """Kullanıcının toplam kullanımını sıfırlar."""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE users SET total_usage = 0 WHERE email=?", (email,))
        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except Exception as e:
        print("admin_reset_usage error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/admin_extend_user", methods=["POST"])
@admin_required
def admin_extend_user():
    """+30 gün ekler. Eğer süresi geçmişse bugünden; değilse mevcut bitişten itibaren."""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        extra_days = int(data.get("days") or 30)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT end_date FROM users WHERE email=?", (email,))
        row = cur.fetchone()

        if not row:
            conn.close()
            return jsonify({"success": False, "message": "Kullanıcı bulunamadı."}), 404

        now = datetime.now()
        current_end = datetime.fromisoformat(row[0])
        base = current_end if current_end > now else now
        new_end = base + timedelta(days=extra_days)

        cur.execute("UPDATE users SET end_date=? WHERE email=?", (new_end.isoformat(), email))
        conn.commit()
        conn.close()

        return jsonify({"success": True})
    except Exception as e:
        print("admin_extend_user error:", e)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/admin_packages")
@admin_required
def admin_get_packages():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, slug, display_name, description, price_monthly, price_quarterly, price_yearly FROM packages")
    rows = cur.fetchall()
    conn.close()
    packages = []
    for r in rows:
        packages.append({
            "id": r[0],
            "slug": r[1],
            "display_name": r[2],
            "description": r[3],
            "price_monthly": r[4],
            "price_quarterly": r[5],
            "price_yearly": r[6],
        })
    return jsonify({"success": True, "packages": packages})

@app.route("/admin_packages/save", methods=["POST"])
@admin_required
def admin_save_packages():
    data = request.get_json(silent=True) or {}
    packages = data.get("packages", [])
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        for pkg in packages:
            pkg_id = pkg.get("id")
            if not pkg_id:
                continue
            cur.execute(
                """
                UPDATE packages
                SET display_name=?, description=?, price_monthly=?, price_quarterly=?, price_yearly=?
                WHERE id=?
                """,
                (
                    pkg.get("display_name") or "",
                    pkg.get("description") or "",
                    pkg.get("price_monthly") or "",
                    pkg.get("price_quarterly") or "",
                    pkg.get("price_yearly") or "",
                    pkg_id,
                ),
            )
        conn.commit()
        return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        print("admin_save_packages error:", e)
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        conn.close()

# ============================================================
# VEKTÖRLEŞTİRME FONKSİYONLARI
# ============================================================

def remove_background(image_rgb: np.ndarray):
    """
    Daha sağlam arka plan kaldırma:
    - GrabCut ile ilk maske
    - Morfolojik temizlik
    - En büyük objeyi (köpek, yüz vs.) bırak
    - Hafif blur ile kenarları yumuşat
    Dönen:
        fg_rgb : arka planı siyaha çekilmiş RGB
        alpha  : 0–255 maske (0 = şeffaf, 255 = görünür)
    """
    h, w = image_rgb.shape[:2]

    # OpenCV BGR ister, o yüzden çeviriyoruz
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 1) GrabCut için başlangıç maskesi + dikdörtgen (kenarlardan %5 boşluk)
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        # arka plan / muhtemel arka plan → 0, diğerleri → 255
        mask = np.where(
            (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
            0,
            255
        ).astype("uint8")
    except Exception:
        # Hata olursa tüm resmi ön plan kabul et
        mask = np.full((h, w), 255, dtype="uint8")

    # 2) Maske temizle (delikleri kapat, ufak gürültüyü at)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3) En büyük objeyi bırak (köpek / yüz vs.)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # 0 index arka plan, 1..N objeler
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype("uint8")

    # 4) Kenarları yumuşat
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # 5) Maskeyi uygula – arka planı siyaha çek
    fg_rgb = image_rgb.copy()
    fg_rgb[mask == 0] = (0, 0, 0)

    alpha = mask  # 0–255
    return fg_rgb, alpha

def quantize_colors(image_rgb, k=4):
    """
    Renk sayısını azaltarak düz, poster tarzı görüntü üretir.
    'Normal vektör' stilinde kullanıyoruz.
    """
    k = max(2, min(12, int(k or 4)))  # mantıklı sınırlar
    img_color = cv2.medianBlur(image_rgb, 5)
    Z = np.float32(img_color.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    reduced = centers[labels.flatten()].reshape(img_color.shape)
    return reduced


def extract_edge_lines(image_rgb):
    """
    Beyaz arka plan üzerinde siyah çizgileri çıkarır.
    'Sadece çizgi' stilinde kullanıyoruz.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 80, 140)
    edges = cv2.dilate(edges, None, iterations=1)

    h, w = gray.shape
    line_img = np.full((h, w), 255, dtype=np.uint8)  # beyaz zemin
    line_img[edges > 0] = 0                          # kenarlar siyah
    line_rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    return line_rgb


def vector_normal_style(image_rgb, k=4):
    """
    Normal vektör: sadece renk sadeleştirme.
    (Düz renklere indirgenmiş, çizgisiz poster görünümü)
    """
    return quantize_colors(image_rgb, k)


def vector_cartoon_style(image_rgb, k=4):
    """
    Çizgi vektör: sade renk + siyah kontur.
    (Düz renk + kalın siyah çizgiler)
    """
    base = quantize_colors(image_rgb, k)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 80, 140)
    edges = cv2.dilate(edges, None, iterations=1)

    # Kenarları maskeye çevir
    edges = 255 - edges
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    cartoon = cv2.bitwise_and(base, edges_rgb)
    return cartoon


def vector_lines_style(image_rgb):
    """
    Sadece çizgi: renksiz line-art.
    (Beyaz zemin üzerinde siyah hatlar)
    """
    return extract_edge_lines(image_rgb)


def png_encode(arr, alpha=None):
    """
    PNG'yi base64 döner. alpha verilirse transparan arka planı korur.
    """
    if alpha is not None:
        # alpha 2D ise RGBA'ya dönüştür
        if alpha.ndim == 2:
            alpha_channel = alpha
        else:
            alpha_channel = cv2.cvtColor(alpha, cv2.COLOR_RGB2GRAY)

        if arr.shape[:2] != alpha_channel.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (arr.shape[1], arr.shape[0]))

        rgba = np.dstack([arr, alpha_channel])
        pil = Image.fromarray(rgba)
    else:
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

    # Frontend'den gelen parametreler
    style = request.form.get("style", "normal")   # "normal", "cartoon", "lines"
    colors = int(request.form.get("colors", 4) or 4)
    mode = request.form.get("mode", "color")      # "color" veya "bw"

    # Görseli oku
    img = Image.open(io.BytesIO(file.read()))
    arr = np.array(img.convert("RGB"))

    # 1) Önce arka planı kaldır
    arr_nb, alpha = remove_background(arr)

    # 2) Seçilen stile göre vektörleştir
    if style == "cartoon":            # Çizgi Vektör
        out = vector_cartoon_style(arr_nb, colors)
    elif style == "lines":            # Sadece Çizgi
        out = vector_lines_style(arr_nb)
    else:                             # Normal Vektör (varsayılan)
        out = vector_normal_style(arr_nb, colors)

    # 3) Siyah-beyaz isteniyorsa
    if mode == "bw":
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 4) Kullanım kaydı
    register_usage()

    # 5) PNG + alpha (şeffaf arka plan)
    return jsonify({
        "success": True,
        "image_data": png_encode(out, alpha)
    })


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




