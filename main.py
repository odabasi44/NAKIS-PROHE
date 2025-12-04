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
from functools import wraps # <<< EKLENDİ

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__, static_folder=".", static_url_path="")
# SECRET KEY: Oturum (session) verilerini şifrelemek için KRİTİK.
app.secret_key = "BOTLAB_SECRET_123" 
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024 # Max dosya boyutu 200MB

# ============================================================
# SETTINGS SYSTEM
# ============================================================
def load_settings():
    """settings.json dosyasını yükler."""
    if not os.path.exists("settings.json"):
        return {"admin": {"email": "", "password": ""}, "limits": {"pdf": {}, "image": {}, "vector": {}}, "site": {}}
    with open("settings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_settings(data):
    """settings.json dosyasını kaydeder."""
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# TICKET SYSTEM (Destek Talepleri)
# ============================================================
TICKET_FILE = "tickets.json"

def load_tickets():
    if not os.path.exists(TICKET_FILE):
        return []
    try:
        with open(TICKET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("HATA: tickets.json dosyası bozuk veya boş.")
        return []

def save_tickets(data):
    with open(TICKET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# PREMIUM USER SYSTEM
# ============================================================
PREMIUM_FILE = "users.json" # KULLANICI DOSYASI DOĞRU

def load_premium_users():
    """users.json dosyasını yükler."""
    if not os.path.exists(PREMIUM_FILE):
        return []
    try:
        with open(PREMIUM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("HATA: users.json dosyası bozuk veya boş.")
        return []

def save_premium_users(data):
    """users.json dosyasını kaydeder."""
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_user_data_by_email(email):
    """Veritabanından (users.json) kullanıcı verisini getirir."""
    users = load_premium_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None

# ============================================================
# SESSION KONTROLÜ (KRİTİK)
# ============================================================

@app.before_request
def check_session_status():
    """Tüm rotalar çalışmadan önce çalışır ve oturum durumunu kontrol eder."""

    # 1. Admin Paneli Kontrolü:
    if request.path.startswith("/admin") and request.path != "/admin_login":
        if not session.get("admin_logged"):
            return redirect("/admin_login")

    # 2. Premium Kullanıcı Kontrolü:
    if "user_email" in session and "is_premium" not in session:
        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                # 'end_date' alanı yoksa varsayılan tarih kullan
                end_date_str = user.get("end_date", "1970-01-01") 
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                if end_date >= datetime.now():
                    session["is_premium"] = True
                else:
                    session["is_premium"] = False 
             except ValueError:
                 session["is_premium"] = False # Tarih formatı hatalıysa
        else:
            session["is_premium"] = False
    
    # Tüm rotalar için is_premium varsayılan değeri
    if "is_premium" not in session:
        session["is_premium"] = False


# ============================================================
# PREMIUM / LIMIT CHECKER
# ============================================================
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()

    # PREMIUM mı?
    premium_user = None
    for u in users:
        if u.get("email", "").lower() == email.lower():
            end_date_str = u.get("end_date")
            if end_date_str:
                end = datetime.strptime(end_date_str, "%Y-%m-%d")
                if end >= datetime.now():
                    premium_user = u
                break
    
    # Araç limitlerini almak (varsayılan değerlerle güvenli)
    tool_limits = settings["limits"].get(tool, {})
    limits = tool_limits.get(subtool, {})

    # Tool'un bakımda olup olmadığını kontrol et
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": session.get("is_premium", False)}
    
    # PREMIUM KULLANICI KONTROLÜ
    if premium_user:
        if tool_status.get("premium_only", False) or limits.get("premium", 9999) > 0:
            premium_limit = limits.get("premium", 9999)
            tool_usage = premium_user.get("usage_stats", {}).get(subtool, 0)
            
            left = premium_limit - tool_usage
            
            if left <= 0:
                return {"allowed": False, "reason": "premium_limit_full", "left": 0, "premium": True}
            return {"allowed": True, "premium": True, "left": left}
        
    # FREE KULLANICI KONTROLÜ
    if tool_status.get("premium_only", False) and not premium_user:
         return {"allowed": False, "reason": "premium_only", "left": 0, "premium": False}

    free_limit = limits.get("free", 0)
    if "free_usage" not in session:
        session["free_usage"] = {}
    if tool not in session["free_usage"]:
        session["free_usage"][tool] = {}
    
    usage = session["free_usage"][tool].get(subtool, 0)
    left = free_limit - usage

    if left <= 0:
        return {"allowed": False, "reason": "free_limit_full", "left": 0, "premium": False}

    return {"allowed": True, "premium": False, "left": left}


def increase_usage(email, tool, subtool):
    """Kullanıcının kullanım sayısını artırır."""
    users = load_premium_users()

    # PREMIUM ise dosyada artar (Usage Stats)
    for u in users:
        if u.get("email", "").lower() == email.lower():
            if "usage_stats" not in u:
                u["usage_stats"] = {}
            
            current_usage = u["usage_stats"].get(subtool, 0)
            u["usage_stats"][subtool] = current_usage + 1
            u["usage"] = u.get("usage", 0) + 1 
            
            save_premium_users(users)
            return

    # FREE ise session içinde artar
    if "free_usage" not in session:
        session["free_usage"] = {}
    if tool not in session["free_usage"]:
        session["free_usage"][tool] = {}
        
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1


# ============================================================
# ADMIN LOGIN SYSTEM
# ============================================================

@app.route("/admin_login")
def admin_login_page():
    if session.get("admin_logged"):
        return redirect("/admin")
    return render_template("admin_login.html")


@app.route("/admin_login", methods=["POST"])
def admin_login_post():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    settings = load_settings()

    if email == settings["admin"]["email"] and password == settings["admin"]["password"]:
        session["admin_logged"] = True
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "error", "message": "Geçersiz giriş"}), 401


@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged", None)
    return redirect("/admin_login")


@app.route("/admin")
def admin_panel():
    return render_template("admin.html")


# ============================================================
# ADMIN API ROTLARI (USERS)
# ============================================================
@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    """Tüm Premium kullanıcı listesini döner."""
    if not session.get("admin_logged"):
        return jsonify({"status": "error", "message": "Yetki yok"}), 403

    users = load_premium_users()
    return jsonify(users)

@app.route("/api/admin/add_user", methods=["POST"])
def add_premium_user():
    if not session.get("admin_logged"):
        return jsonify({"status": "error", "message": "Yetki yok"}), 403
        
    data = request.get_json()
    email = data.get("email").lower()
    end_date_str = data.get("end_date") # YYYY-MM-DD formatında beklenir
    
    if not email or not end_date_str:
        return jsonify({"status": "error", "message": "E-posta veya bitiş tarihi eksik"}), 400

    users = load_premium_users()
    
    # Mevcut kullanıcı kontrolü
    if any(u.get("email", "").lower() == email for u in users):
        return jsonify({"status": "error", "message": "Bu e-posta zaten kayıtlı"}), 409

    # Yeni kullanıcı objesi
    new_user = {
        "email": email,
        "end_date": end_date_str,
        "usage": 0, 
        "is_admin": False,
        "usage_stats": {
            "remove_bg": 0,
            "pdf_merge": 0
        }
    }
    
    users.append(new_user)
    save_premium_users(users)
    return jsonify({"status": "ok", "message": "Kullanıcı başarıyla eklendi"})


@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def delete_premium_user(email):
    if not session.get("admin_logged"):
        return jsonify({"status": "error", "message": "Yetki yok"}), 403

    email_lower = email.lower()
    users = load_premium_users()
    
    original_len = len(users)
    users = [u for u in users if u.get("email", "").lower() != email_lower]
    
    if len(users) == original_len:
        return jsonify({"status": "error", "message": "Kullanıcı bulunamadı"}), 404

    save_premium_users(users)
    return jsonify({"status": "ok", "message": f"{email} silindi"})


# ============================================================
# PREMIUM DASHBOARD VE USER ROTASININ EKLENMESİ
# ============================================================

@app.route("/dashboard")
def dashboard_page():
    if not session.get("is_premium"):
        return redirect("/") 
        
    user_info = get_user_data_by_email(session.get("user_email")) or {}
    
    return render_template("dashboard.html", user=user_info)


@app.route("/user_login", methods=["POST"])
def user_login_endpoint():
    """Ana sayfadaki modal ile premium girişi yapar."""
    data = request.get_json()
    email = data.get("email")
    
    # 1. ADMIN kontrolü
    settings = load_settings()
    if email == settings["admin"]["email"]:
        return jsonify({"status": "admin"}) 

    # 2. PREMIUM Kullanıcı kontrolü
    user = get_user_data_by_email(email)
    
    if user:
        try:
            end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
            
            if end_date >= datetime.now():
                session["user_email"] = email
                session["is_premium"] = True
                return jsonify({"status": "premium", "name": user.get("name", "Kullanıcı")})
            else:
                return jsonify({"status": "expired"}) # Süresi dolmuş
        except ValueError:
            return jsonify({"status": "error", "message": "Tarih formatı hatalı"}), 500

    
    return jsonify({"status": "not_found"}) 


@app.route("/logout")
def user_logout():
    # session.pop ile tüm kullanıcı oturumlarını temizle
    session.pop("user_email", None)
    session.pop("is_premium", None)
    return redirect("/")


# ============================================================
# SETTINGS API (Admin paneli ayar API'ları)
# ============================================================
@app.route("/get_settings")
def api_get_settings():
    """Admin olmayan kullanıcılar için sadece site ayarlarını döner."""
    settings = load_settings()
    if session.get("admin_logged"):
        return jsonify(settings)
    
    # Sadece halka açık ayarları gönder
    return jsonify({
        "site": settings.get("site", {}), 
        "limits": settings.get("limits", {}), 
        "tool_status": settings.get("tool_status", {})
    })


@app.route("/api/admin/save_limits", methods=["POST"])
def save_tool_limits_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
        
    data = request.get_json()
    settings = load_settings()
    
    if "limits" in data:
        settings["limits"] = data["limits"]
    
    if "tool_status" in data:
         settings["tool_status"] = data["tool_status"]

    save_settings(settings)
    return jsonify({"status": "ok", "message": "Limitler ve durumlar güncellendi."})


@app.route("/api/admin/save_site_settings", methods=["POST"])
def save_global_settings():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
        
    data = request.get_json()
    settings = load_settings()

    settings["site"]["title"] = data.get("title", settings["site"].get("title", ""))
    settings["site"]["whatsapp_number"] = data.get("whatsapp_number", settings["site"].get("whatsapp_number", ""))
    settings["site"]["announcement_bar"] = data.get("announcement_bar", settings["site"].get("announcement_bar", ""))

    save_settings(settings)
    return jsonify({"status": "ok", "message": "Site ayarları güncellendi."})


@app.route("/api/admin/save_admin_credentials", methods=["POST"])
def save_admin_credentials():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
        
    data = request.get_json()
    settings = load_settings()

    settings["admin"]["email"] = data.get("email", settings["admin"].get("email", ""))
    if data.get("password"):
        settings["admin"]["password"] = data["password"]

    save_settings(settings)
    return jsonify({"status": "ok", "message": "Admin bilgileri güncellendi."})


# ============================================================
# GENEL KULLANIM ROTLARI (Aşama 4 için eklendi)
# ============================================================

@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    """
    Frontend'in anlık olarak kullanıcının limit durumunu kontrol etmesini sağlar.
    """
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    
    # Kullanım istatistiklerini de ekleyelim
    user = get_user_data_by_email(email)
    
    # Geçmiş kullanımı çek
    usage = user.get("usage_stats", {}).get(subtool, 0) if user else 0
    
    return jsonify({
        "allowed": status["allowed"],
        "reason": status["reason"],
        "left": status["left"],
        "premium": status["premium"],
        "usage": usage
    })


# ============================================================
# PDF MERGE API (Limit kontrolü eklendi)
# ============================================================
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "pdf", "merge")
    
    if not status["allowed"]:
        return jsonify({
            "success": False,
            "reason": status["reason"],
            "left": status["left"],
            "premium": status["premium"]
        }), 403

    if "pdf_files" not in request.files:
        return jsonify({"success": False, "message": "PDF seçilmedi"}), 400

    files = request.files.getlist("pdf_files")

    if len(files) < 2:
        return jsonify({"success": False, "message": "En az 2 PDF gerekli"}), 400

    merger = PdfMerger()
    for f in files:
        f.seek(0)
        merger.append(io.BytesIO(f.read()))

    output = io.BytesIO()
    merger.write(output)
    output.seek(0)
    merger.close()

    increase_usage(email, "pdf", "merge")

    pdf_base64 = base64.b64encode(output.read()).decode("utf-8")

    return jsonify({
        "success": True,
        "file": pdf_base64,
        "filename": "birlesik.pdf"
    })


# ============================================================
# BACKGROUND REMOVER (ONNX) (Limit kontrolü eklendi)
# ============================================================
print("Loading U2Net ONNX model...")
U2NET_MODEL = "/app/models/u2net.onnx"

try:
    u2net_session = ort.InferenceSession(
        U2NET_MODEL,
        providers=["CPUExecutionProvider"]
    )
except ort.OnnxRuntimeError as e:
    print(f"UYARI: ONNX model yüklenemedi: {e}")
    u2net_session = None 


def preprocess_bg(img):
    img = img.convert("RGB")
    img = img.resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr.reshape(1, 3, 320, 320)

def postprocess_bg(mask, size):
    mask = mask.squeeze()
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    
    if not u2net_session:
        return jsonify({"success": False, "message": "Model yüklenemedi. Bakım modunda."}), 503

    email = session.get("user_email", "guest")

    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]:
        return jsonify({
            "success": False,
            "reason": status["reason"],
            "left": status["left"],
            "premium": status["premium"]
        }), 403

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Resim seçilmedi"}), 400

    file = request.files["image"]
    img = Image.open(file.stream)

    ow, oh = img.size

    inp = preprocess_bg(img)
    output = u2net_session.run(None, {"input": inp})[0]

    mask = postprocess_bg(output, (ow, oh))

    rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)

    out = Image.fromarray(rgba)
    buf = io.BytesIO()
    out.save(buf, format="PNG")

    encoded = base64.b64encode(buf.getvalue()).decode()

    increase_usage(email, "image", "remove_bg")

    return jsonify({
        "success": True,
        "file": encoded,
        "filename": "arka_plan_silindi.png"
    })


# ============================================================
# VECTOR API (Şimdilik Fake)
# ============================================================
@app.route("/vectorize_style", methods=["POST"])
def vectorize_with_style():
    return jsonify({
        "success": True,
        "image_data": base64.b64encode(b"FAKE_VECTOR").decode()
    })


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    settings = load_settings()
    return render_template("index.html", settings=settings)

@app.route("/vektor")
def vektor_page():
    return render_template("vektor.html")

@app.route("/pdf/merge")
def pdf_merge_page():
    return render_template("pdf_merge.html")

@app.route("/remove-bg")
def remove_bg_page():
    return render_template("background_remove.html")


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("BOTLAB SUITE BACKEND STARTING...")
    app.run(host="0.0.0.0", port=port, debug=True)
