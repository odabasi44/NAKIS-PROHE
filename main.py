import os
import io
import json
import base64
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from PyPDF2 import PdfMerger
from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# --- AYARLAR ---
def load_settings():
    default_settings = {
        "admin": {"email": "admin@botlab.com", "password": "admin"},
        "limits": {
            "pdf": {"merge": {"free": 3, "premium": 100}},
            "image": {"remove_bg": {"free": 3, "premium": 1000}},
            "vector": {"default": {"free": 1, "premium": 50}}
        },
        "site": {},
        "tool_status": {}
    }
    
    if not os.path.exists("settings.json"):
        return default_settings
        
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if "limits" not in data: data["limits"] = default_settings["limits"]
            return data
    except:
        return default_settings

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- KULLANICI SİSTEMİ ---
PREMIUM_FILE = "users.json"

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE): return []
    try:
        with open(PREMIUM_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_premium_users(data):
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_user_data_by_email(email):
    users = load_premium_users()
    for u in users:
        if u.get("email", "").lower() == email.lower(): return u
    return None

# --- OTURUM KONTROLÜ ---
@app.before_request
def check_session_status():
    if request.path.startswith("/admin") and request.path != "/admin_login":
        if not session.get("admin_logged"): return redirect("/admin_login")

    # Kullanıcı oturumunu kontrol et, süresi bittiyse düşür
    if "user_email" in session:
        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
                session["is_premium"] = (end_date >= datetime.now())
             except: session["is_premium"] = False
        else: 
            session.pop("user_email", None)
            session["is_premium"] = False
    
    if "is_premium" not in session: session["is_premium"] = False

# --- LİMİT KONTROL ---
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()
    premium_user = None
    
    # Email "guest" değilse ve veritabanında varsa Premium kullanıcısıdır
    if email != "guest":
        for u in users:
            if u.get("email", "").lower() == email.lower():
                end_date_str = u.get("end_date")
                try:
                    if end_date_str and datetime.strptime(end_date_str, "%Y-%m-%d") >= datetime.now():
                        premium_user = u
                except: pass
                break
    
    tool_limits = settings.get("limits", {}).get(tool, {})
    limits = tool_limits.get(subtool, {"free": 3, "premium": 1000})
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    
    # Bakım Modu
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": False}
    
    # Premium Kullanıcı Mantığı
    if premium_user:
        premium_limit = limits.get("premium", 9999)
        tool_usage = premium_user.get("usage_stats", {}).get(subtool, 0)
        left = premium_limit - tool_usage
        if left <= 0:
            return {"allowed": False, "reason": "premium_limit_full", "left": 0, "premium": True}
        return {"allowed": True, "reason": "", "premium": True, "left": left}
        
    # Sadece Premium Araç
    if tool_status.get("premium_only", False):
         return {"allowed": False, "reason": "premium_only", "left": 0, "premium": False}

    # Ücretsiz (Misafir) Limit Mantığı
    free_limit = limits.get("free", 3)
    
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    
    usage = session["free_usage"][tool].get(subtool, 0)
    left = free_limit - usage
    
    if left <= 0:
        return {"allowed": False, "reason": "free_limit_full", "left": 0, "premium": False}

    return {"allowed": True, "reason": "", "premium": False, "left": left}

def increase_usage(email, tool, subtool):
    users = load_premium_users()
    
    # Email guest değilse DB'ye yaz
    if email != "guest":
        for u in users:
            if u.get("email", "").lower() == email.lower():
                if "usage_stats" not in u: u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                u["usage"] = u.get("usage", 0) + 1
                save_premium_users(users)
                return

    # Değilse Session'a yaz
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1

# --- API ENDPOINTLERİ ---

# [ÖNEMLİ] Frontend bu endpointi kullanarak limiti sorar
@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    # Eğer session'da email varsa onu kullan, yoksa 'guest'
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    user = get_user_data_by_email(email)
    usage = user.get("usage_stats", {}).get(subtool, 0) if user else 0
    
    return jsonify({
        "allowed": status.get("allowed", False),
        "reason": status.get("reason", ""),
        "left": status.get("left", 0),
        "premium": status.get("premium", False),
        "usage": usage
    })

# --- REMOVE BG (AKILLI MODEL YÜKLEYİCİ) ---
u2net_session = None
model_input_name = "input" # Varsayılan, aşağıda otomatik güncellenir

print("--- MODEL YÜKLEME BAŞLIYOR ---")
possible_paths = [
    "/data/ai-models/u2net.onnx", 
    "u2net.onnx",
    "models/u2net.onnx",
    "/app/models/u2net.onnx",
    "/app/u2net.onnx"
]

found_path = None
for path in possible_paths:
    if os.path.exists(path):
        found_path = path
        print(f"✅ Model bulundu: {path}")
        break

if found_path:
    try:
        u2net_session = ort.InferenceSession(found_path, providers=["CPUExecutionProvider"])
        # [ÖNEMLİ] Modelin giriş ismini (input.1 veya input) otomatik bul
        model_input_name = u2net_session.get_inputs()[0].name
        print(f"🚀 ONNX Modeli Yüklendi! Giriş Parametresi: {model_input_name}")
    except Exception as e:
        print(f"❌ Model Yükleme Hatası: {e}")
else:
    print("⚠️ UYARI: u2net.onnx modeli bulunamadı.")

def preprocess_bg(img):
    img = img.convert("RGB").resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr.reshape(1, 3, 320, 320)

def postprocess_bg(mask, size):
    mask = mask.squeeze()
    mask = cv2.resize(mask, size)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask

@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    if not u2net_session: 
        return jsonify({"success": False, "reason": "AI Modeli Sunucuda Bulunamadı."}), 503
        
    # [ÖNEMLİ] Kullanıcıyı session'dan alıyoruz, formdan gelen veriye güvenmiyoruz.
    email = session.get("user_email", "guest")
    is_premium = session.get("is_premium", False)
    
    # 1. Limit Kontrolü
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]: 
        return jsonify(status), 403

    if "image" not in request.files: return jsonify({"success": False, "message": "Resim yok"}), 400
    
    file = request.files["image"]
    
    # 2. Dosya Boyutu Kontrolü
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    # Misafir: 5MB, Premium: 50MB
    limit_mb = 50 if is_premium else 5
    if file_size > limit_mb * 1024 * 1024:
        return jsonify({
            "success": False, 
            "reason": "file_size_limit", 
            "message": f"Dosya boyutu çok büyük! {'Misafir limiti: 5MB' if not is_premium else 'Premium limiti: 50MB'}"
        }), 413
    
    try:
        img = Image.open(file.stream)
        ow, oh = img.size
        
        # [ÖNEMLİ] Dinamik input ismini kullanıyoruz (hata çözümü)
        output = u2net_session.run(None, {model_input_name: preprocess_bg(img)})[0]
        mask = postprocess_bg(output, (ow, oh))
        
        rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        
        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        print(f"Processing Error: {e}")
        return jsonify({"success": False, "message": f"İşlem hatası: {str(e)}"}), 500

# --- PDF MERGE ---
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "pdf", "merge")
    if not status["allowed"]: return jsonify(status), 403

    if "pdf_files" not in request.files: return jsonify({"success": False, "message": "Dosya yok"}), 400
    files = request.files.getlist("pdf_files")
    if len(files) < 2: return jsonify({"success": False, "message": "En az 2 dosya gerekli"}), 400

    merger = PdfMerger()
    for f in files:
        f.seek(0)
        merger.append(io.BytesIO(f.read()))

    output = io.BytesIO()
    merger.write(output)
    output.seek(0)
    
    increase_usage(email, "pdf", "merge")
    return jsonify({"success": True, "file": base64.b64encode(output.getvalue()).decode("utf-8")})

# --- SAYFALAR ---
@app.route("/")
def home(): return render_template("index.html")
@app.route("/remove-bg")
def remove_bg_page(): return render_template("background_remove.html")
@app.route("/pdf/merge")
def pdf_merge_page(): return render_template("pdf_merge.html")
@app.route("/admin_login")
def admin_login_page(): return render_template("admin_login.html")
@app.route("/admin")
def admin_panel(): return render_template("admin.html")
@app.route("/dashboard")
def dashboard_page():
    if not session.get("is_premium"): return redirect("/")
    user = get_user_data_by_email(session.get("user_email"))
    return render_template("dashboard.html", user=user or {})
@app.route("/vektor")
def vektor_page(): return render_template("vektor.html")

# --- AUTH ---
@app.route("/admin_login", methods=["POST"])
def admin_login_post():
    data = request.get_json()
    settings = load_settings()
    if data.get("email") == settings["admin"]["email"] and data.get("password") == settings["admin"]["password"]:
        session["admin_logged"] = True
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 401

@app.route("/user_login", methods=["POST"])
def user_login_endpoint():
    data = request.get_json()
    email = data.get("email")
    settings = load_settings()
    
    if email == settings["admin"]["email"]: return jsonify({"status": "admin"})
    
    user = get_user_data_by_email(email)
    if user:
        try:
            if datetime.strptime(user.get("end_date"), "%Y-%m-%d") >= datetime.now():
                session["user_email"] = email
                session["is_premium"] = True
                return jsonify({"status": "premium"})
            else: return jsonify({"status": "expired"})
        except: return jsonify({"status": "error"})
    return jsonify({"status": "not_found"})

@app.route("/logout")
def user_logout():
    session.clear()
    return redirect("/")

@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged", None)
    return redirect("/admin_login")

# --- ADMIN API ---
@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    if not session.get("admin_logged"): return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def add_premium_user():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json()
    users = load_premium_users()
    if any(u["email"] == data["email"] for u in users):
         return jsonify({"status": "error", "message": "Bu e-posta zaten kayıtlı"}), 409
    users.append({"email": data["email"], "end_date": data["end_date"], "usage_stats": {}})
    save_premium_users(users)
    return jsonify({"status": "ok", "message": "Kullanıcı eklendi"})

@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def delete_premium_user(email):
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    users = load_premium_users()
    new_users = [u for u in users if u["email"] != email]
    save_premium_users(new_users)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
