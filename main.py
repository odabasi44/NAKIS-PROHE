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
            "pdf": {"merge": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}},
            "image": {"remove_bg": {"free": 2, "starter": 20, "pro": 200, "unlimited": 9999}},
            "vector": {"default": {"free": 0, "starter": 0, "pro": 10, "unlimited": 9999}}
        },
        "packages": {
            "free": {"name": "Ücretsiz"},
            "starter": {"name": "Başlangıç"},
            "pro": {"name": "Pro"},
            "unlimited": {"name": "Sınırsız"}
        },
        "site": {},
        "tool_status": {}
    }
    if not os.path.exists("settings.json"): return default_settings
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # Eksik alanları tamamla
            if "limits" not in data: data["limits"] = default_settings["limits"]
            if "packages" not in data: data["packages"] = default_settings["packages"]
            return data
    except: return default_settings

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- KULLANICI SİSTEMİ ---
PREMIUM_FILE = "users.json"

# Hangi pakette hangi araçlar TAMAMEN KAPALI? (Görünür ama kilitli)
TIER_RESTRICTIONS = {
    "free": ["vector", "pdf_split", "pdf_compress", "word2pdf"], 
    "starter": ["vector"], # Başlangıç paketi vektörü göremez
    "pro": [], # Her şey açık
    "unlimited": []
}

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

    # Kullanıcı oturumu kontrolü
    if "user_email" in session:
        # Admin ise kontrolü atla
        settings = load_settings()
        if session["user_email"] == settings["admin"]["email"]:
            return

        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
                # Paket bilgisini session'a işle
                session["user_tier"] = user.get("tier", "free") # Varsayılan free
                # Süresi bitmişse free'ye düşür, oturumu kapatma
                if end_date < datetime.now():
                    session["user_tier"] = "free"
                    session["is_premium"] = False
                else:
                    session["is_premium"] = True
             except: 
                 session["is_premium"] = False
                 session["user_tier"] = "free"
        else: 
            # Kullanıcı silinmişse oturumu düşür
            session.pop("user_email", None)
            session["is_premium"] = False
            session["user_tier"] = "free"
    
    if "is_premium" not in session: session["is_premium"] = False
    if "user_tier" not in session: session["user_tier"] = "free"

# --- LİMİT KONTROL ---
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()
    user_data = None
    
    # Kullanıcı verisini bul
    if email != "guest":
        for u in users:
            if u.get("email", "").lower() == email.lower():
                user_data = u
                break
    
    # Paket belirle
    user_tier = "free"
    if user_data:
        try:
            end_date = datetime.strptime(user_data.get("end_date"), "%Y-%m-%d")
            if end_date >= datetime.now():
                user_tier = user_data.get("tier", "free")
        except: pass
    
    # Kısıtlı Araç Kontrolü
    check_key = subtool if subtool else tool
    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier, "left": 0, "premium": (user_tier != "free")}

    # Limitleri Çek
    tool_limits = settings.get("limits", {}).get(tool, {})
    # Pakete özel limit
    limit = tool_limits.get(subtool, {}).get(user_tier, 0)
    
    # Bakım Modu
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": (user_tier != "free")}
    
    # Limit Kontrolü
    current_usage = 0
    if user_tier == "free":
        # Free ise session'dan oku
        if "free_usage" not in session: 
            session["free_usage"] = {}
            session.modified = True
        if tool not in session["free_usage"]: 
            session["free_usage"][tool] = {}
            session.modified = True
        current_usage = session["free_usage"][tool].get(subtool, 0)
    else:
        # Premium ise DB'den oku
        current_usage = user_data.get("usage_stats", {}).get(subtool, 0)

    left = limit - current_usage
    
    if left <= 0:
        reason = "free_limit_full" if user_tier == "free" else "premium_limit_full"
        return {"allowed": False, "reason": reason, "left": 0, "premium": (user_tier != "free"), "tier": user_tier}

    return {"allowed": True, "reason": "", "premium": (user_tier != "free"), "left": left, "tier": user_tier}

def increase_usage(email, tool, subtool):
    users = load_premium_users()
    
    # Kullanıcıyı bul
    user_idx = -1
    for i, u in enumerate(users):
        if u.get("email", "").lower() == email.lower():
            user_idx = i
            break
            
    # Eğer kayıtlı kullanıcı ise ve süresi varsa DB'ye işle
    if user_idx != -1:
        user = users[user_idx]
        try:
            end_date = datetime.strptime(user.get("end_date"), "%Y-%m-%d")
            if end_date >= datetime.now():
                if "usage_stats" not in user: user["usage_stats"] = {}
                user["usage_stats"][subtool] = user["usage_stats"].get(subtool, 0) + 1
                # Toplam kullanım
                user["usage"] = user.get("usage", 0) + 1
                save_premium_users(users)
                return
        except: pass

    # Değilse (veya süresi bitmişse) Session'a işle
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    
    current = session["free_usage"][tool].get(subtool, 0)
    session["free_usage"][tool][subtool] = current + 1
    session.modified = True 

# --- API ENDPOINTLERİ ---
@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    
    # Kullanım bilgisini de dönelim
    usage = 0
    user = get_user_data_by_email(email)
    if user and status["premium"]:
        usage = user.get("usage_stats", {}).get(subtool, 0)
    else:
        # Free usage
        if "free_usage" in session and tool in session["free_usage"]:
            usage = session["free_usage"][tool].get(subtool, 0)
            
    return jsonify({
        "allowed": status.get("allowed", False),
        "reason": status.get("reason", ""),
        "left": status.get("left", 0),
        "premium": status.get("premium", False),
        "tier": status.get("tier", "free"),
        "usage": usage
    })

# --- REMOVE BG ---
u2net_session = None
model_input_name = "input"

# Model Bulucu
possible_paths = ["/data/ai-models/u2net.onnx", "u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"]
found_path = None
for path in possible_paths:
    if os.path.exists(path):
        found_path = path; break

if found_path:
    try:
        u2net_session = ort.InferenceSession(found_path, providers=["CPUExecutionProvider"])
        model_input_name = u2net_session.get_inputs()[0].name
        print(f"Model OK: {model_input_name}")
    except Exception as e: print(f"Model Error: {e}")

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
    if not u2net_session: return jsonify({"success": False, "reason": "AI Modeli Yok"}), 503
    email = session.get("user_email", "guest")
    
    # 1. Limit Kontrolü
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]: return jsonify(status), 403

    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    
    # 2. Boyut Kontrolü 
    file.seek(0, os.SEEK_END); size = file.tell(); file.seek(0)
    # Pakete göre boyut limiti
    limit_mb = 5 # Free
    if status["premium"]:
        if status["tier"] == "starter": limit_mb = 10
        elif status["tier"] == "pro": limit_mb = 50
        elif status["tier"] == "unlimited": limit_mb = 100
        
    if size > limit_mb * 1024 * 1024:
         return jsonify({"success": False, "reason": "file_size_limit", "message": f"Limit aşıldı. Max: {limit_mb}MB"}), 413
    
    try:
        img = Image.open(file.stream)
        ow, oh = img.size
        output = u2net_session.run(None, {model_input_name: preprocess_bg(img)})[0]
        mask = postprocess_bg(output, (ow, oh))
        rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        
        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- PDF MERGE ---
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "pdf", "merge")
    if not status["allowed"]: return jsonify(status), 403

    if "pdf_files" not in request.files: return jsonify({"success": False}), 400
    files = request.files.getlist("pdf_files")
    
    merger = PdfMerger()
    for f in files: merger.append(io.BytesIO(f.read()))
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
    if not session.get("user_email"): return redirect("/")
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
        session["user_email"] = data.get("email") # Admin için de email set et
        session["user_tier"] = "unlimited"
        session["is_premium"] = True
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
                session["user_tier"] = user.get("tier", "starter")
                return jsonify({"status": "premium", "tier": session["user_tier"]})
            else: return jsonify({"status": "expired"})
        except: return jsonify({"status": "error"})
    return jsonify({"status": "not_found"})

@app.route("/logout")
def user_logout():
    session.clear()
    return redirect("/")

@app.route("/admin_logout")
def admin_logout():
    session.clear()
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
         return jsonify({"status": "error", "message": "Kayıtlı"}), 409
    # Paket seçimi ile ekle
    users.append({
        "email": data["email"], 
        "end_date": data["end_date"], 
        "tier": data.get("tier", "starter"), 
        "usage_stats": {}
    })
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
