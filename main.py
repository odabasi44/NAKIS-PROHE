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

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# ============================================================
# AYARLAR SİSTEMİ
# ============================================================
def load_settings():
    default_settings = {
        "admin": {"email": "admin@botlab.com", "password": "admin"},
        "limits": {}, 
        "packages": {}, 
        "site": {}, 
        "tool_status": {}
    }
    
    if not os.path.exists("settings.json"): return default_settings
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except: return default_settings

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ============================================================
# KULLANICI SİSTEMİ
# ============================================================
PREMIUM_FILE = "users.json"
TIER_RESTRICTIONS = {
    "free": ["vector", "pdf_split", "word2pdf"], 
    "starter": ["vector"], 
    "pro": [], 
    "unlimited": []
}

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE): return []
    try: with open(PREMIUM_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_premium_users(data):
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f: json.dump(data, f, indent=4, ensure_ascii=False)

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

    if "user_email" in session:
        settings = load_settings()
        if session["user_email"] == settings["admin"]["email"]: return
        
        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
                session["user_tier"] = user.get("tier", "free")
                session["is_premium"] = (end_date >= datetime.now())
                
                # Süre bittiyse paketi free yap
                if not session["is_premium"]: session["user_tier"] = "free"
             except: 
                 session["is_premium"] = False
                 session["user_tier"] = "free"
        else: 
            session.pop("user_email", None)
            session["is_premium"] = False
            session["user_tier"] = "free"
    
    if "is_premium" not in session: session["is_premium"] = False
    if "user_tier" not in session: session["user_tier"] = "free"

# --- LİMİT KONTROL ---
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()
    premium_user = None
    
    if email != "guest":
        for u in users:
            if u.get("email", "").lower() == email.lower():
                try:
                    if datetime.strptime(u.get("end_date"), "%Y-%m-%d") >= datetime.now(): premium_user = u
                except: pass
                break
    
    user_tier = premium_user.get("tier", "starter") if premium_user else "free"
    
    # Kısıtlı Araç Kontrolü
    check_key = subtool if subtool else tool
    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier, "left": 0, "premium": bool(premium_user)}

    # Limitleri Çek
    tool_limits = settings.get("limits", {}).get(tool, {})
    limit = tool_limits.get(subtool, {}).get(user_tier, 2)
    
    # Bakım Modu
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": (user_tier != "free")}

    current_usage = 0
    if user_tier == "free":
        # Free kullanıcının limiti session'da
        if "free_usage" not in session: session["free_usage"] = {}; session.modified = True
        if tool not in session["free_usage"]: session["free_usage"][tool] = {}; session.modified = True
        current_usage = session["free_usage"][tool].get(subtool, 0)
    else:
        # Premium kullanıcının limiti veritabanında
        current_usage = premium_user.get("usage_stats", {}).get(subtool, 0)

    left = limit - current_usage
    
    if left <= 0:
        reason = "free_limit_full" if user_tier == "free" else "premium_limit_full"
        return {"allowed": False, "reason": reason, "left": 0, "premium": (user_tier != "free"), "tier": user_tier}

    return {"allowed": True, "reason": "", "premium": (user_tier != "free"), "left": left, "tier": user_tier}

def increase_usage(email, tool, subtool):
    if email != "guest":
        users = load_premium_users()
        for u in users:
            if u.get("email", "").lower() == email.lower():
                if "usage_stats" not in u: u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                u["usage"] = u.get("usage", 0) + 1
                save_premium_users(users)
                return
    
    # Misafir Limiti
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1
    session.modified = True

# ============================================================
# API ROUTES
# ============================================================
@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    user = get_user_data_by_email(email)
    usage = user.get("usage_stats", {}).get(subtool, 0) if user else 0
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
possible_paths = ["/data/ai-models/u2net.onnx", "u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"]
for path in possible_paths:
    if os.path.exists(path):
        try: u2net_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); model_input_name = u2net_session.get_inputs()[0].name; break
        except: pass

def preprocess_bg(img):
    img = img.convert("RGB").resize((320, 320))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr.reshape(1, 3, 320, 320)
def postprocess_bg(mask, size):
    mask = mask.squeeze(); mask = cv2.resize(mask, size); mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8); return mask

@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    if not u2net_session: return jsonify({"success": False, "reason": "AI Modeli Yok"}), 503
    email = session.get("user_email", "guest")
    
    # Limit Kontrolü
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]: return jsonify(status), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    
    # Dosya Boyutu Kontrolü
    file.seek(0, os.SEEK_END); size = file.tell(); file.seek(0)
    settings = load_settings()
    tier = status.get("tier", "free")
    limit_mb = settings["limits"]["file_size"].get(tier, 5)
    
    if size > limit_mb * 1024 * 1024: 
        return jsonify({"success": False, "reason": "file_size_limit", "message": f"Dosya limiti: {limit_mb}MB"}), 413
    
    try:
        img = Image.open(file.stream); ow, oh = img.size
        output = u2net_session.run(None, {model_input_name: preprocess_bg(img)})[0]
        mask = postprocess_bg(output, (ow, oh))
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG")
        
        # Orijinal işlem
        rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        out_buf = io.BytesIO()
        Image.fromarray(rgba).save(out_buf, format="PNG")
        
        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(out_buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

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
    merger.write(output); output.seek(0)
    increase_usage(email, "pdf", "merge")
    return jsonify({"success": True, "file": base64.b64encode(output.getvalue()).decode("utf-8")})

# ============================================================
# PAGE ROUTES (Settings verisi gönderiliyor)
# ============================================================
@app.route("/")
def home():
    settings = load_settings()
    # Frontend'in fiyatları ve paketleri görmesi için settings'i gönderiyoruz
    return render_template("index.html", settings=settings)

@app.route("/remove-bg")
def remove_bg_page():
    settings = load_settings()
    return render_template("background_remove.html", settings=settings)

@app.route("/pdf/merge")
def pdf_merge_page():
    return render_template("pdf_merge.html")

@app.route("/admin_login")
def admin_login_page():
    return render_template("admin_login.html")

@app.route("/admin")
def admin_panel():
    if session.get("user_role") != "admin": return redirect("/admin_login")
    return render_template("admin.html")

@app.route("/dashboard")
def dashboard_page():
    if not session.get("user_email"): return redirect("/")
    user = get_user_data_by_email(session.get("user_email"))
    return render_template("dashboard.html", user=user or {})

@app.route("/vektor")
def vektor_page():
    return render_template("vektor.html")

# --- AUTH ---
@app.route("/admin_login", methods=["POST"])
def admin_login_post():
    data = request.get_json()
    settings = load_settings()
    if data.get("email") == settings["admin"]["email"] and data.get("password") == settings["admin"]["password"]:
        session["admin_logged"] = True
        session["user_email"] = data.get("email")
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

# --- ADMIN API (AYARLAR & KULLANICILAR) ---
@app.route("/api/admin/users", methods=["GET"])
def get_all_users(): 
    if not session.get("admin_logged"): return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def add_premium_user():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json(); users = load_premium_users()
    if any(u["email"] == data["email"] for u in users): return jsonify({"status": "error", "message": "Kayıtlı"}), 409
    
    # Kullanıcıyı tier (paket) bilgisiyle kaydet
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
    users = load_premium_users(); new_users = [u for u in users if u["email"] != email]
    save_premium_users(new_users); return jsonify({"status": "ok"})

# [YENİ] PAKET BİLGİLERİNİ KAYDETME
@app.route("/api/admin/save_packages", methods=["POST"])
def save_packages_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json()
    settings = load_settings()
    if "packages" in data: settings["packages"] = data["packages"]
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/api/admin/save_limits", methods=["POST"])
def save_tool_limits_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json(); settings = load_settings()
    if "limits" in data: settings["limits"] = data["limits"]
    save_settings(settings); return jsonify({"status": "ok"})

@app.route("/get_settings", methods=["GET"])
def api_get_settings():
    return jsonify(load_settings())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
