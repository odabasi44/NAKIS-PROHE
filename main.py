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
        # Varsayılan ayarlar (dosya yoksa hata vermesin)
        return {
            "site": {"title": "Botlab Tools", "footer": "2025 Botlab"},
            "admin": {"email": "admin@botlab.com", "password": "admin"},
            "limits": {
                "pdf": {"merge": {"free": 3, "premium": 50}},
                "image": {"remove_bg": {"free": 3, "premium": 50}},
                "vector": {"free": 1, "premium": 20}
            },
            "popup": {"title": "Hoşgeldin", "content": "Duyuru..."}
        }

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ============================================================
# PREMIUM USER SYSTEM & TIERS
# ============================================================
PREMIUM_FILE = "premium_users.json"

# Paket Kısıtlamaları
TIER_RESTRICTIONS = {
    "free": ["pdf_merge", "pdf_split", "vector", "word2pdf", "resize"],
    "starter": ["vector", "pdf_split", "word2pdf"],
    "pro": [],      
    "unlimited": [] 
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
# PREMIUM / LIMIT CHECKER
# ============================================================
def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()

    # Kullanıcıyı bul
    premium_user = next((u for u in users if u["email"] == email), None)
    
    # Paket Kontrolü
    user_tier = "free"
    if premium_user:
        try:
            end_date = datetime.strptime(premium_user["end"], "%Y-%m-%d")
            if end_date >= datetime.now():
                user_tier = premium_user.get("tier", "starter")
        except:
            pass # Tarih formatı hatası olursa free say

    # Kısıtlı özellik kontrolü
    tool_key = subtool if subtool else tool 
    if tool_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier}

    # Limit Ayarları
    if tool == "pdf":
        limits = settings["limits"]["pdf"].get(subtool, {"free":0, "premium":0})
    elif tool == "image":
        limits = settings["limits"]["image"].get(subtool, {"free":0, "premium":0})
    elif tool == "vector":
        limits = settings["limits"]["vector"]
    else:
        return {"allowed": False, "reason": "invalid_tool"}

    # PREMIUM LİMİT
    if premium_user and user_tier != "free":
        left = limits["premium"] - premium_user.get("usage", 0)
        if user_tier == "unlimited":
            return {"allowed": True, "premium": True, "tier": user_tier, "left": 9999}
            
        if left <= 0:
            return {"allowed": False, "reason": "premium_limit_full", "left": 0}
        return {"allowed": True, "premium": True, "tier": user_tier, "left": left}

    # FREE LİMİT
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
            u["usage"] = u.get("usage", 0) + 1
            save_premium_users(users)
            return
    
    # Free kullanıcı session artır
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    if subtool not in session["free_usage"][tool]: session["free_usage"][tool][subtool] = 0
    session["free_usage"][tool][subtool] += 1

# ============================================================
# AUTH ROUTES (LOGIN / LOGOUT)
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
        session["user_tier"] = "unlimited"
        return jsonify({"status": "admin"})

    # 2. Premium Kullanıcı Kontrolü
    users = load_premium_users()
    user = next((u for u in users if u["email"] == email), None)

    if user:
        try:
            end_date = datetime.strptime(user["end"], "%Y-%m-%d")
            if end_date >= datetime.now():
                session["user_email"] = email
                session["user_role"] = "premium"
                session["user_tier"] = user.get("tier", "starter")
                return jsonify({"status": "premium", "tier": session["user_tier"]})
        except:
            pass # Tarih hatası
    
    return jsonify({"status": "error", "message": "Kullanıcı bulunamadı veya süresi dolmuş."}), 401

@app.route("/user_logout")
def user_logout():
    session.clear()
    return redirect("/")

@app.route("/admin_login")
def admin_login_page():
    return render_template("admin_login.html")

@app.route("/admin_login", methods=["POST"])
def admin_login_post():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    settings = load_settings()

    if email == settings["admin"]["email"] and password == settings["admin"]["password"]:
        session["user_email"] = email
        session["user_role"] = "admin"
        session["user_tier"] = "unlimited"
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "error"}), 401

# ============================================================
# ADMIN PANEL API (USER MANAGEMENT - BU KISIM EKSİKTİ)
# ============================================================
@app.route("/admin")
def admin_panel():
    if session.get("user_role") != "admin":
        return redirect("/admin_login")
    return render_template("admin.html")

@app.route("/api/admin/users", methods=["GET"])
def api_admin_users():
    if session.get("user_role") != "admin": return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def api_admin_add_user():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    
    data = request.get_json()
    users = load_premium_users()
    
    # E-posta var mı kontrol et
    if any(u["email"] == data["email"] for u in users):
        return jsonify({"status": "error", "message": "Bu e-posta zaten kayıtlı."})

    new_user = {
        "email": data["email"],
        "end": data["end"],
        "usage": 0,
        "tier": data.get("tier", "starter") # Varsayılan starter
    }
    users.append(new_user)
    save_premium_users(users)
    return jsonify({"status": "ok"})

@app.route("/api/admin/delete_user", methods=["POST"])
def api_admin_delete_user():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    
    data = request.get_json()
    users = load_premium_users()
    users = [u for u in users if u["email"] != data["email"]]
    save_premium_users(users)
    return jsonify({"status": "ok"})

# ============================================================
# SETTINGS API
# ============================================================
@app.route("/get_settings")
def api_get_settings():
    return jsonify(load_settings())

@app.route("/save_tool_limits", methods=["POST"])
def save_tool_limits_api():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    data = request.get_json()
    settings = load_settings()
    settings["limits"]["pdf"] = data["pdf"]
    settings["limits"]["image"] = data["image"]
    settings["limits"]["vector"] = data["vector"]
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/save_popup", methods=["POST"])
def save_popup():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    data = request.get_json()
    settings = load_settings()
    settings["popup"] = data
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/save_global_settings", methods=["POST"])
def save_global_settings():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    data = request.get_json()
    settings = load_settings()
    settings["site"]["title"] = data["title"]
    settings["site"]["footer"] = data["footer"]
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/save_admin", methods=["POST"])
def save_admin():
    if session.get("user_role") != "admin": return jsonify({"status":"error"}), 403
    data = request.get_json()
    settings = load_settings()
    settings["admin"]["email"] = data["email"]
    if data["password"]:
        settings["admin"]["password"] = data["password"]
    save_settings(settings)
    return jsonify({"status": "ok"})

# ============================================================
# TOOL APIs
# ============================================================

# PDF MERGE
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "pdf", "merge")
    
    if not status["allowed"]:
        return jsonify({"success": False, "reason": status["reason"]}), 403

    if "pdf_files" not in request.files:
        return jsonify({"success": False, "message": "PDF seçilmedi"}), 400

    files = request.files.getlist("pdf_files")
    if len(files) < 2:
        return jsonify({"success": False, "message": "En az 2 PDF gerekli"}), 400

    try:
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

        return jsonify({"success": True, "file": pdf_base64, "filename": "birlesik.pdf"})
    except Exception as e:
        print(e)
        return jsonify({"success": False, "message": "PDF Hatası"}), 500

# BACKGROUND REMOVER (ONNX)
# Model yoksa hata vermemesi için try-except
u2net_session = None
try:
    U2NET_MODEL = "/app/models/u2net.onnx" # Coolify path
    if os.path.exists(U2NET_MODEL):
        print("Loading U2Net ONNX model...")
        u2net_session = ort.InferenceSession(U2NET_MODEL, providers=["CPUExecutionProvider"])
    else:
        print("ONNX Model not found at", U2NET_MODEL)
except Exception as e:
    print("ONNX Init Error:", e)

def preprocess_bg(img):
    img = img.convert("RGB").resize((320, 320))
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
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "remove_bg")
    
    if not status["allowed"]:
        return jsonify({"success": False, "reason": status["reason"]}), 403

    if "image" not in request.files:
        return jsonify({"success": False, "message": "Resim yok"}), 400
    
    if u2net_session is None:
        return jsonify({"success": False, "message": "AI Modeli Yüklü Değil (Bakımda)"}), 503

    try:
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

        increase_usage(email, "image", "remove_bg")
        encoded = base64.b64encode(buf.getvalue()).decode()

        return jsonify({"success": True, "file": encoded, "filename": "arka_plan_silindi.png"})
    except Exception as e:
        print("BG Remove Error:", e)
        return jsonify({"success": False, "message": "İşlem Hatası"}), 500

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

@app.route("/vektor")
def vektor_page():
    return render_template("vector.html")

@app.route("/pdf/merge")
def pdf_merge_page():
    return render_template("pdf_merge.html")

@app.route("/remove-bg")
def remove_bg_page():
    return render_template("background_remove.html")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
