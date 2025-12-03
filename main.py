import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from PyPDF2 import PdfMerger
from flask import Flask, request, jsonify, render_template, session, redirect, send_file
from flask_cors import CORS
from datetime import datetime, timedelta
import json

# ============================================================
# FLASK UYGULAMASI
# ============================================================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.secret_key = "REIS_SUPER_SECRET_KEY_987654"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

SETTINGS_FILE = "settings.json"
PREMIUM_FILE = "premium_users.json"


# ============================================================
# JSON LOAD & SAVE
# ============================================================

def load_settings():
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE):
        with open(PREMIUM_FILE, "w") as f:
            json.dump([], f)
    with open(PREMIUM_FILE, "r") as f:
        return json.load(f)

def save_premium_users(users):
    with open(PREMIUM_FILE, "w") as f:
        json.dump(users, f, indent=4)


# ============================================================
# PREMIUM KONTROL SİSTEMİ
# ============================================================

def check_user_status(email, tool, subtool):
    settings = load_settings()
    users = load_premium_users()

    # Premium kullanıcı bul
    user = None
    for u in users:
        if u["email"] == email:
            user = u
            break

    is_premium = False
    if user:
        end = datetime.strptime(user["end"], "%Y-%m-%d")
        if end >= datetime.now():
            is_premium = True

    # Limitleri çek
    if tool == "pdf":
        limits = settings["limits"]["pdf"][subtool]
    elif tool == "image":
        limits = settings["limits"]["image"][subtool]
    elif tool == "vector":
        limits = settings["limits"]["vector"]
    else:
        return {"allowed": False, "reason": "invalid_tool", "left": 0}

    # Premium kullanıcı
    if is_premium:
        max_limit = limits["premium"]
        usage = user["usage"]
        left = max_limit - usage

        if left <= 0:
            return {"allowed": False, "reason": "premium_limit_full", "left": 0}

        return {"allowed": True, "reason": "premium_ok", "left": left}

    # Free kullanıcı session limiti
    if "free_usage" not in session:
        session["free_usage"] = {}

    if tool not in session["free_usage"]:
        session["free_usage"][tool] = {}

    if subtool not in session["free_usage"][tool]:
        session["free_usage"][tool][subtool] = 0

    usage = session["free_usage"][tool][subtool]
    max_limit = limits["free"]
    left = max_limit - usage

    if left <= 0:
        return {"allowed": False, "reason": "free_limit_full", "left": 0}

    return {"allowed": True, "reason": "free_ok", "left": left}


def increase_usage(email, tool, subtool):
    users = load_premium_users()

    # premium kullanıcıysa JSON'a yaz
    for u in users:
        if u["email"] == email:
            u["usage"] += 1
            save_premium_users(users)
            return

    # free kullanıcıysa session'a yaz
    session["free_usage"][tool][subtool] += 1


def check_if_premium(email):
    users = load_premium_users()
    now = datetime.now()

    for u in users:
        if u["email"] == email:
            end = datetime.strptime(u["end"], "%Y-%m-%d")
            return end >= now

    return False


# ============================================================
# SAYFALAR
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vektor')
def vektor():
    return render_template('vector.html')

@app.route('/pdf/merge')
def pdf_merge_page():
    return render_template('pdf_merge.html')

@app.route('/admin')
def admin():
    if not session.get("admin_logged"):
        return redirect("/admin_login.html")
    return render_template('admin.html')

@app.route('/admin_login')
def admin_login_page():
    return render_template('admin_login.html')


# ============================================================
# ADMIN LOGIN API
# ============================================================

@app.route("/admin_login", methods=["POST"])
def admin_login():
    data = request.json
    settings = load_settings()

    if data["email"] == settings["admin"]["email"] and data["password"] == settings["admin"]["password"]:
        session["admin_logged"] = True
        return jsonify({"status": "ok"})

    return jsonify({"status": "error"}), 401


@app.route("/admin_logout")
def admin_logout():
    session.clear()
    return redirect("/admin_login.html")


# ============================================================
# SETTINGS ENDPOINTS
# ============================================================

@app.route("/get_settings")
def get_settings():
    return jsonify(load_settings())


@app.route("/save_global_settings", methods=["POST"])
def save_global_settings():
    data = request.json
    settings = load_settings()

    settings["site"]["title"] = data["title"]
    settings["site"]["footer"] = data["footer"]

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_admin", methods=["POST"])
def save_admin():
    data = request.json
    settings = load_settings()

    settings["admin"]["email"] = data["email"]

    if data["password"].strip():
        settings["admin"]["password"] = data["password"]

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_tool_limits", methods=["POST"])
def save_tool_limits():
    data = request.json
    settings = load_settings()

    settings["limits"] = data

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_popup", methods=["POST"])
def save_popup():
    data = request.json
    settings = load_settings()

    settings["popup"]["title"] = data["title"]
    settings["popup"]["desc"] = data["desc"]
    settings["popup"]["benefits"] = data["benefits"]
    settings["popup"]["phone"] = data["phone"]

    save_settings(settings)
    return jsonify({"status": "ok"})


# ============================================================
# PREMIUM USER ENDPOINTS
# ============================================================

@app.route("/get_premium_users")
def get_premium_users():
    return jsonify(load_premium_users())


@app.route("/add_premium_user", methods=["POST"])
def add_premium_user():
    data = request.json
    users = load_premium_users()

    start = datetime.now()
    end = start + timedelta(days=data["days"])

    users.append({
        "email": data["email"],
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "active": True,
        "usage": 0
    })

    save_premium_users(users)
    return jsonify({"status": "ok"})


@app.route("/reset_usage", methods=["POST"])
def reset_usage():
    data = request.json
    users = load_premium_users()

    for u in users:
        if u["email"] == data["email"]:
            u["usage"] = 0
            break

    save_premium_users(users)
    return jsonify({"status": "ok"})


@app.route("/delete_premium_user", methods=["POST"])
def delete_user():
    data = request.json
    users = load_premium_users()

    users = [u for u in users if u["email"] != data["email"]]

    save_premium_users(users)
    return jsonify({"status": "ok"})


# ============================================================
# PDF MERGE (SENİN KODUN + PREMIUM SYSTEM ENTEGRE)
# ============================================================

@app.route('/api/pdf/merge', methods=['POST', 'OPTIONS'])
def api_pdf_merge():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        print("PDF Merge endpoint çağrıldı")

        user_email = request.cookies.get('user_email') or 'guest'

        # PREMIUM KONTROL BURADA
        status = check_user_status(user_email, "pdf", "merge")
        if not status["allowed"]:
            return jsonify({
                "success": False,
                "reason": status["reason"],
                "left": status["left"],
                "popup": True
            }), 403

        if 'pdf_files' not in request.files:
            return jsonify({"success": False, "message": "Lütfen PDF dosyaları seçin."}), 400
        
        files = request.files.getlist("pdf_files")

        # Boyut limitlerini settings.json’dan çekelim
        settings = load_settings()
        MAX_TOTAL_SIZE = settings["limits"]["pdf"]["max_size"] * 1024 * 1024

        merger = PdfMerger()
        pdf_files = []
        total_size = 0

        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                continue

            f.seek(0, 2)
            size = f.tell()
            f.seek(0)

            total_size += size
            if total_size > MAX_TOTAL_SIZE:
                return jsonify({"success": False, "message": "Toplam PDF boyutu limit aşımı"}), 413

            pdf_files.append(f)

        if len(pdf_files) < 2:
            return jsonify({"success": False, "message": "En az 2 PDF gerekir"}), 400

        for pdf in pdf_files:
            pdf.seek(0)
            merger.append(io.BytesIO(pdf.read()))

        output = io.BytesIO()
        merger.write(output)
        merger.close()
        output.seek(0)

        increase_usage(user_email, "pdf", "merge")

        pdf_data = output.read()
        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

        return jsonify({
            "success": True,
            "file": pdf_base64,
            "filename": "birlesik.pdf"
        })

    except Exception as e:
        print("PDF Merge Error:", e)
        return jsonify({"success": False, "message": str(e)}), 500


# ============================================================
# VEKTÖRLEŞTİRME
# ============================================================

def allowed_image(filename):
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions

def load_image_from_file(file_storage):
    image_bytes = file_storage.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Resim okunamadı.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

@app.route('/vectorize_style', methods=['POST'])
def vectorize_with_style():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Resim dosyası yok.'}), 400

        file = request.files['image']
        if not allowed_image(file.filename):
            return jsonify({'success': False, 'error': 'Geçersiz format.'}), 400

        image = load_image_from_file(file)

        # (Buraya senin vektör kodlarını ekleyeceğiz)

        vector_base64 = "base64_encoded_image"

        return jsonify({'success': True, 'image_data': vector_base64})

    except Exception as e:
        print("vectorize error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\nSERVER RUNNING on PORT:", port)
    app.run(host='0.0.0.0', port=port, debug=True)
