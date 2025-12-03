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
    with open("settings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



# ============================================================
# PREMIUM USER SYSTEM
# ============================================================
PREMIUM_FILE = "premium_users.json"

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

    # PREMIUM mı?
    premium_user = None
    for u in users:
        if u["email"] == email:
            end = datetime.strptime(u["end"], "%Y-%m-%d")
            if end >= datetime.now():
                premium_user = u
            break

    if tool == "pdf":
        limits = settings["limits"]["pdf"][subtool]
    elif tool == "image":
        limits = settings["limits"]["image"][subtool]
    elif tool == "vector":
        limits = settings["limits"]["vector"]
    else:
        return {"allowed": False, "reason": "invalid_tool"}

    # PREMIUM
    if premium_user:
        left = limits["premium"] - premium_user["usage"]
        if left <= 0:
            return {"allowed": False, "reason": "premium_limit_full", "left": 0}
        return {"allowed": True, "premium": True, "left": left}

    # FREE
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

    return {"allowed": True, "premium": False, "left": left}



def increase_usage(email, tool, subtool):
    users = load_premium_users()

    # PREMIUM ise dosyada artar
    for u in users:
        if u["email"] == email:
            u["usage"] += 1
            save_premium_users(users)
            return

    # FREE ise session içinde artar
    session["free_usage"][tool][subtool] += 1



# ============================================================
# ADMIN LOGIN SYSTEM
# ============================================================
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
        session["admin_logged"] = True
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "error"}), 401


@app.route("/admin_logout")
def admin_logout():
    session.pop("admin_logged", None)
    return redirect("/admin_login")


@app.route("/admin")
def admin_panel():
    if not session.get("admin_logged"):
        return redirect("/admin_login")
    return render_template("admin.html")



# ============================================================
# SETTINGS API
# ============================================================
@app.route("/get_settings")
def api_get_settings():
    return jsonify(load_settings())



@app.route("/save_tool_limits", methods=["POST"])
def save_tool_limits_api():
    data = request.get_json()
    settings = load_settings()

    settings["limits"]["pdf"] = data["pdf"]
    settings["limits"]["image"] = data["image"]
    settings["limits"]["vector"] = data["vector"]

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_popup", methods=["POST"])
def save_popup():
    data = request.get_json()
    settings = load_settings()

    settings["popup"] = data

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_global_settings", methods=["POST"])
def save_global_settings():
    data = request.get_json()
    settings = load_settings()

    settings["site"]["title"] = data["title"]
    settings["site"]["footer"] = data["footer"]

    save_settings(settings)
    return jsonify({"status": "ok"})


@app.route("/save_admin", methods=["POST"])
def save_admin():
    data = request.get_json()
    settings = load_settings()

    settings["admin"]["email"] = data["email"]
    if data["password"]:
        settings["admin"]["password"] = data["password"]

    save_settings(settings)
    return jsonify({"status": "ok"})



# ============================================================
# PDF MERGE API
# ============================================================
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():

    email = request.form.get("email", "guest")

    status = check_user_status(email, "pdf", "merge")
    if not status["allowed"]:
        return jsonify({
            "success": False,
            "reason": status["reason"],
            "left": status["left"]
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
# BACKGROUND REMOVER (ONNX)
# ============================================================
print("Loading U2Net ONNX model...")
U2NET_MODEL = "/app/models/u2net.onnx"

u2net_session = ort.InferenceSession(
    U2NET_MODEL,
    providers=["CPUExecutionProvider"]
)

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

    email = request.form.get("email", "guest")

    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]:
        return jsonify({
            "success": False,
            "reason": status["reason"],
            "left": status["left"]
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
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("BOTLAB SUITE BACKEND STARTING...")
    app.run(host="0.0.0.0", port=port, debug=True)
