import os
import io
import json
import base64
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from PyPDF2 import PdfMerger
from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
import onnxruntime as ort

# --- YENİ KÜTÜPHANE: MEDIAPIPE (Yüz Hatlarını Çizmek İçin) ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
    print("✅ MediaPipe Yüklendi: Yüz hatları anatomik olarak çizilecek.")
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ UYARI: 'mediapipe' kütüphanesi eksik! 'pip install mediapipe' yapınız. (Sistem çalışır ama yüz çizgileri eksik olur)")

# --- AI MODEL YÜKLEME (WHITE-BOX CARTOONIZATION) ---
gan_session = None
base_dir = os.path.dirname(os.path.abspath(__file__))

# Model Yolları (Coolify ve Local uyumlu)
possible_model_paths = [
    "/app/models/whitebox_cartoon.onnx",               # 1. Coolify Standart
    "/data/models/whitebox_cartoon.onnx",              # 2. Volume
    os.path.join(base_dir, "models", "whitebox_cartoon.onnx"), # 3. Proje içi
    os.path.join(base_dir, "whitebox_cartoon.onnx"),           # 4. Ana dizin
]

print("--- AI MODEL ARAMA BAŞLADI (White-box) ---")
final_model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)

if final_model_path:
    try:
        gan_session = ort.InferenceSession(final_model_path, providers=["CPUExecutionProvider"])
        print(f"🚀 WHITE-BOX AI MODELİ BAŞARIYLA YÜKLENDİ: {final_model_path}")
    except Exception as e:
        print(f"⚠️ DOSYA VAR AMA YÜKLENEMEDİ: {e}")
        gan_session = None
else:
    print("🚨 KRİTİK HATA: 'whitebox_cartoon.onnx' bulunamadı! Lütfen models klasörünü kontrol edin.")

# --- U2NET MODELİ (Arka Plan Silme) ---
u2net_session = None
u2net_input_name = "input"
u2net_path = next((p for p in ["u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"] if os.path.exists(p)), None)

if u2net_path:
    try:
        u2net_session = ort.InferenceSession(u2net_path, providers=["CPUExecutionProvider"])
        u2net_input_name = u2net_session.get_inputs()[0].name
        print(f"✅ U2Net Modeli Yüklendi: {u2net_path}")
    except: pass

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# --- AYARLAR SİSTEMİ ---
SETTINGS_FILE = "settings.json"
PREMIUM_FILE = "users.json"

def load_settings():
    default_settings = {
        "admin": {"email": "admin@botlab.com", "password": "admin"},
        "limits": {
            "image": {
                "remove_bg": {"free": 2, "starter": 20, "pro": 200, "unlimited": 9999},
                "compress": {"free": 5, "starter": 50, "pro": 500, "unlimited": 9999},
                "convert": {"free": 5, "starter": 50, "pro": 500, "unlimited": 9999}
            },
            "pdf": {
                "merge": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "split": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}, # Eksikse ekle
                "compress": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "word2pdf": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}
            },
            "vector": {
                "default": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}
            },
            "file_size": {"free": 5, "starter": 10, "pro": 50, "unlimited": 100}
        },
        "packages": {},
        "site": {},
        "tool_status": {}
    }
    
    if not os.path.exists(SETTINGS_FILE): return default_settings
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Eksik anahtarları varsayılanlarla doldur
            if "limits" not in data: data["limits"] = default_settings["limits"]
            return data
    except: return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- KULLANICI SİSTEMİ ---
TIER_RESTRICTIONS = {
    "free": ["vector", "pdf_split", "word2pdf"], 
    "starter": ["vector"], 
    "pro": [], 
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
    # Admin paneli koruması
    if request.path.startswith("/admin") and request.path != "/admin_login":
        if not session.get("admin_logged"): return redirect("/admin_login")

    # Kullanıcı oturum ve paket kontrolü
    if "user_email" in session:
        settings = load_settings()
        # Admin ise kontrol etme
        if session["user_email"] == settings.get("admin", {}).get("email"): return

        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
                session["user_tier"] = user.get("tier", "free")
                session["is_premium"] = (end_date >= datetime.now())
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

# --- LİMİT KONTROL MOTORU ---
def check_user_status(email, tool, subtool):
    settings = load_settings()
    user_tier = "free"
    user_data = None
    
    if email != "guest":
        user_data = get_user_data_by_email(email)
        if user_data:
            try:
                if datetime.strptime(user_data.get("end_date"), "%Y-%m-%d") >= datetime.now():
                    user_tier = user_data.get("tier", "free")
            except: pass
            
    check_key = subtool if subtool else tool
    # Yasaklı araç kontrolü
    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier, "left": 0, "premium": (user_tier != "free")}

    # Bakım modu
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": (user_tier != "free")}

    # Limit hesaplama
    tool_limits = settings.get("limits", {}).get(tool, {})
    limit = tool_limits.get(subtool, {}).get(user_tier, 0)
    
    current_usage = 0
    if user_tier == "free":
        if "free_usage" not in session: 
            session["free_usage"] = {}
            session.modified = True
        if tool not in session["free_usage"]: 
            session["free_usage"][tool] = {}
            session.modified = True
        current_usage = session["free_usage"][tool].get(subtool, 0)
    else:
        current_usage = user_data.get("usage_stats", {}).get(subtool, 0)

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
                save_premium_users(users)
                return

    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    current = session["free_usage"][tool].get(subtool, 0)
    session["free_usage"][tool][subtool] = current + 1
    session.modified = True

# --- VEKTÖR MOTORU (MediaPipe + AI + Clean Style) ---
# --- VEKTÖR MOTORU (MediaPipe + AI + Clean Style) ---
class VectorEngine:
    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if self.original_img is None:
            raise ValueError("Görüntü okunamadı")

        # Şeffaflık varsa beyaz yap
        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            alpha = self.original_img[:, :, 3]
            rgb = self.original_img[:, :, :3]
            white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            self.img = (rgb * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)
        else:
            self.img = self.original_img[:, :, :3]

        # Vektör kalitesi için boyut ayarı (1024px)
        h, w = self.img.shape[:2]
        target_dim = 1024
        if max(h, w) > target_dim:
            scale = target_dim / max(h, w)
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        self.h, self.w = self.img.shape[:2]

    # MediaPipe Face Edges
    def extract_face_edges(self, img_in):
        if not HAS_MEDIAPIPE: 
            return np.zeros((img_in.shape[0], img_in.shape[1]), dtype=np.uint8)
        
        h, w = img_in.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                contours_indices = [
                    mp_face_mesh.FACEMESH_LIPS,
                    mp_face_mesh.FACEMESH_LEFT_EYE,
                    mp_face_mesh.FACEMESH_RIGHT_EYE,
                    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    mp_face_mesh.FACEMESH_FACE_OVAL
                ]
                for indices in contours_indices:
                    points_set = set()
                    for connection in indices:
                        points_set.add(connection[0])
                        points_set.add(connection[1])
                    pts = []
                    for idx in points_set:
                        x = int(lm[idx].x * w)
                        y = int(lm[idx].y * h)
                        pts.append((x, y))
                    if pts:
                        hull = cv2.convexHull(np.array(pts))
                        cv2.polylines(mask, [hull], True, 255, 2)
        return mask

    # --- YENİ ALGORİTMA: Yüzdeki Gölgeleri Temizleyen V2 ---
    def process_cartoon_style_v2(self):
        """
        DÜZELTME: Yüz ortasındaki 'banding' (kalın gölge çizgileri) sorununu çözer.
        Yöntem: Gamma Correction (Aydınlatma) + Canny Edge (Sadece keskin kenar)
        """
        h, w = self.img.shape[:2]

        # 1. GAMMA CORRECTION (Gölge Açma)
        # Yüzdeki hafif gölgeleri uçurmak için resmi aydınlatıyoruz
        gamma = 1.3  # Değer 1.0'dan büyük olmalı (1.3 ideal)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        bright_img = cv2.LUT(self.img, table)

        # 2. YUMUŞATMA (Detay Azaltma)
        # Sivilce, benek gibi detayları yok etmek için güçlü blur
        blurred = cv2.bilateralFilter(bright_img, d=15, sigmaColor=100, sigmaSpace=100)

        # 3. RENK DÜZLEŞTİRME (K-Means)
        data = np.float32(blurred).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # 9 Renk bloğu yeterli
        _, label, center = cv2.kmeans(data, 9, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        flat_color = center[label.flatten()].reshape(blurred.shape)

        # 4. KENAR TESPİTİ (CANNY)
        # AdaptiveThreshold yerine Canny kullanıyoruz. 
        # Canny, gölge geçişlerini değil, sadece keskin renk farklarını çizer.
        gray = cv2.cvtColor(flat_color, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # 50-150 eşik değerleri yüzdeki yumuşak gölgeleri görmezden gelir
        edges = cv2.Canny(gray, 50, 150)
        
        # Çizgileri biraz kalınlaştır
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 5. MEDIAPIPE İLE BİRLEŞTİRME
        if HAS_MEDIAPIPE:
            mp_mask = self.extract_face_edges(self.img)
            # mp_mask: Siyah zemin, Beyaz çizgi
            # edges: Siyah zemin, Beyaz çizgi
            combined_edges = cv2.bitwise_or(edges, mp_mask)
        else:
            combined_edges = edges

        # 6. SONUÇ BİRLEŞTİRME
        # Maskeyi ters çevir (Beyaz zemin, Siyah çizgi)
        mask_inv = cv2.bitwise_not(combined_edges)
        mask_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        
        # Renkli resim ile maskeyi çarp (Siyah çizgiler basılır)
        self.img = cv2.bitwise_and(flat_color, mask_bgr)

    # Eski outline metodu (Dursun, zarar gelmez)
    def process_outline(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        self.img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # SVG Çıktı
    def generate_svg(self):
        h, w = self.img.shape[:2]
        proc_img = self.img
        
        # SVG boyutu şişmesin diye renkleri biraz daha azalt (32 renk)
        data = np.float32(proc_img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, 32, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        proc_img = center[label.flatten()].reshape(proc_img.shape)
        
        unique_colors = np.unique(proc_img.reshape(-1, 3), axis=0)
        
        svg = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        
        for color in unique_colors:
            b, g, r = color
            mask = cv2.inRange(proc_img, color, color)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hex_c = "#{:02x}{:02x}{:02x}".format(r, g, b)
            
            path_d = ""
            for cnt in contours:
                if cv2.contourArea(cnt) < 25: continue
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) < 3: continue
                
                pts = approx.reshape(-1, 2)
                path_d += f"M {pts[0][0]} {pts[0][1]} "
                for p in pts[1:]:
                    path_d += f"L {p[0]} {p[1]} "
                path_d += "Z "
            
            if path_d:
                svg += f'<path d="{path_d}" fill="{hex_c}" stroke="none" />'
        
        svg += '</svg>'
        return svg

# --- API ENDPOINTS ---

# 1. VEKTÖR API (GÜNCELLENMİŞ)
# 1. VEKTÖR API (GÜNCELLENMİŞ)
@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "vector", "default")
    if not status["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403

    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    method = request.form.get("method", "normal") 
    
    try:
        engine = VectorEngine(file)
        
        if method == "outline":
            engine.process_outline()
        else:
            # ARTIK "Cartoon" veya "Normal" seçildiğinde 
            # YENİ V2 MOTORU (Gamma + Canny) çalışacak.
            # Eski lekeli yöntem devre dışı bırakıldı.
            engine.process_cartoon_style_v2()

        svg_str = engine.generate_svg()
        b64_svg = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
        
        # Önizleme
        preview = cv2.resize(engine.img, (engine.w, engine.h))
        _, buf = cv2.imencode('.png', preview)
        b64_png = base64.b64encode(buf).decode('utf-8')
        
        increase_usage(email, "vector", "default")
        return jsonify({"success": True, "file": b64_svg, "preview_img": b64_png})
    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# 2. REMOVE BG API
@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    if not u2net_session: return jsonify({"success": False, "reason": "AI Modeli Yok"}), 503
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]: return jsonify(status), 403

    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    
    try:
        img = Image.open(file.stream).convert("RGB").resize((320, 320))
        inp = np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1))
        inp = np.expand_dims(inp, 0)
        out = u2net_session.run(None, {u2net_input_name: inp})[0].squeeze()
        mask = cv2.resize(out, Image.open(request.files["image"]).size)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        rgba = cv2.cvtColor(np.array(Image.open(request.files["image"]).convert("RGB")), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# 3. GÖRSEL SIKIŞTIRMA API
@app.route("/api/img/compress", methods=["POST"])
def api_img_compress():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "compress")
    if not status["allowed"]: return jsonify(status), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    quality = int(request.form.get("quality", 70))
    
    try:
        img = Image.open(file.stream).convert("RGB")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode('utf-8')
        increase_usage(email, "image", "compress")
        return jsonify({"success": True, "file": encoded, "new_size": f"{len(output.getvalue())/1024:.1f} KB"})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

# 4. FORMAT ÇEVİRME API
@app.route("/api/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "convert")
    if not status["allowed"]: return jsonify(status), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    fmt = request.form.get("format", "jpeg").lower()
    
    # Basit eşleme
    pil_fmt = {"jpg":"JPEG", "jpeg":"JPEG", "png":"PNG", "webp":"WEBP", "pdf":"PDF"}.get(fmt, "JPEG")
    
    try:
        img = Image.open(file.stream).convert("RGB")
        out = io.BytesIO()
        img.save(out, format=pil_fmt)
        encoded = base64.b64encode(out.getvalue()).decode('utf-8')
        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": encoded})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

# 5. PDF API (Merge)
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "pdf", "merge")
    if not status["allowed"]: return jsonify(status), 403
    
    if "pdf_files" not in request.files: return jsonify({"success": False}), 400
    files = request.files.getlist("pdf_files")
    
    try:
        merger = PdfMerger()
        for f in files: merger.append(io.BytesIO(f.read()))
        out = io.BytesIO()
        merger.write(out)
        merger.close()
        encoded = base64.b64encode(out.getvalue()).decode("utf-8")
        increase_usage(email, "pdf", "merge")
        return jsonify({"success": True, "file": encoded})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

# --- ADMIN / AUTH ROUTE ---
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login_route():
    if request.method == "GET": return render_template("admin_login.html")
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
        end = datetime.strptime(user.get("end_date"), "%Y-%m-%d")
        if end >= datetime.now():
            session["user_email"] = email
            session["is_premium"] = True
            session["user_tier"] = user.get("tier", "starter")
            return jsonify({"status": "premium", "tier": session["user_tier"]})
        else: return jsonify({"status": "expired"})
    return jsonify({"status": "not_found"})

@app.route("/api/admin/users", methods=["GET"])
def get_users_api():
    if not session.get("admin_logged"): return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    data = request.get_json()
    users = load_premium_users()
    users.append({"email": data["email"], "end_date": data["end_date"], "tier": data["tier"], "usage_stats":{}})
    save_premium_users(users)
    return jsonify({"status":"ok"})

@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"): return jsonify({}), 403
    users = [u for u in load_premium_users() if u["email"]!=email]
    save_premium_users(users)
    return jsonify({"status":"ok"})

@app.route("/api/check_tool_status/<tool>/<subtool>")
def status_api(tool, subtool):
    return jsonify(check_user_status(session.get("user_email","guest"), tool, subtool))

@app.route("/logout")
def logout(): session.clear(); return redirect("/")

# --- SAYFALAR ---
@app.route("/")
def home(): return render_template("index.html")
@app.route("/<page>")
def render_page(page):
    if os.path.exists(f"templates/{page}.html"): return render_template(f"{page}.html")
    return redirect("/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

