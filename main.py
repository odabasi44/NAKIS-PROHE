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

# --- AI MODEL YÜKLEME (GELİŞMİŞ) ---
gan_session = None

# Projenin çalıştığı tam yolu al
base_dir = os.path.dirname(os.path.abspath(__file__))

# Olası model yolları (Sırayla dener - Volume mount dahil)
possible_model_paths = [
    "/data/models/face_paint_512_v2.onnx",              # 1. Öncelik: Volume Mount
    os.path.join(base_dir, "models", "face_paint_512_v2.onnx"), # 2. Öncelik: Proje içi models
    os.path.join(base_dir, "face_paint_512_v2.onnx"),           # 3. Öncelik: main.py yanı
    "/app/models/face_paint_512_v2.onnx"                        # 4. Öncelik: Docker app klasörü
]

final_model_path = None

print("--- AI MODEL ARAMA BAŞLADI ---")
for path in possible_model_paths:
    if os.path.exists(path):
        print(f"✅ DOSYA BULUNDU: {path}")
        final_model_path = path
        break
    else:
        print(f"❌ Yol boş: {path}")

if final_model_path:
    try:
        # Modeli Yükle
        gan_session = ort.InferenceSession(final_model_path, providers=["CPUExecutionProvider"])
        print(f"🚀 AI MODELİ BAŞARIYLA YÜKLENDİ! Giriş: {gan_session.get_inputs()[0].name}")
    except Exception as e:
        print(f"⚠️ DOSYA VAR AMA YÜKLENEMEDİ (Kütüphane Hatası): {e}")
        gan_session = None
else:
    print("🚨 KRİTİK HATA: Model dosyası hiçbir yerde bulunamadı! Lütfen 'models' klasörünü kontrol edin.")

print("--- AI MODEL ARAMA BİTTİ ---")

# --- U2NET MODELİ (BG Remove) ---
u2net_session = None
u2net_input_name = "input"
possible_u2net_paths = ["u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"]
u2net_path = None
for path in possible_u2net_paths:
    if os.path.exists(path): u2net_path = path; break
if u2net_path:
    try:
        u2net_session = ort.InferenceSession(u2net_path, providers=["CPUExecutionProvider"])
        u2net_input_name = u2net_session.get_inputs()[0].name
        print(f"U2Net Modeli Yüklendi: {u2net_path}")
    except: pass


app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# --- AYARLAR SİSTEMİ ---
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
                "split": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "compress": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "word2pdf": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}
            },
            "vector": {
                "default": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}
            },
            "generator": {
                "qr": {"free": 0, "starter": 10, "pro": 100, "unlimited": 9999},
                "logo": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}
            },
            "file_size": {
                "free": 5, "starter": 10, "pro": 50, "unlimited": 100
            }
        },
        "packages": {},
        "site": {},
        "tool_status": {}
    }
    
    if not os.path.exists("settings.json"):
        return default_settings
        
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if "limits" not in data: data["limits"] = default_settings["limits"]
            
            # Eksik kategorileri tamamla
            defaults = default_settings["limits"]
            for main_cat, tools in defaults.items():
                if main_cat not in data["limits"]:
                    data["limits"][main_cat] = tools
                else:
                    if isinstance(tools, dict):
                        for tool, limits in tools.items():
                            if tool not in data["limits"][main_cat]:
                                data["limits"][main_cat][tool] = limits
            return data
    except:
        return default_settings

def save_settings(data):
    with open("settings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- KULLANICI SİSTEMİ ---
PREMIUM_FILE = "users.json"
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
    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier, "left": 0, "premium": (user_tier != "free")}

    tool_status = settings.get("tool_status", {}).get(subtool, {})
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": (user_tier != "free")}

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

# --- VEKTÖR MOTORU (AI DESTEKLİ + GELİŞMİŞ) ---
class VectorEngine:
    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        if self.original_img is None:
            raise ValueError("Görüntü okunamadı")

        # Şeffaflık varsa beyaz yap (AI için)
        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            alpha = self.original_img[:, :, 3]
            rgb = self.original_img[:, :, :3]
            white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
            alpha_factor = alpha[:, :, np.newaxis] / 255.0
            self.img = (rgb * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)
        else:
            self.img = self.original_img[:, :, :3]

        # İşlem hızı için max boyut
        h, w = self.img.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        self.h, self.w = self.img.shape[:2]

    def enhance_image(self):
        """
        GÜÇLENDİRİLMİŞ AYDINLATMA (Stüdyo Işığı Efekti)
        Karanlık fotoğrafları hem aydınlatır hem de renkleri canlı hale getirir.
        """
        # 1. Otomatik Beyaz Dengesi (Simple White Balance)
        # Bu işlem, sarı veya karanlık tonları temizler.
        try:
            wb = cv2.xphoto.createSimpleWB()
            self.img = wb.balanceWhite(self.img)
        except: pass # Modül yoksa devam et
        # 2. CLAHE (Kontrast Dengeleme) - Lab Renk Uzayında
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # ClipLimit'i artırarak (2.0 -> 3.0) kontrastı daha çok açıyoruz
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        self.img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 3. Gamma Düzeltmesi (Parlaklık Artırma)
        # Gamma değerini düşürerek (1.2 -> 1.5) daha fazla ışık veriyoruz.
        # Bu sayede gölgelerdeki detaylar ortaya çıkar.
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.img = cv2.LUT(self.img, table)

        # 4. Hafif Doygunluk (Saturation) Artışı
        # Renklerin soluk kalmaması için
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 30) # Doygunluğu artır
        v = cv2.add(v, 20) # Parlaklığı biraz daha artır
        final_hsv = cv2.merge((h, s, v))
        self.img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def process_with_ai_model(self):
        """AI Modeli ile Anime/Çizim efekti uygular"""
        global gan_session
        if gan_session is None:
            return # Model yoksa işlem yapma

        try:
            # AI için 512x512'ye yeniden boyutlandır
            resized_input = cv2.resize(self.img, (512, 512))
            x = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, axis=0)

            input_name = gan_session.get_inputs()[0].name
            output = gan_session.run(None, {input_name: x})[0]

            output = (output.squeeze().transpose(1, 2, 0) + 1.0) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # AI çıktısını çalışma boyutuna geri getir
            self.img = cv2.resize(output, (self.w, self.h))
        except Exception as e:
            print(f"AI Hatası: {e}")

    def process_cartoon_smart(self):
        """
        GELİŞMİŞ AVATAR MODU: Aydınlatma + AI + Kontur
        """
        # 1. YENİ: Resmi aydınlat ve detayları aç
        self.enhance_image()

        # 2. AI ile Yüzü Pürüzsüzleştir
        self.process_with_ai_model()

        # 3. Siyah Kontur Çizgilerini Çıkar
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        # BlockSize'ı artırarak daha temiz çizgiler alalım (9 -> 11)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
        
        # 4. Renkleri Düzleştir (MeanShift)
        self.img = cv2.pyrMeanShiftFiltering(self.img, sp=15, sr=30)
        
        # 5. Çizgileri Ekle
        self.img = cv2.bitwise_and(self.img, self.img, mask=edges)
        
        # 6. Son Renk Azaltma (K-Means)
        # Renk sayısını biraz artıralım ki yüz detayları kaybolmasın (8 -> 12)
        self.reduce_colors_kmeans(k=12)

    def reduce_colors_kmeans(self, k=8):
        """Standart K-Means Renk Azaltma"""
        data = np.float32(self.img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        try:
            _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            self.img = center[label.flatten()].reshape((self.img.shape))
        except: pass

    def process_outline(self):
        """Sadece Dış Hatlar"""
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        self.img = cv2.bitwise_not(dilated)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def generate_svg(self):
        """SVG Oluşturucu"""
        proc_img = self.img
        # SVG çok şişmesin diye işleme boyutunu biraz düşürebiliriz
        if max(self.h, self.w) > 600:
             scale = 600 / max(self.h, self.w)
             proc_img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        h, w = proc_img.shape[:2]
        pixels = proc_img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        svg = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {w} {h}">'
        
        for color in unique_colors:
            b, g, r = color
            # Beyaza çok yakın renkleri (arka plan) atla
            if r > 245 and g > 245 and b > 245: continue
            
            mask = cv2.inRange(proc_img, color, color)
            # Gürültü temizliği
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hex_c = "#{:02x}{:02x}{:02x}".format(r, g, b)
            
            path_d = ""
            for cnt in contours:
                if cv2.contourArea(cnt) < 30: continue
                
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
        elif method == "cartoon":
            # --- YENİ KONTROL: AI Model Yoksa Uyarı Ver ---
            if gan_session is None:
                return jsonify({"success": False, "message": "AI Modeli Yüklü Değil! Lütfen sunucu loglarını kontrol edin."}), 500
            
            # Gelişmiş Cartoon Modu (AI + Kontur + Aydınlatma)
            engine.process_cartoon_smart()
        else: # normal mod
            # Normal mod için de AI kullanabiliriz (daha pürüzsüz olur)
            if gan_session: engine.process_with_ai_model()
            engine.reduce_colors_kmeans(k=16)

        svg_str = engine.generate_svg()
        b64_svg = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
        
        # Önizleme PNG
        preview_img = cv2.resize(engine.img, (engine.w, engine.h), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.png', preview_img)
        b64_png = base64.b64encode(buf).decode('utf-8')
        
        increase_usage(email, "vector", "default")
        return jsonify({"success": True, "file": b64_svg, "preview_img": b64_png})
    except Exception as e:
        print(f"Vektör Hatası: {e}")
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
    file.seek(0, os.SEEK_END); size = file.tell(); file.seek(0)
    settings = load_settings()
    limit_mb = settings["limits"]["file_size"].get(status.get("tier", "free"), 5)
    
    if size > limit_mb * 1024 * 1024:
         return jsonify({"success": False, "reason": "file_size_limit", "message": f"Dosya limiti: {limit_mb}MB"}), 413
    
    try:
        img = Image.open(file.stream)
        def preprocess_bg(i):
            i = i.convert("RGB").resize((320, 320))
            arr = np.array(i).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return arr.reshape(1, 3, 320, 320)
        def postprocess_bg(m, s):
            m = m.squeeze(); m = cv2.resize(m, s)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            return m

        ow, oh = img.size
        output = u2net_session.run(None, {u2net_input_name: preprocess_bg(img)})[0]
        mask = postprocess_bg(output, (ow, oh))
        rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        out_buf = io.BytesIO()
        Image.fromarray(rgba).save(out_buf, format="PNG")
        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(out_buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# 3. GÖRSEL SIKIŞTIRMA API
@app.route("/api/img/compress", methods=["POST"])
def api_img_compress():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "compress")
    if not status["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403

    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    quality = int(request.form.get("quality", 70))
    
    try:
        file.seek(0, os.SEEK_END); orig_size = file.tell(); file.seek(0)
        img = Image.open(file.stream)
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        new_size = output.tell()
        
        encoded_img = base64.b64encode(output.getvalue()).decode('utf-8')
        size_str = f"{new_size/1024:.2f} KB" if new_size < 1024*1024 else f"{new_size/(1024*1024):.2f} MB"
        saving = int(((orig_size - new_size) / orig_size) * 100) if orig_size > 0 else 0
        
        increase_usage(email, "image", "compress")
        return jsonify({"success": True, "file": encoded_img, "new_size": size_str, "saving": saving})
    except Exception as e:
        return jsonify({"success": False, "message": "Hata."}), 500

# 4. FORMAT ÇEVİRME API
@app.route("/api/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "convert")
    if not status["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403

    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    target_ext = request.form.get("format", "jpeg").lower()
    
    format_map = {'jpeg':'JPEG', 'jpg':'JPEG', 'png':'PNG', 'webp':'WEBP', 'pdf':'PDF', 'ico':'ICO', 'bmp':'BMP', 'tiff':'TIFF', 'gif':'GIF'}
    pil_format = format_map.get(target_ext, 'JPEG')

    try:
        img = Image.open(file.stream)
        if pil_format in ['JPEG', 'PDF', 'BMP'] and img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == 'P': img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3] if len(img.split()) > 3 else None)
            img = bg
        
        if pil_format == 'ICO':
            if img.width > 256 or img.height > 256: img.thumbnail((256, 256))
            
        output = io.BytesIO()
        save_args = {"format": pil_format}
        if pil_format == 'JPEG': save_args["quality"] = 95
        
        img.save(output, **save_args)
        encoded_img = base64.b64encode(output.getvalue()).decode('utf-8')
        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": encoded_img})
    except Exception as e:
        return jsonify({"success": False, "message": "Format hatası."}), 500

# 5. PDF API
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
    if any(u["email"] == data["email"] for u in users): return jsonify({"status": "error", "message": "Kayıtlı"}), 409
    users.append({"email": data["email"], "end_date": data["end_date"], "tier": data.get("tier", "starter"), "usage_stats": {}})
    save_premium_users(users)
    return jsonify({"status": "ok", "message": "Kullanıcı Eklendi"})

@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def delete_premium_user(email):
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    users = load_premium_users()
    new_users = [u for u in users if u["email"] != email]
    save_premium_users(new_users)
    return jsonify({"status": "ok"})

@app.route("/api/admin/save_limits", methods=["POST"])
def save_tool_limits_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json()
    settings = load_settings()
    if "limits" in data: settings["limits"] = data["limits"]
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/api/admin/save_packages", methods=["POST"])
def save_packages_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    data = request.get_json()
    settings = load_settings()
    settings["packages"] = data.get("packages", {})
    save_settings(settings)
    return jsonify({"status": "ok"})

# --- GENEL ROUTE ---
@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    usage = 0
    if email != "guest":
        user = get_user_data_by_email(email)
        if user: usage = user.get("usage_stats", {}).get(subtool, 0)
    else:
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

@app.route("/get_settings", methods=["GET"])
def api_get_settings(): return jsonify(load_settings())

# --- SAYFALAR ---
@app.route("/")
def home(): return render_template("index.html")
@app.route("/remove-bg")
def remove_bg_page(): return render_template("background_remove.html")
@app.route("/vektor")
def vektor_page(): return render_template("vektor.html")
@app.route("/img/compress")
def img_compress_page(): return render_template("image_compress.html")
@app.route("/img/convert")
def img_convert_page(): return render_template("image_convert.html")
@app.route("/pdf/merge")
def pdf_merge_page(): return render_template("pdf_merge.html")
@app.route("/dashboard")
def dashboard_page():
    if not session.get("user_email"): return redirect("/")
    user = get_user_data_by_email(session.get("user_email"))
    return render_template("dashboard.html", user=user or {})

# AUTH & ADMIN PAGES
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
def user_logout(): session.clear(); return redirect("/")
@app.route("/admin_logout")
def admin_logout(): session.clear(); return redirect("/admin_login")
@app.route("/admin")
def admin_panel(): return render_template("admin.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)



