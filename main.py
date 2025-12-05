# main.py dosyasının en üstüne, diğer importların yanına ekleyin:
import cv2
import random
import os
import io
import json
import base64
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

# --- AYARLAR SİSTEMİ (GÜNCELLENMİŞ) ---
def load_settings():
    # Varsayılan Paket ve Limit Ayarları
    default_settings = {
        "admin": {"email": "admin@botlab.com", "password": "admin"},
        "limits": {
            # GÖRSEL ARAÇLAR
            "image": {
                "remove_bg": {"free": 2, "starter": 20, "pro": 200, "unlimited": 9999},
                "compress": {"free": 5, "starter": 50, "pro": 500, "unlimited": 9999},
                "convert": {"free": 5, "starter": 50, "pro": 500, "unlimited": 9999}
            },
            # PDF ARAÇLARI
            "pdf": {
                "merge": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "split": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "compress": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "word2pdf": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}
            },
            # VEKTÖR & AI
            "vector": {
                "default": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}
            },
            # YENİ ÜRETİCİLER (İndirme Limiti)
            # Free pakete '0' vererek indirmeyi engelleyeceğiz ama sayfayı açacağız.
            "generator": {
                "qr": {"free": 0, "starter": 10, "pro": 100, "unlimited": 9999},
                "logo": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}
            },
            # Dosya Boyutu Limitleri (MB)
            "file_size": {
                "free": 5, "starter": 10, "pro": 50, "unlimited": 100
            }
        },
        "packages": {}, # Paket tanımları (daha önce eklediklerimiz korunur)
        "site": {},
        "tool_status": {}
    }
    
    if not os.path.exists("settings.json"):
        return default_settings
        
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            # Eksik alanları varsayılanlarla doldur (migration)
            if "limits" not in data: data["limits"] = default_settings["limits"]
            
            # Yeni eklenen araçları kontrol et ve eksikse ekle
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

# Hangi araçlar hangi pakette tamamen yasaklı (Görünür ama kilitli)
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
        # Admin kontrolü
        settings = load_settings()
        if session["user_email"] == settings["admin"]["email"]: return

        user = get_user_data_by_email(session["user_email"])
        if user:
             try:
                end_date = datetime.strptime(user.get("end_date", "1970-01-01"), "%Y-%m-%d")
                # Paket bilgisini session'a işle
                session["user_tier"] = user.get("tier", "free")
                session["is_premium"] = (end_date >= datetime.now())
                # Süre bittiyse tier'ı free'ye çek
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
    
    # 1. Kullanıcı Paketini Belirle
    user_tier = "free"
    user_data = None
    
    if email != "guest":
        user_data = get_user_data_by_email(email)
        if user_data:
            try:
                end_date = datetime.strptime(user_data.get("end_date"), "%Y-%m-%d")
                if end_date >= datetime.now():
                    user_tier = user_data.get("tier", "free")
            except: pass
            
    # 2. Kısıtlı Araç Kontrolü (Tier Restriction)
    check_key = subtool if subtool else tool
    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
         return {"allowed": False, "reason": "tier_restricted", "tier": user_tier, "left": 0, "premium": (user_tier != "free")}

    # 3. Bakım Modu
    tool_status = settings.get("tool_status", {}).get(subtool, {})
    if tool_status.get("maintenance", False):
        return {"allowed": False, "reason": "maintenance", "left": 0, "premium": (user_tier != "free")}

    # 4. Limitleri Çek
    tool_limits = settings.get("limits", {}).get(tool, {})
    limit = tool_limits.get(subtool, {}).get(user_tier, 0)
    
    # 5. Kullanım Miktarını Bul
    current_usage = 0
    if user_tier == "free":
        # Session'dan oku
        if "free_usage" not in session: 
            session["free_usage"] = {}
            session.modified = True
        if tool not in session["free_usage"]: 
            session["free_usage"][tool] = {}
            session.modified = True
        current_usage = session["free_usage"][tool].get(subtool, 0)
    else:
        # DB'den oku
        current_usage = user_data.get("usage_stats", {}).get(subtool, 0)

    left = limit - current_usage
    
    if left <= 0:
        reason = "free_limit_full" if user_tier == "free" else "premium_limit_full"
        return {"allowed": False, "reason": reason, "left": 0, "premium": (user_tier != "free"), "tier": user_tier}

    return {"allowed": True, "reason": "", "premium": (user_tier != "free"), "left": left, "tier": user_tier}

def increase_usage(email, tool, subtool):
    # Premium kullanıcı DB'ye yaz
    if email != "guest":
        users = load_premium_users()
        for u in users:
            if u.get("email", "").lower() == email.lower():
                if "usage_stats" not in u: u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                save_premium_users(users)
                return

    # Misafir Session'a yaz (VE KAYDET)
    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    
    current = session["free_usage"][tool].get(subtool, 0)
    session["free_usage"][tool][subtool] = current + 1
    session.modified = True # <--- LIMITIN DÜŞMESİ İÇİN ŞART

# --- VEKTÖR MOTORU (GELİŞMİŞ-v.2-ODABASI) ---
class VectorEngine:
    def __init__(self, image_stream):
        # Dosyayı OpenCV formatına çevir
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Görüntü okunamadı")
        self.original_h, self.original_w = self.img.shape[:2]

    def process_enhanced_quality(self, mode='normal'):
        """
        Wilcom ve Nakış için özel optimize edilmiş işleme motoru.
        pyrMeanShiftFiltering kullanarak 'Yağlı Boya' etkisi yaratır, 
        bu da gereksiz dikişleri (kırışıklıkları) yok eder.
        """
        
        # 1. Gürültü Azaltma ve Düzleştirme (En Önemli Adım)
        # sp: Mekansal pencere yarıçapı (Büyüdükçe detaylar azalır, bloklar büyür)
        # sr: Renk penceresi yarıçapı (Benzer renkleri birleştirir)
        if mode == 'cartoon':
            # Daha agresif düzleştirme (Büyük bloklar)
            self.img = cv2.pyrMeanShiftFiltering(self.img, sp=25, sr=40)
        else:
            # Dengeli düzleştirme (Detayları korur ama paraziti siler)
            self.img = cv2.pyrMeanShiftFiltering(self.img, sp=15, sr=30)

        # 2. Renk Sayısını Optimize Et (K-Means)
        # Nakış makinesi için renk sayısını limitlemek iplik değişimini azaltır.
        k = 16 if mode == 'normal' else 8
        data = np.float32(self.img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        self.img = center[label.flatten()].reshape((self.img.shape))

        # 3. Kenar Yumuşatma (Median Blur)
        # Vektör kenarlarındaki tırtıkları alır
        self.img = cv2.medianBlur(self.img, 3)

    def process_outline(self):
        # Outline modu için özel işlem
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0) # Hafif bulanıklık keskin kenar verir
        edges = cv2.Canny(gray, 75, 150)
        
        # Çizgileri birleştir (Dilation)
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        self.img = cv2.bitwise_not(dilated)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def generate_svg(self, smoothness='high'):
        """
        Vektör verisini oluşturur.
        smoothness: 'high' ise köşeleri daha çok yuvarlar (Wilcom için iyi).
        """
        pixels = self.img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        # SVG Başlığı
        svg_output = f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{self.original_w}" height="{self.original_h}" viewBox="0 0 {self.original_w} {self.original_h}">'
        
        # Epsilon değeri: Eğri hassasiyeti. 
        # Değer ne kadar BÜYÜK olursa çizgi o kadar DÜZ/BASİT olur.
        # Değer ne kadar KÜÇÜK olursa çizgi o kadar DETAYLI/TIRTIKLI olur.
        epsilon_factor = 0.001 # Varsayılan (High Detail)
        if smoothness == 'ultra': epsilon_factor = 0.003 # Daha yuvarlak hatlar
        elif smoothness == 'medium': epsilon_factor = 0.0005 # Daha keskin hatlar

        for color in unique_colors:
            mask = cv2.inRange(self.img, color, color)
            # RETR_EXTERNAL yerine RETR_TREE kullanıyoruz ki iç boşlukları da alsın
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours: continue
            
            b, g, r = color
            # Beyaz veya beyaza çok yakın arka planı atla (Opsiyonel)
            if r > 250 and g > 250 and b > 250: continue
            
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            
            path_data = ""
            for i, cnt in enumerate(contours):
                # Alan kontrolü: Çok küçük noktaları (gürültü) vektöre çevirme
                if cv2.contourArea(cnt) < 20: continue 

                epsilon = epsilon_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3: continue
                
                points = approx.reshape(-1, 2)
                path_data += f"M {points[0][0]} {points[0][1]} "
                for p in points[1:]:
                    path_data += f"L {p[0]} {p[1]} "
                path_data += "Z " 
            
            if path_data:
                svg_output += f'<path d="{path_data}" fill="{hex_color}" stroke="none" />'
        
        svg_output += '</svg>'
        return svg_output

# --- API ENDPOINTLERİ ---
# --- API ENDPOINTLERİ KISMINA EKLEYİN ---

# --- API GÜNCELLEMESİ ---
@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    status = check_user_status(email, "vector", "default")
    
    if not status["allowed"]:
        return jsonify({"success": False, "reason": "limit"}), 403

    if "image" not in request.files:
        return jsonify({"success": False}), 400
    
    file = request.files["image"]
    method = request.form.get("method", "normal") 
    quality = request.form.get("quality", "high") 

    try:
        engine = VectorEngine(file)
        
        # 1. İşleme (Enhanced Mode)
        if method == "outline":
            engine.process_outline()
        elif method == "cartoon":
            engine.process_enhanced_quality(mode='cartoon')
        else: # normal
            engine.process_enhanced_quality(mode='normal')
            
        # 2. SVG Oluştur (Wilcom için 'high' veya 'ultra' smoothness önerilir)
        smoothness = 'high'
        if quality == 'ultra': smoothness = 'ultra'
        elif quality == 'medium': smoothness = 'medium'
        
        svg_string = engine.generate_svg(smoothness=smoothness)
        
        # 3. Çıktıları Hazırla
        # a. Vektör (SVG)
        encoded_svg = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
        
        # b. Önizleme PNG (İşlenmiş, pürüzsüz halini geri gönderiyoruz)
        # Kullanıcı Wilcom'da "Auto Trace" yapacaksa bu temiz PNG çok işine yarar.
        _, buffer = cv2.imencode('.png', engine.img)
        encoded_png = base64.b64encode(buffer).decode('utf-8')
        
        increase_usage(email, "vector", "default")
        
        # Hem SVG hem PNG döndürüyoruz
        return jsonify({
            "success": True, 
            "file": encoded_svg,      # İndirilebilir Vektör
            "preview_img": encoded_png # Ekranda görünecek temizlenmiş resim
        })

    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/admin/save_packages", methods=["POST"])
def save_packages_api():
    if not session.get("admin_logged"): return jsonify({"status": "error"}), 403
    
    data = request.get_json()
    new_packages = data.get("packages", {})
    
    settings = load_settings()
    
    # Mevcut "free" paketini koruyalım, diğerlerini güncelleyelim
    current_packages = settings.get("packages", {})
    
    # Gelen veriyi işle
    for tier, info in new_packages.items():
        if tier in ["starter", "pro", "unlimited"]:
            # Eğer settings'de yoksa oluştur, varsa güncelle
            if tier not in current_packages: current_packages[tier] = {}
            current_packages[tier].update(info)
            
    settings["packages"] = current_packages
    save_settings(settings)
    
    return jsonify({"status": "ok", "message": "Paketler güncellendi."})

@app.route("/api/check_tool_status/<tool>/<subtool>", methods=["GET"])
def check_tool_status_endpoint(tool, subtool):
    email = session.get("user_email", "guest")
    status = check_user_status(email, tool, subtool)
    
    # Kullanım miktarını döndür
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

# --- MODEL YÜKLEME ---
u2net_session = None
model_input_name = "input"
possible_paths = ["/data/ai-models/u2net.onnx", "u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"]
found_path = None
for path in possible_paths:
    if os.path.exists(path): found_path = path; break
if found_path:
    try:
        u2net_session = ort.InferenceSession(found_path, providers=["CPUExecutionProvider"])
        model_input_name = u2net_session.get_inputs()[0].name
        print(f"Model OK: {model_input_name}")
    except: pass

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
    
    # 2. Dosya Boyutu Kontrolü (Pakete göre)
    file.seek(0, os.SEEK_END); size = file.tell(); file.seek(0)
    
    settings = load_settings()
    tier = status.get("tier", "free")
    limit_mb = settings["limits"]["file_size"].get(tier, 5) # Varsayılan 5MB
    
    if size > limit_mb * 1024 * 1024:
         return jsonify({"success": False, "reason": "file_size_limit", "message": f"Dosya limiti: {limit_mb}MB"}), 413
    
    try:
        img = Image.open(file.stream)
        ow, oh = img.size
        output = u2net_session.run(None, {model_input_name: preprocess_bg(img)})[0]
        mask = postprocess_bg(output, (ow, oh))
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(buf, format="PNG") # Maskeyi döndür (geçici) veya full işlem
        
        # Orijinal işlem
        rgba = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        out_buf = io.BytesIO()
        Image.fromarray(rgba).save(out_buf, format="PNG")

        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(out_buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- DİĞER ROTALAR ---
@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    # ... (PDF kodu aynı, increase_usage çağırıyor)
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

# --- SAYFALAR & AUTH ---
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
    data = request.get_json()
    users = load_premium_users()
    if any(u["email"] == data["email"] for u in users): return jsonify({"status": "error", "message": "Kayıtlı"}), 409
    users.append({"email": data["email"], "end_date": data["end_date"], "tier": data.get("tier", "starter"), "usage_stats": {}})
    save_premium_users(users)
    return jsonify({"status": "ok"})

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
    # Frontend'den gelen tam limit yapısını kaydet
    if "limits" in data: settings["limits"] = data["limits"]
    save_settings(settings)
    return jsonify({"status": "ok"})

@app.route("/get_settings", methods=["GET"])
def api_get_settings():
    return jsonify(load_settings())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)





