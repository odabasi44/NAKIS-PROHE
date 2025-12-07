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

# --- MEDIAPIPE (Yüz Hatlarını Çizmek İçin) ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
    print("✅ MediaPipe Yüklendi: Yüz hatları anatomik olarak çizilecek.")
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ UYARI: 'mediapipe' kütüphanesi eksik!")

# --- AI MODEL YÜKLEME (WHITE-BOX CARTOONIZATION) ---
gan_session = None
base_dir = os.path.dirname(os.path.abspath(__file__))

possible_model_paths = [
    "/app/models/whitebox_cartoon.onnx",
    "/data/models/whitebox_cartoon.onnx",
    os.path.join(base_dir, "models", "whitebox_cartoon.onnx"),
    os.path.join(base_dir, "whitebox_cartoon.onnx"),
]

print("--- AI MODEL ARAMA BAŞLADI (White-box) ---")
final_model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)

if final_model_path:
    try:
        gan_session = ort.InferenceSession(final_model_path, providers=["CPUExecutionProvider"])
        print(f"🚀 WHITE-BOX MODEL YÜKLENDİ: {final_model_path}")
    except Exception as e:
        print(f"⚠️ MODEL YÜKLENEMEDİ: {e}")
else:
    print("🚨 'whitebox_cartoon.onnx' bulunamadı.")

# --- U2NET MODELİ (Arka Plan Silme) ---
u2net_session = None
u2net_input_name = "input"
u2net_path = next((p for p in ["u2net.onnx", "models/u2net.onnx", "/app/models/u2net.onnx"] if os.path.exists(p)), None)

if u2net_path:
    try:
        u2net_session = ort.InferenceSession(u2net_path, providers=["CPUExecutionProvider"])
        u2net_input_name = u2net_session.get_inputs()[0].name
        print(f"✅ U2Net Modeli Yüklendi: {u2net_path}")
    except:
        pass

app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# --------------------------------------------------------------------
# AYARLAR – KULLANICI – OTURUM – LİMİT SİSTEMİ
# --------------------------------------------------------------------

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
                "split": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "compress": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999},
                "word2pdf": {"free": 2, "starter": 10, "pro": 50, "unlimited": 9999}
            },
            "vector": { "default": {"free": 0, "starter": 5, "pro": 50, "unlimited": 9999}},
            "file_size": {"free": 5, "starter": 10, "pro": 50, "unlimited": 100}
        },
        "packages": {},
        "site": {},
        "tool_status": {}
    }

    if not os.path.exists(SETTINGS_FILE):
        return default_settings

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except:
        return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


TIER_RESTRICTIONS = {
    "free": ["vector", "pdf_split", "word2pdf"],
    "starter": ["vector"],
    "pro": [],
    "unlimited": []
}

def load_premium_users():
    if not os.path.exists(PREMIUM_FILE):
        return []
    try:
        with open(PREMIUM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_premium_users(data):
    with open(PREMIUM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_user_data_by_email(email):
    users = load_premium_users()
    for u in users:
        if u.get("email", "").lower() == email.lower():
            return u
    return None

@app.before_request
def check_session_status():
    if request.path.startswith("/admin") and request.path != "/admin_login":
        if not session.get("admin_logged"):
            return redirect("/admin_login")

    if "user_email" in session:
        settings = load_settings()
        if session["user_email"] == settings["admin"]["email"]:
            return

        user = get_user_data_by_email(session["user_email"])
        if user:
            try:
                end_date = datetime.strptime(user.get("end_date"), "%Y-%m-%d")
                session["user_tier"] = user.get("tier", "free")
                session["is_premium"] = (end_date >= datetime.now())
                if not session["is_premium"]:
                    session["user_tier"] = "free"
            except:
                session["user_email"] = None
                session["is_premium"] = False
                session["user_tier"] = "free"
        else:
            session["user_email"] = None
            session["is_premium"] = False
            session["user_tier"] = "free"

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
            except:
                pass

    check_key = subtool if subtool else tool

    if check_key in TIER_RESTRICTIONS.get(user_tier, []):
        return {"allowed": False, "reason": "tier_restricted"}

    tool_limits = settings["limits"][tool][subtool][user_tier]
    current = 0

    if user_tier == "free":
        if "free_usage" not in session:
            session["free_usage"] = {}
        if tool not in session["free_usage"]:
            session["free_usage"][tool] = {}

        current = session["free_usage"][tool].get(subtool, 0)

    else:
        current = user_data.get("usage_stats", {}).get(subtool, 0)

    if current >= tool_limits:
        return {"allowed": False, "reason": "limit"}

    return {"allowed": True}

def increase_usage(email, tool, subtool):
    if email != "guest":
        users = load_premium_users()
        for u in users:
            if u["email"].lower() == email.lower():
                if "usage_stats" not in u:
                    u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                save_premium_users(users)
                return

    if "free_usage" not in session:
        session["free_usage"] = {}
    if tool not in session["free_usage"]:
        session["free_usage"][tool] = {}
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1

# --------------------------------------------------------------------
# ---------------------- GELİŞMİŞ VEKTÖR MOTORU ----------------------
# --------------------------------------------------------------------

class AdvancedVectorEngine:
    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        # SVG ve Preview için base tutucu
        self.vector_base = None

        if self.original_img is None:
            raise ValueError("Görüntü okunamadı.")

        # Saydamlık düzeltme
        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            alpha = self.original_img[:, :, 3]
            rgb = self.original_img[:, :, :3]
            white = np.ones_like(rgb) * 255
            self.img = (rgb * (alpha[:, :, None] / 255.0) + white * (1 - alpha[:, :, None] / 255.0)).astype(np.uint8)
        else:
            self.img = self.original_img[:, :, :3]

        # Wilcom Optimizasyonu: 1000px idealdir.
        h, w = self.img.shape[:2]
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.h, self.w = self.img.shape[:2]

    # ---------------- 1. XDoG (Extended Difference of Gaussians) ----------------
    def get_xdog_edges(self, img, gamma=0.97, phi=200, epsilon=0.01, k=1.6, sigma=0.8):
        """
        GERÇEK XDoG FORMÜLÜ:
        1. İki farklı Gaussian Blur al (g1, g2).
        2. Farkı ağırlıklı çıkar: D = g1 - (gamma * g2).
        3. Soft Thresholding (Tanh) uygula: E = 1 if D < eps else 1 + tanh(phi*(D-eps)).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 1. Gaussian Blur aşamaları
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
        
        # 2. Ağırlıklı Fark (XDoG Difference)
        diff = g1 - (gamma * g2)
        
        # 3. Soft Thresholding (Tanh fonksiyonu ile)
        # Nakış için "beyaz zemin üzerine siyah çizgi" istiyoruz.
        # Formülü tersine çevirerek siyah çizgileri (düşük değerleri) yakalıyoruz.
        
        # Kenarları vurgula
        edges = diff - epsilon
        edges = np.tanh(phi * edges)
        
        # 0-1 aralığına çek (Siyah çizgiler 0'a, beyaz zemin 1'e yaklaşır)
        edges = (edges + 1) / 2.0
        
        # Binary'e çevir (Keskin hatlar için)
        edges = (edges * 255).astype(np.uint8)
        _, binary_edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
        
        # Temizlik (Tekil pikselleri at)
        binary_edges = cv2.medianBlur(binary_edges, 3)
        
        return binary_edges # Siyah Çizgi (0), Beyaz Zemin (255)

    # ---------------- 2. BÖLGESEL MASKELEME (FACE vs BODY) ----------------
    def get_face_mask(self, img):
        """Yüz bölgesini (cilt + yüz ovali) beyaz maske olarak döndürür."""
        if not HAS_MEDIAPIPE: return None
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # Yüz ovalini al (Face Contour)
                contour = mp_face_mesh.FACEMESH_FACE_OVAL
                pts = []
                for source_idx, target_idx in contour:
                    pt = lm[source_idx]
                    pts.append([int(pt.x * w), int(pt.y * h)])
                
                if pts:
                    pts = np.array(pts)
                    hull = cv2.convexHull(pts)
                    cv2.fillPoly(mask, [hull], 255)
                    
                    # Maskeyi biraz yumuşat (Keskin geçiş olmasın)
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask

    # ---------------- 3. MEDIAPIPE YARDIMCI ÇİZGİLER (Complementary) ----------------
    def get_complementary_face_lines(self, img):
        """XDoG'un kaçırdığı kaş, göz ve dudak çizgilerini tamamlar."""
        if not HAS_MEDIAPIPE: return None
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255 # Beyaz zemin
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                
                # Yüz boyutuna göre dinamik kalınlık
                face_width = abs(lm[454].x - lm[234].x) * w
                th = max(1, int(face_width * 0.006)) # Biraz daha belirgin
                
                # Sadece kritik iç hatlar (Dış oval yok)
                lines = [
                    mp_face_mesh.FACEMESH_LIPS,
                    mp_face_mesh.FACEMESH_LEFT_EYEBROW, mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    mp_face_mesh.FACEMESH_LEFT_EYE, mp_face_mesh.FACEMESH_RIGHT_EYE,
                    mp_face_mesh.FACEMESH_NOSE
                ]
                for grp in lines:
                    for a,b in grp:
                        p1 = (int(lm[a].x*w), int(lm[a].y*h))
                        p2 = (int(lm[b].x*w), int(lm[b].y*h))
                        cv2.line(mask, p1, p2, 0, th) # Siyah çizgi
        return mask

    # ---------------- 4. QUANTIZATION ENGINE (K-Means) ----------------
    def quantize_region(self, img, k):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        return center[label.flatten()].reshape(img.shape)

    # -------------------- ERC ENGINE V3 (WAR MACHINE) --------------------
    def process_hybrid_cartoon(self, edge_thickness=2, color_count=16):
        """
        ERC V3 MİMARİSİ:
        1. Region Split: Yüz (High K, OilPaint) vs. Vücut (Low K, Bilateral).
        2. XDoG Edges: Gerçek karikatür çizgileri.
        3. Smart Blending: Yüz çizgileri XDoG'yi kesmez, tamamlar.
        """
        
        # --- ADIM 1: Bölgesel Maskeleme ---
        face_mask_gray = self.get_face_mask(self.img)
        
        if face_mask_gray is None:
            # Yüz bulunamazsa tüm resme standart uygula (Fallback)
            return self.process_basic_cartoon()

        # Maskeleri 3 kanala çevir (Renkli işlem için)
        face_mask = cv2.cvtColor(face_mask_gray, cv2.COLOR_GRAY2BGR)
        inv_mask = cv2.bitwise_not(face_mask)
        
        face_mask_f = face_mask.astype(float) / 255.0
        inv_mask_f = inv_mask.astype(float) / 255.0

        # --- ADIM 2: YÜZ İŞLEME (Oil Painting + High K) ---
        # Sadece yüze özel ışık ve renk
        face_part = self.img.copy()
        try:
            # Oil Painting sadece yüze! (Giysiyi bozmaz)
            face_part = cv2.xphoto.oilPainting(face_part, 5, 1)
        except:
            face_part = cv2.bilateralFilter(face_part, 5, 75, 75)
            
        # Yüz için yüksek renk sayısı (Ten tonu bozulmasın diye: 16-24)
        k_face = max(16, min(color_count + 8, 24))
        face_quant = self.quantize_region(face_part, k_face)

        # --- ADIM 3: VÜCUT/ARKA PLAN İŞLEME (Bilateral + Low K) ---
        body_part = self.img.copy()
        # Vücut için güçlü Bilateral (Doku yok etme)
        body_part = cv2.bilateralFilter(body_part, 9, 100, 100)
        
        # Vücut için düşük renk sayısı (Giysi blokları net olsun: 8-12)
        k_body = max(8, min(color_count - 4, 12))
        body_quant = self.quantize_region(body_part, k_body)

        # --- ADIM 4: MULTI-RESOLUTION BLENDING ---
        # İki farklı işlenmiş görüntüyü maske ile birleştir
        # (face_quant * mask) + (body_quant * inv_mask)
        final_color = (face_quant * face_mask_f + body_quant * inv_mask_f).astype(np.uint8)

        # --- ADIM 5: ÇİZGİ MOTORU (XDoG + FaceMesh Fusion) ---
        
        # A) XDoG Çizgileri (Siyah çizgi, Beyaz zemin)
        xdog_edges = self.get_xdog_edges(self.img, gamma=0.96, phi=150, epsilon=0.005)
        
        # B) FaceMesh Tamamlayıcı Çizgiler (Siyah çizgi, Beyaz zemin)
        fm_lines = self.get_complementary_face_lines(self.img)
        
        # C) Smart Fusion (Anti-Ghosting)
        # Siyah çizgileri birleştirmek için 'bitwise_and' (0 yutan eleman)
        # VEYA pixel wise multiplication (Normalize edip)
        # En temizi: Her ikisi de Beyaz zemin/Siyah çizgi olduğu için 'minimum' operatörü (darkest pixel wins)
        combined_edges = cv2.min(xdog_edges, fm_lines)

        # Kenar Kalınlaştırma (Opsiyonel)
        if edge_thickness > 1:
            # Siyah çizgileri kalınlaştırmak için ERODE kullanılır (Beyazı yer)
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            combined_edges = cv2.erode(combined_edges, kernel, iterations=1)

        # --- ADIM 6: FİNAL OVERLAY ---
        # Renkli resim üzerine Siyah çizgileri bas
        # Maskeleme: combined_edges siyah (0) ise Siyah yap, yoksa renkli kalsın.
        
        edges_bgr = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
        
        # Anti-aliasing (Nakış için sert değil, yumuşak geçişli çizgi iyidir)
        edges_bgr = cv2.GaussianBlur(edges_bgr, (3,3), 0)
        
        # Overlay
        final_result = cv2.bitwise_and(final_color, edges_bgr)
        
        # SVG için temiz kopya
        self.vector_base = final_result.copy()

        return final_result

    # ---------------- ÖZEL FONKSİYON: Gözleri Canlandır ----------------
    def enhance_facial_features(self, img):
        if not HAS_MEDIAPIPE: return img
        out = img.copy()
        h, w = out.shape[:2]
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                mask = np.zeros((h, w), dtype=np.uint8)
                for indices in [left_eye, right_eye]:
                    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices])
                    cv2.fillPoly(mask, [pts], 255)
                mask = cv2.GaussianBlur(mask, (3, 3), 1)
                hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                v_ch = np.where(mask > 0, np.clip(v_ch + 40, 0, 255), v_ch) 
                s_ch = np.where(mask > 0, np.clip(s_ch - 30, 0, 255), s_ch)
                final_hsv = cv2.merge((h_ch, s_ch, v_ch))
                out = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return out

    # -------------------- WHITEBOX MODEL --------------------
    def process_with_whitebox_cartoon(self, img_input):
        if gan_session is None: return img_input
        try:
            img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512))
            img_norm = (img_resized / 127.5) - 1.0
            img_norm = np.transpose(img_norm, (2, 0, 1)).astype(np.float32)
            img_norm = np.expand_dims(img_norm, 0)
            outputs = gan_session.run(None, {'input': img_norm})
            cartoon = outputs[0][0]
            cartoon = np.transpose(cartoon, (1, 2, 0))
            cartoon = (cartoon + 1.0) * 127.5
            cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
            cartoon = cv2.resize(cartoon, (self.w, self.h))
            return cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
        except: return img_input

    # -------------------- BASIC CARTOON (FALLBACK) --------------------
    def process_basic_cartoon(self):
        # Fallback için XDoG kullan (Daha iyi sonuç verir)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = self.get_xdog_edges(self.img)
        color = cv2.bilateralFilter(self.img, 9, 75, 75)
        quant = self.quantize_region(color, 12)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(quant, edges_bgr)
        self.vector_base = result
        return result

    # -------------------- SKETCH STYLE --------------------
    def process_sketch_style(self):
        # Sketch için XDoG mükemmeldir.
        dog = self.get_xdog_edges(self.img, phi=250) # Biraz daha sert çizgiler
        res = cv2.cvtColor(dog, cv2.COLOR_GRAY2BGR)
        self.vector_base = res
        return res

    # -------------------- ARTISTIC STYLES --------------------
    def process_artistic_style(self, style="cartoon", options=None):
        if options is None: options = {}
        edge = options.get("edge_thickness", 2)
        colors = options.get("color_count", 16)
        
        base_result = self.process_hybrid_cartoon(edge_thickness=edge, color_count=colors)
        effect_result = base_result.copy()
        
        if style == "comic": effect_result = self.apply_comic_effect(effect_result)
        elif style == "anime": effect_result = self.apply_anime_effect(effect_result)
        elif style == "painting": effect_result = self.process_painting_style(effect_result)
        
        return effect_result

    # -------------------- STYLE EFFECTS --------------------
    def apply_comic_effect(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_anime_effect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_painting_style(self, img):
        try:
            return cv2.xphoto.oilPainting(img, 7, 1)
        except:
            return cv2.bilateralFilter(img, 9, 75, 75)

    # -------------------- SVG OUTPUT --------------------
    def generate_artistic_svg(self, num_colors=16, simplify_factor=0.007, stroke_width=1):
        proc = self.vector_base if self.vector_base is not None else self.img
        data = np.float32(proc).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = max(2, num_colors)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        quantized = center[label.flatten()].reshape(proc.shape)
        
        h, w = quantized.shape[:2]
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        bg_color = self.get_dominant_color(quantized)
        r, g, b = int(bg_color[2]), int(bg_color[1]), int(bg_color[0])
        svg += f'<rect width="{w}" height="{h}" fill="#{r:02x}{g:02x}{b:02x}"/>'
        
        colors = np.unique(quantized.reshape(-1, 3), axis=0)
        for color in colors:
            if np.array_equal(color, bg_color): continue
            b, g, r = color.astype(int)
            mask = cv2.inRange(quantized, color, color)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: continue
                epsilon = simplify_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                points = approx.reshape(-1, 2)
                if len(points) < 3: continue
                path_data = f"M {points[0][0]} {points[0][1]}"
                for point in points[1:]: path_data += f" L {point[0]} {point[1]}"
                path_data += " Z"
                stroke_r, stroke_g, stroke_b = max(0, r-40), max(0, g-40), max(0, b-40)
                svg += f'<path d="{path_data}" fill="#{r:02x}{g:02x}{b:02x}" stroke="#{stroke_r:02x}{stroke_g:02x}{stroke_b:02x}" stroke-width="{stroke_width}"/>'
        svg += "</svg>"
        return svg

    def get_dominant_color(self, img, k=1):
        pixels = np.float32(img).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        unique, counts = np.unique(labels, return_counts=True)
        return centers[unique[np.argmax(counts)]]

# ------------------------------------------------------------------------
# ---------------------------- API ENDPOINTS ------------------------------
# ------------------------------------------------------------------------

@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "vector", "default")["allowed"]:
        return jsonify({"success": False, "reason": "limit"}), 403

    if "image" not in request.files:
        return jsonify({"success": False}), 400

    file = request.files["image"]
    method = request.form.get("method", "cartoon")
    style = request.form.get("style", "cartoon")
    edge_thickness = int(request.form.get("edge_thickness", 2))
    color_count = int(request.form.get("color_count", 16))

    try:
        engine = AdvancedVectorEngine(file)
        
        options = {
            "edge_thickness": edge_thickness,
            "color_count": color_count,
            "simplify": 0.007 # Default güncellendi
        }
        
        if method == "outline":
            engine.img = engine.process_sketch_style()
        else:
            # self.img = EFEKTLİ (Preview için)
            # self.vector_base = TEMİZ (SVG için) otomatik dolar
            engine.img = engine.process_artistic_style(style=style, options=options)
        
        # SVG oluşturulurken self.vector_base kullanılacak
        svg = engine.generate_artistic_svg(
            num_colors=color_count,
            simplify_factor=0.007,
            stroke_width=max(1, edge_thickness // 2)
        )
        
        svg_b64 = base64.b64encode(svg.encode()).decode()
        
        # Preview görseli (Efektli olan self.img'den üretilir)
        _, buf = cv2.imencode(".png", engine.img)
        preview_b64 = base64.b64encode(buf).decode()

        increase_usage(email, "vector", "default")

        return jsonify({
            "success": True,
            "file": svg_b64,
            "preview_img": preview_b64,
            "info": {
                "style": style,
                "colors": color_count,
                "edges": edge_thickness
            }
        })

    except Exception as e:
        print("VECTOR ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

# --------------------------------------------------------------------
# ------------------------- REMOVE BG --------------------------------
# --------------------------------------------------------------------

@app.route("/api/remove_bg", methods=["POST"])
def api_remove_bg():
    if not u2net_session:
        return jsonify({"success": False, "reason": "AI Model Yok"}), 503

    if "image" not in request.files:
        return jsonify({"success": False}), 400

    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "remove_bg")["allowed"]:
        return jsonify({"success": False}), 403

    try:
        original = Image.open(request.files["image"]).convert("RGB")
        small = original.resize((320, 320))

        inp = np.transpose(np.array(small).astype(np.float32) / 255.0, (2, 0, 1))
        inp = np.expand_dims(inp, 0)

        out = u2net_session.run(None, {u2net_input_name: inp})[0].squeeze()
        mask = cv2.resize(out, original.size)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        rgba = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, "PNG")

        increase_usage(email, "image", "remove_bg")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------------------
# ----------- COMPRESS / CONVERT / PDF MERGE --------------------------
# ---------------------------------------------------------------------

@app.route("/api/img/compress", methods=["POST"])
def api_img_compress():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "compress")["allowed"]:
        return jsonify({"success": False}), 403

    if "image" not in request.files:
        return jsonify({"success": False}), 400

    quality = int(request.form.get("quality", 70))

    try:
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)

        increase_usage(email, "image", "compress")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "convert")["allowed"]:
        return jsonify({"success": False}), 403

    if "image" not in request.files:
        return jsonify({"success": False}), 400

    fmt = request.form.get("format", "jpeg").lower()
    fmt = {"jpg":"JPEG","jpeg":"JPEG","png":"PNG","webp":"WEBP","pdf":"PDF"}.get(fmt, "JPEG")

    try:
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, fmt)

        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "pdf", "merge")["allowed"]:
        return jsonify({"success": False}), 403

    if "pdf_files" not in request.files:
        return jsonify({"success": False}), 400

    files = request.files.getlist("pdf_files")

    try:
        merger = PdfMerger()
        for f in files:
            merger.append(io.BytesIO(f.read()))
        out = io.BytesIO()
        merger.write(out)
        merger.close()

        increase_usage(email, "pdf", "merge")
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------------------------------------------------------------------
# -------------------------- ADMIN / LOGIN -----------------------------
# ---------------------------------------------------------------------

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login_route():
    if request.method == "GET":
        return render_template("admin_login.html")

    data = request.get_json()
    settings = load_settings()

    if data.get("email") == settings["admin"]["email"] and data.get("password") == settings["admin"]["password"]:
        session["admin_logged"] = True
        session["user_email"] = data["email"]
        session["user_tier"] = "unlimited"
        session["is_premium"] = True
        return jsonify({"status":"ok"})

    return jsonify({"status":"error"}), 401

@app.route("/user_login", methods=["POST"])
def user_login_endpoint():
    data = request.get_json()
    email = data.get("email")
    settings = load_settings()

    if email == settings["admin"]["email"]:
        return jsonify({"status":"admin"})

    user = get_user_data_by_email(email)
    if not user:
        return jsonify({"status":"not_found"})

    try:
        end = datetime.strptime(user.get("end_date"), "%Y-%m-%d")
        if end >= datetime.now():
            session["user_email"] = email
            session["is_premium"] = True
            session["user_tier"] = user.get("tier", "starter")
            return jsonify({"status":"premium","tier":session["user_tier"]})
        else:
            return jsonify({"status":"expired"})
    except:
        return jsonify({"status":"error"})

@app.route("/api/admin/users", methods=["GET"])
def get_users_api():
    if not session.get("admin_logged"):
        return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"):
        return jsonify({}), 403
    data = request.get_json()
    users = load_premium_users()
    users.append({
        "email": data["email"],
        "end_date": data["end_date"],
        "tier": data["tier"],
        "usage_stats": {}
    })
    save_premium_users(users)
    return jsonify({"status":"ok"})

@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"):
        return jsonify({}), 403

    users = [u for u in load_premium_users() if u["email"] != email]
    save_premium_users(users)
    return jsonify({"status":"ok"})

@app.route("/api/check_tool_status/<tool>/<subtool>")
def status_api(tool, subtool):
    return jsonify(check_user_status(session.get("user_email","guest"), tool, subtool))

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ----------------------------------------------------------------------
# ----------------------------- SAYFALAR --------------------------------
# ----------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/<page>")
def render_page(page):
    if os.path.exists(f"templates/{page}.html"):
        return render_template(f"{page}.html")
    return redirect("/")

# --------------------------------------------------------------------
# ----------------------------- RUN APP ------------------------------
# --------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

