import os
import io
import json
import base64
import math
import random
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
import onnxruntime as ort

# ==============================================================================
# 1. KÜTÜPHANE KONTROLLERİ VE AYARLAR
# ==============================================================================

# --- MediaPipe ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
    print("✅ MediaPipe Yüklendi.")
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ UYARI: 'mediapipe' eksik! Yüz hatları çalışmayabilir.")

# --- PyEmbroidery (Nakış) ---
try:
    import pyembroidery
    HAS_EMBROIDERY = True
    print("✅ PyEmbroidery Yüklendi (Nakış Modülü Aktif).")
except ImportError:
    HAS_EMBROIDERY = False
    print("⚠️ UYARI: 'pyembroidery' eksik! Nakış çıktısı alınamaz.")

# --- Flask Uygulaması ---
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = "BOTLAB_SECRET_123"  # Güvenlik için değiştirin
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB Limit

# --- AI Model Yolları ---
gan_session = None
u2net_session = None
base_dir = os.path.dirname(os.path.abspath(__file__))

# Whitebox Cartoon Model
possible_wb = [
    "/app/models/whitebox_cartoon.onnx",
    "models/whitebox_cartoon.onnx",
    "whitebox_cartoon.onnx",
    os.path.join(base_dir, "models", "whitebox_cartoon.onnx")
]
wb_path = next((p for p in possible_wb if os.path.exists(p)), None)
if wb_path:
    try:
        gan_session = ort.InferenceSession(wb_path, providers=["CPUExecutionProvider"])
        print(f"🚀 Whitebox Model Yüklendi: {wb_path}")
    except Exception as e:
        print(f"⚠️ Whitebox Model Hatası: {e}")

# U2Net (Background Remove) Model
possible_u2 = [
    "/app/models/u2net.onnx",
    "models/u2net.onnx",
    "u2net.onnx",
    os.path.join(base_dir, "models", "u2net.onnx")
]
u2_path = next((p for p in possible_u2 if os.path.exists(p)), None)
u2net_input_name = "input"
if u2_path:
    try:
        u2net_session = ort.InferenceSession(u2_path, providers=["CPUExecutionProvider"])
        u2net_input_name = u2net_session.get_inputs()[0].name
        print(f"✅ U2Net Model Yüklendi: {u2_path}")
    except Exception as e:
        print(f"⚠️ U2Net Model Hatası: {e}")


# ==============================================================================
# 2. VERİ YÖNETİMİ (AYARLAR & KULLANICILAR)
# ==============================================================================

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
            "vector": { "default": {"free": 5, "starter": 50, "pro": 200, "unlimited": 9999}},
            "file_size": {"free": 5, "starter": 10, "pro": 50, "unlimited": 100}
        }
    }
    if not os.path.exists(SETTINGS_FILE):
        return default_settings
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default_settings

def save_settings(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

TIER_RESTRICTIONS = {
    "free": [], 
    "starter": [],
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
    # Admin paneli koruması
    if request.path.startswith("/admin") and request.path != "/admin_login":
        if not session.get("admin_logged"):
            return redirect("/admin_login")

    # Kullanıcı oturum kontrolü
    if "user_email" in session:
        settings = load_settings()
        if session["user_email"] == settings["admin"]["email"]:
            return # Admin her şeye erişir

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

    # Limit Kontrolü
    try:
        tool_limits = settings["limits"][tool][subtool][user_tier]
    except:
        tool_limits = 5 # Varsayılan limit

    current = 0
    if user_tier == "free":
        if "free_usage" not in session: session["free_usage"] = {}
        if tool not in session["free_usage"]: session["free_usage"][tool] = {}
        current = session["free_usage"][tool].get(subtool, 0)
    else:
        current = user_data.get("usage_stats", {}).get(subtool, 0)

    left = max(0, tool_limits - current)
    return {"allowed": current < tool_limits, "reason": "limit" if current >= tool_limits else None, "left": left, "premium": session.get("is_premium", False)}

def increase_usage(email, tool, subtool):
    if email != "guest":
        users = load_premium_users()
        for u in users:
            if u["email"].lower() == email.lower():
                if "usage_stats" not in u: u["usage_stats"] = {}
                u["usage_stats"][subtool] = u["usage_stats"].get(subtool, 0) + 1
                save_premium_users(users)
                return

    if "free_usage" not in session: session["free_usage"] = {}
    if tool not in session["free_usage"]: session["free_usage"][tool] = {}
    session["free_usage"][tool][subtool] = session["free_usage"][tool].get(subtool, 0) + 1


# ==============================================================================
# 3. MOTOR: ERC V4 PRO MAX (VEKTÖR İŞLEME)
# ==============================================================================

class AdvancedVectorEngine:
    """
    ERC V4 PRO MAX (OPTIMIZED)
    - XDoG & FDoG Hybrid
    - Smart Region Masking (Yüz, Saç, Vücut ayrımı)
    - Wilcom Ready Cleanup
    """

    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
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

    # --- YARDIMCI: Yüz Maskesi ---
    def get_face_mask(self, img):
        if not HAS_MEDIAPIPE: return None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                contour = mp_face_mesh.FACEMESH_FACE_OVAL
                pts = []
                for source_idx, target_idx in contour:
                    pt = lm[source_idx]
                    pts.append([int(pt.x * w), int(pt.y * h)])
                
                if pts:
                    pts = np.array(pts)
                    hull = cv2.convexHull(pts)
                    cv2.fillPoly(mask, [hull], 255)
                    # Maskeyi yumuşat
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    # --- 1. HAIR FLOW (Optimize) ---
    def get_hair_flow(self, img, face_mask):
        """Saç yönünü bulur ama YÜZE BULAŞMASINI ENGELLER."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Hız için adım sayısı 30
        responses = []
        for theta in range(0, 180, 30):
            kernel = cv2.getGaborKernel((15, 15), 4.0, np.deg2rad(theta), 10.0, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            responses.append(resp)

        stack = np.stack(responses, axis=-1)
        max_filter = np.max(stack, axis=-1)
        max_filter = cv2.normalize(max_filter, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Sadece çok belirgin çizgileri al
        _, hair_lines = cv2.threshold(max_filter, 200, 255, cv2.THRESH_BINARY)
        hair_lines = cv2.ximgproc.thinning(hair_lines)
        
        # SİYAH çizgi (0), BEYAZ zemin (255) formatına çevir
        hair_lines_inv = cv2.bitwise_not(hair_lines)

        # MASK ELEME: Yüz bölgesindeki saç çizgilerini sil (Beyaz yap)
        if face_mask is not None:
            hair_lines_inv = cv2.bitwise_or(hair_lines_inv, face_mask)

        return hair_lines_inv

    # --- 2. CLOTH WRINKLES (Optimize & Maskeli) ---
    def get_cloth_wrinkles(self, img, face_mask):
        """Kırışıklıkları bulur ama YÜZE BULAŞMASINI ENGELLER."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g1 = cv2.GaussianBlur(gray, (0,0), 1.2)
        g2 = cv2.GaussianBlur(gray, (0,0), 2.4)
        diff = cv2.subtract(g1, g2)
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Daha yüksek eşik (sadece derin kırışıklıklar)
        _, wr = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
        wr = cv2.ximgproc.thinning(wr)
        
        # SİYAH çizgi, BEYAZ zemin
        wr_inv = cv2.bitwise_not(wr)

        # MASK ELEME: Yüz bölgesindeki kırışıklıkları sil
        if face_mask is not None:
            wr_inv = cv2.bitwise_or(wr_inv, face_mask)

        return wr_inv

    # --- 3. XDoG ---
    def get_xdog(self, img, gamma=0.97, phi=200, epsilon=0.01):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.GaussianBlur(gray, (0, 0), 0.8)
        g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)
        diff = g1 - gamma * g2
        edges = np.tanh(phi * (diff - epsilon))
        edges = (edges + 1) / 2.0
        edges = (edges * 255).astype(np.uint8)
        _, bin_edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
        return cv2.medianBlur(bin_edges, 3) # Siyah çizgi, Beyaz zemin

    # --- 4. FDoG (Hafifletilmiş) ---
    def get_fdog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy) # Daha hızlı
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        g1 = cv2.GaussianBlur(mag, (0,0), 1.0)
        g2 = cv2.GaussianBlur(mag, (0,0), 2.0)
        diff = cv2.subtract(g1, g2)
        _, fdog = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        
        # İncelterek ve Siyah-Beyaz
        fdog = cv2.ximgproc.thinning(fdog)
        return cv2.bitwise_not(fdog) # Siyah çizgi, Beyaz zemin

    # --- QUANTIZATION ---
    def quantize(self, img, k):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
        _, lab, cen = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        cen = np.uint8(cen)
        return cen[lab.flatten()].reshape(img.shape)

    # ---------------- ANA MOTOR (ERC V4) ----------------
    def process_hybrid_cartoon(self, edge_thickness=2, color_count=18):
        
        # 1. MASKELEME
        face_mask = self.get_face_mask(self.img)
        if face_mask is None:
            face_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        face_mask_f = face_mask.astype(float) / 255.0
        # Genişletilmiş mask (inv)
        inv_mask_f = 1.0 - face_mask_f

        # 2. RENK İŞLEME (Color Flattening)
        # YÜZ: Oil Painting (Yumuşak geçiş)
        try:
            face_part = cv2.xphoto.oilPainting(self.img, 5, 1) # Boyutu küçülttüm
        except:
            # Fallback: opencv-contrib yoksa
            face_part = cv2.bilateralFilter(self.img, 7, 75, 75)
        
        # VÜCUT: Bilateral (Doku yok etme)
        body_part = cv2.bilateralFilter(self.img, 9, 100, 100)

        # Quantization (Ayrı Ayrı)
        face_quant = self.quantize(face_part, max(16, color_count))
        body_quant = self.quantize(body_part, min(12, color_count - 4))

        # Birleştirme (Blending)
        final_color = (face_quant * face_mask_f[..., None] + body_quant * inv_mask_f[..., None]).astype(np.uint8)

        # 3. ÇİZGİ MOTORU (Fusion)
        xdog = self.get_xdog(self.img) # Ana hatlar
        fdog = self.get_fdog(self.img) # Anime detayları
        
        # Hair ve Wrinkle, YÜZ MASKESİ KULLANILARAK oluşturulur
        hair = self.get_hair_flow(self.img, face_mask) 
        wrinkle = self.get_cloth_wrinkles(self.img, face_mask)

        # Siyah Çizgileri Birleştir (MIN operatörü: En koyu olan kazanır)
        combined = cv2.min(xdog, fdog)
        combined = cv2.min(combined, hair)
        combined = cv2.min(combined, wrinkle)

        # Çizgi Kalınlaştırma (İsteğe bağlı)
        if edge_thickness > 1:
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            combined = cv2.erode(combined, kernel, iterations=1) # Siyahı büyüt

        # 4. FİNAL MASK
        mask_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.GaussianBlur(mask_bgr, (3,3), 0) # Anti-alias

        result = cv2.bitwise_and(final_color, mask_bgr)
        self.vector_base = result.copy()

        return result

    # --- DİĞERLERİ ---
    def process_sketch_style(self):
        xdog = self.get_xdog(self.img, phi=250)
        res = cv2.cvtColor(xdog, cv2.COLOR_GRAY2BGR)
        self.vector_base = res
        return res

    def process_artistic_style(self, style="cartoon", options=None):
        if options is None: options = {}
        edge = options.get("edge_thickness", 2)
        colors = options.get("color_count", 16)
        
        # ERC V4 Motorunu Çağır
        base = self.process_hybrid_cartoon(edge_thickness=edge, color_count=colors)
        res = base.copy()
        
        # Sadece önizleme için efekt uygula
        if style == "comic": 
            res = self.apply_comic_effect(res)
        elif style == "anime": 
            res = self.apply_anime_effect(res)
        
        return res

    def apply_comic_effect(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def apply_anime_effect(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def process_painting_style(self):
        try: return cv2.xphoto.oilPainting(self.img, 7, 1)
        except: return cv2.bilateralFilter(self.img, 9, 75, 75)

    # -------------------- SVG OUTPUT (FIXED FOR ERC V4) --------------------
    def generate_artistic_svg(self, num_colors=16, simplify_factor=0.003, stroke_width=1):
        """ERC V4 için optimize edilmiş SVG motoru."""
        img = self.vector_base if self.vector_base is not None else self.img
        # Sertleştirme (Blur Temizliği)
        img = cv2.pyrMeanShiftFiltering(img, 10, 40)

        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = max(4, min(num_colors, 32))
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(img.shape)

        h, w = img.shape[:2]
        svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']

        unique_colors = np.unique(np.uint8(center), axis=0)
        layers = []
        for color in unique_colors:
            mask = cv2.inRange(quantized, color, color)
            area = cv2.countNonZero(mask)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            is_black = (color[0] < 45 and color[1] < 45 and color[2] < 45)
            layers.append({"hex": hex_color, "area": area, "is_black": is_black, "mask": mask})

        # Büyükten küçüğe, siyahlar en son
        layers.sort(key=lambda x: x["area"], reverse=True)

        for layer in layers:
            if layer["is_black"]: continue
            svg_output.append(self._get_svg_path(layer["mask"], layer["hex"], simplify_factor))
        
        for layer in layers:
            if not layer["is_black"]: continue
            svg_output.append(self._get_svg_path(layer["mask"], layer["hex"], simplify_factor))

        svg_output.append('</svg>')
        return "".join(svg_output)

    def _get_svg_path(self, mask, hex_color, epsilon_factor):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path_str = ""
        for cnt in contours:
            if cv2.contourArea(cnt) < 20: continue
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3: continue
            points = " ".join([f"{p[0][0]},{p[0][1]}" for p in approx])
            path_str += f"M {points} Z "
        
        if not path_str: return ""
        return f'<path d="{path_str}" fill="{hex_color}" stroke="none"/>'
    
    def get_dominant_color(self, img, k=1):
        return [0,0,0]


# ==============================================================================
# 4. MOTOR: NAKIŞ (AUTO-DIGITIZING)
# ==============================================================================
class EmbroideryGenerator:
    """
    Pikselleri dikiş komutlarına çeviren Basit Auto-Digitizer.
    - Tatami Dolgu (Scanline Fill)
    - Running Stitch (Kontur)
    - DST/PES/EXP Çıktısı
    """
    def __init__(self, image_stream):
        # Resmi Vektör Motorundan geçirip temiz halini alıyoruz
        vec_engine = AdvancedVectorEngine(image_stream)
        # Nakış için renk sayısını düşük tutuyoruz (max 12)
        vec_engine.process_hybrid_cartoon(edge_thickness=2, color_count=12)
        self.img = vec_engine.vector_base # Temizlenmiş, cartoonize edilmiş resim
        
        # Nakış için sertleştirme (Anti-alias yok)
        self.img = cv2.pyrMeanShiftFiltering(self.img, 15, 50)

    def generate_pattern(self, file_format="dst"):
        if not HAS_EMBROIDERY:
            raise ImportError("pyembroidery yüklü değil.")

        pattern = pyembroidery.EmbPattern()
        
        # 1. Renkleri Ayrıştır (K-Means)
        data = np.float32(self.img).reshape((-1, 3))
        k = 10 # Nakışta 10 renk idealdir
        _, label, center = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(self.img.shape)
        unique_colors = np.unique(np.uint8(center), axis=0)

        # 2. Her renk için Dikiş Üret
        # PyEmbroidery koordinatları 0.1mm cinsindendir.
        SCALE = 1.0 
        
        for color in unique_colors:
            is_black = (color[0] < 40 and color[1] < 40 and color[2] < 40)
            
            # Maske oluştur
            mask = cv2.inRange(quantized, color, color)
            
            # Renk ekle
            pattern.add_thread(pyembroidery.EmbThread(color[2], color[1], color[0])) # RGB
            pattern.add_command(pyembroidery.COLOR_CHANGE)
            
            # Konturları bul
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: continue # Küçükleri at
                
                # B) KONTUR (OUTLINE) - Running Stitch
                # Her şeklin etrafını dön
                approx = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)
                for point in approx:
                    x, y = point[0]
                    pattern.add_stitch_absolute(pyembroidery.STITCH, x * SCALE, y * SCALE)
                
                # Şekli kapat
                x, y = approx[0][0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, x * SCALE, y * SCALE)
                pattern.add_command(pyembroidery.JUMP) # Diğer şekle atla

        # Çıktı
        out_stream = io.BytesIO()
        
        # Format eşleşmesi
        fmt = file_format.lower()
        if fmt == "emb": fmt = "dst" # PyEmbroidery EMB yazamaz (okur), DST'ye fallback
        
        pyembroidery.write_dst(pattern, out_stream) if fmt == "dst" else \
        pyembroidery.write_pes(pattern, out_stream) if fmt == "pes" else \
        pyembroidery.write_exp(pattern, out_stream) if fmt == "exp" else \
        pyembroidery.write_jef(pattern, out_stream) if fmt == "jef" else \
        pyembroidery.write_vp3(pattern, out_stream) if fmt == "vp3" else \
        pyembroidery.write_xxx(pattern, out_stream) if fmt == "xxx" else \
        pyembroidery.write_dst(pattern, out_stream)

        return out_stream.getvalue()


# ==============================================================================
# 5. API ENDPOINTS (ROUTES)
# ==============================================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/<page>")
def render_page(page):
    if os.path.exists(f"templates/{page}.html"):
        return render_template(f"{page}.html")
    return redirect("/")

# --- VEKTÖR VE NAKIŞ ---
@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    # Limit kontrolü eklenebilir
    
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
            "color_count": color_count
        }
        
        if method == "outline":
            engine.img = engine.process_sketch_style()
        else:
            # ERC V4 Motoru
            engine.img = engine.process_artistic_style(style=style, options=options)
        
        # SVG Üret
        svg = engine.generate_artistic_svg(num_colors=color_count, simplify_factor=0.003)
        svg_b64 = base64.b64encode(svg.encode()).decode()
        
        # Preview
        _, buf = cv2.imencode(".png", engine.img)
        preview_b64 = base64.b64encode(buf).decode()

        # increase_usage(email, "vector", "default")

        return jsonify({
            "success": True,
            "file": svg_b64,
            "preview_img": preview_b64,
            "info": {"style": style}
        })

    except Exception as e:
        print("VECTOR ERROR:", e)
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/convert_embroidery", methods=["POST"])
def api_convert_embroidery():
    """
    YENİ ENDPOINT: Vektörleştirilmiş görseli nakış dosyasına çevirir.
    Girdi: Image (File) + Format (String)
    Çıktı: .DST/.PES Dosyası (Base64)
    """
    if "image" not in request.files: return jsonify({"success": False, "message": "Görsel yok"}), 400
    file = request.files["image"]
    fmt = request.form.get("format", "dst") # dst, pes, jef...

    try:
        # Nakış Motorunu Başlat
        emb_engine = EmbroideryGenerator(file)
        emb_data = emb_engine.generate_pattern(file_format=fmt)
        
        # Base64 Çevir
        emb_b64 = base64.b64encode(emb_data).decode()
        
        return jsonify({
            "success": True, 
            "file": emb_b64, 
            "filename": f"design.{fmt}",
            "format": fmt
        })
    except Exception as e:
        print(f"NAKIŞ HATASI: {e}")
        return jsonify({"success": False, "message": f"Nakış hatası: {str(e)}"}), 500

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
        return jsonify({"success": False, "message": str(e)}), 500

# --- PDF VE RESİM ARAÇLARI ---
@app.route("/api/img/compress", methods=["POST"])
def api_img_compress():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "compress")["allowed"]: return jsonify({"success": False}), 403
    if "image" not in request.files: return jsonify({"success": False}), 400
    try:
        quality = int(request.form.get("quality", 70))
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        increase_usage(email, "image", "compress")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "convert")["allowed"]: return jsonify({"success": False}), 403
    if "image" not in request.files: return jsonify({"success": False}), 400
    try:
        fmt = request.form.get("format", "jpeg").lower()
        fmt_map = {"jpg":"JPEG","jpeg":"JPEG","png":"PNG","webp":"WEBP","pdf":"PDF"}
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, fmt_map.get(fmt, "JPEG"))
        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "pdf", "merge")["allowed"]: return jsonify({"success": False}), 403
    if "pdf_files" not in request.files: return jsonify({"success": False}), 400
    try:
        merger = PdfMerger()
        for f in request.files.getlist("pdf_files"): merger.append(io.BytesIO(f.read()))
        out = io.BytesIO()
        merger.write(out)
        merger.close()
        increase_usage(email, "pdf", "merge")
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

# --- ADMIN & LOGIN ---
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login_route():
    if request.method == "GET": return render_template("admin_login.html")
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
    if email == settings["admin"]["email"]: return jsonify({"status":"admin"})
    user = get_user_data_by_email(email)
    if not user: return jsonify({"status":"not_found"})
    try:
        if datetime.strptime(user.get("end_date"), "%Y-%m-%d") >= datetime.now():
            session["user_email"] = email
            session["is_premium"] = True
            session["user_tier"] = user.get("tier", "starter")
            return jsonify({"status":"premium","tier":session["user_tier"]})
        else: return jsonify({"status":"expired"})
    except: return jsonify({"status":"error"})

@app.route("/api/admin/users", methods=["GET"])
def get_users_api():
    if not session.get("admin_logged"): return jsonify([]), 403
    return jsonify(load_premium_users())

@app.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    data = request.get_json()
    users = load_premium_users()
    users.append({"email": data["email"], "end_date": data["end_date"], "tier": data["tier"], "usage_stats": {}})
    save_premium_users(users)
    return jsonify({"status":"ok"})

@app.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"): return jsonify({}), 403
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
