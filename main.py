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

# --------------------------------------------------------------------
# ---------------------- GELİŞMİŞ VEKTÖR MOTORU ----------------------
# --------------------------------------------------------------------

class AdvancedVectorEngine:
    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

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

        h, w = self.img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.h, self.w = self.img.shape[:2]

    # ---------------- MediaPipe Edge Extraction --------------------
    def extract_face_edges(self, img_in):
        if not HAS_MEDIAPIPE:
            return np.zeros(img_in.shape[:2], dtype=np.uint8)

        h, w = img_in.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Daha detaylı yüz çizgileri
                contour_sets = [
                    mp_face_mesh.FACEMESH_LIPS,
                    mp_face_mesh.FACEMESH_LEFT_EYE,
                    mp_face_mesh.FACEMESH_RIGHT_EYE,
                    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                    mp_face_mesh.FACEMESH_FACE_OVAL
                ]

                for group in contour_sets:
                    pts = []
                    for a, b in group:
                        x = int(lm[a].x * w)
                        y = int(lm[a].y * h)
                        pts.append([x, y])
                        x = int(lm[b].x * w)
                        y = int(lm[b].y * h)
                        pts.append([x, y])
                    pts = np.array(pts)
                    if len(pts) > 2:
                        hull = cv2.convexHull(pts)
                        # Daha kalın çizgiler
                        cv2.polylines(mask, [hull], True, 255, 3)

        return mask

    # -------------------- WHITEBOX CARTOON PROCESSING --------------------
    def process_with_whitebox_cartoon(self, img_input):
        """Whitebox ONNX modelini kullanarak karikatürleştirme"""
        if gan_session is None:
            return img_input
        
        try:
            # Model için hazırlama
            img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (512, 512))
            img_norm = (img_resized / 127.5) - 1.0
            img_norm = np.transpose(img_norm, (2, 0, 1))
            img_norm = np.expand_dims(img_norm, 0).astype(np.float32)
            
            # Model çalıştırma
            outputs = gan_session.run(None, {'input': img_norm})
            cartoon = outputs[0][0]
            
            # Normalleştirmeyi geri al
            cartoon = np.transpose(cartoon, (1, 2, 0))
            cartoon = (cartoon + 1.0) * 127.5
            cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
            
            # Orijinal boyuta geri döndür
            cartoon = cv2.resize(cartoon, (self.w, self.h))
            cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
            
            return cartoon
        except Exception as e:
            print(f"Whitebox model hatası: {e}")
            return img_input

    # -------------------- HYBRID CARTOON (WHITEBOX + OPENCV) --------------------
    def process_hybrid_cartoon(self, edge_thickness=2, color_count=16):
        """Whitebox model + OpenCV ile hibrit karikatürleştirme"""
        
        # 1. Whitebox ile temel karikatür
        if gan_session:
            cartoon = self.process_with_whitebox_cartoon(self.img)
        else:
            # Fallback: CLAHE + MeanShift
            cartoon = self.process_basic_cartoon()
        
        # 2. Kenarları güçlendir (adaptif threshold)
        gray = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # İnce kenarlar için
        edges_fine = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2)
        
        # Kalın kenarlar için
        edges_thick = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5)
        
        # Kenarları birleştir
        edges = cv2.bitwise_or(edges_fine, edges_thick)
        
        # 3. MediaPipe yüz çizgilerini ekle
        if HAS_MEDIAPIPE:
            face_edges = self.extract_face_edges(cartoon)
            edges = cv2.bitwise_or(edges, face_edges)
        
        # 4. Kenar kalınlığını ayarla
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        # 5. Renkleri düzleştir ve azalt (DÜZELTME BURADA YAPILDI)
        # OpenCV kmeans float32 ister, o yüzden dönüştürüyoruz.
        data = np.float32(cartoon).reshape((-1, 3)) 
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # color_count 0 olamaz, güvenlik kontrolü
        k = max(2, color_count)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        center = np.uint8(center)
        flat = center[label.flatten()].reshape(cartoon.shape)
        
        # 6. Posterize efekti ekle
        flat = self.apply_posterize(flat, levels=6)
        
        # 7. Kenarları siyah çizgi olarak uygula
        mask_inv = cv2.bitwise_not(edges)
        mask_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(flat, mask_bgr)
        
        # Kenar çizgilerini siyah yap
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Beyaz olan yerleri (kenar olmayan) siyah yapma, kenar olanları (siyah) koru mantığı
        # Ama maskeleme zaten yukarıda yapıldı, burada sadece ekleme yapıyoruz
        
        # Çizgileri ekle (biraz şeffaflık ile) - Siyah kenar ekliyoruz
        # edges resminde kenarlar siyah(0) veya beyaz(255) olabilir metoda göre.
        # Threshold binary: 0(siyah) ve 255(beyaz). Genelde kenarlar 0'dır (Adaptive mean c ile).
        # Ama biz yukarıda bitwise_not kullandık, yani kenarlar 255 (beyaz) oldu maskede.
        
        # Basitçe: result zaten kenarları siyah içeriyor (bitwise_and sayesinde).
        # Ekstra koyulaştırma gerekirse burası kalabilir ama gerek yoksa result yeterli.
        
        return result

    # -------------------- BASIC CARTOON (FALLBACK) --------------------
    def process_basic_cartoon(self):
        """Whitebox yoksa kullanılacak temel karikatürleştirme"""
        # CLAHE Işık Düzeltme
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
        bright = cv2.cvtColor(cv2.merge((L, A, B)), cv2.COLOR_LAB2BGR)
        
        # MeanShift (pürüzsüzleştirme)
        shifted = cv2.pyrMeanShiftFiltering(bright, sp=8, sr=20)
        
        # Kenar tespiti
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9)
        
        # MediaPipe yüz çizgileri
        if HAS_MEDIAPIPE:
            face_edges = self.extract_face_edges(bright)
            edges = cv2.bitwise_or(edges, face_edges)
        
        # Renk azaltma (DÜZELTME BURADA)
        data = np.float32(shifted).reshape((-1, 3))
        _, label, center = cv2.kmeans(
            data, 12, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        flat = center[label.flatten()].reshape(shifted.shape)
        
        # Kenarları uygula
        mask_inv = cv2.bitwise_not(edges)
        mask_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(flat, mask_bgr)
        
        return result

    # -------------------- POSTERIZE EFFECT --------------------
    def apply_posterize(self, img, levels=8):
        """Posterize efekti uygular (daha sanatsal görünüm)"""
        # Seviyelere göre böl
        step = 256 // levels
        for i in range(levels):
            low = i * step
            high = (i + 1) * step
            mask = cv2.inRange(img, np.array([low, low, low]), np.array([high, high, high]))
            img[mask > 0] = (low + high) // 2
        
        return img

    # -------------------- ARTISTIC STYLES --------------------
    def process_artistic_style(self, style="cartoon", options=None):
        """Farklı sanatsal stiller"""
        if options is None:
            options = {}
        
        edge_thickness = options.get("edge_thickness", 2)
        color_count = options.get("color_count", 16)
        
        if style == "comic":
            # Çizgi roman stili
            result = self.process_hybrid_cartoon(edge_thickness=3, color_count=8)
            result = self.apply_comic_effect(result)
            
        elif style == "anime":
            # Anime stili
            result = self.process_hybrid_cartoon(edge_thickness=1, color_count=24)
            result = self.apply_anime_effect(result)
            
        elif style == "sketch":
            # Eskiz stili
            result = self.process_sketch_style()
            
        elif style == "painting":
            # Resim stili
            result = self.process_painting_style()
            
        else:  # "cartoon" - Varsayılan
            result = self.process_hybrid_cartoon(edge_thickness=edge_thickness, color_count=color_count)
        
        return result

    # -------------------- STYLE EFFECTS --------------------
    def apply_comic_effect(self, img):
        """Çizgi roman efekti"""
        # Kontrast artırma
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sert gölgeler
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return img

    def apply_anime_effect(self, img):
        """Anime efekti"""
        # Yumuşak gölgeler
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Parlaklık artırma
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return img

    def process_sketch_style(self):
        """Eskiz stili"""
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        
        # Gaussian blur
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        
        # Dodge blend
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        # Threshold
        _, sketch = cv2.threshold(sketch, 200, 255, cv2.THRESH_BINARY)
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def process_painting_style(self):
        """Resim stili"""
        # Yağlıboya efekti
        size = 8
        kernel = np.ones((size, size), np.uint8)
        
        # Open işlemi
        opened = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        
        # Bilateral filter
        filtered = cv2.bilateralFilter(opened, 15, 75, 75)
        
        # Renk canlılığı
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 20)
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result

    # -------------------- SVG OUTPUT (GELİŞMİŞ) --------------------
    def generate_artistic_svg(self, num_colors=16, simplify_factor=0.003, stroke_width=1):
        """Daha sanatsal SVG çıktısı"""
        proc = self.img
        
        # Renk sayısını azalt (DÜZELTME BURADA)
        data = np.float32(proc).reshape((-1, 3))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # num_colors 0 olamaz kontrolü
        k = max(2, num_colors)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        center = np.uint8(center)
        quantized = center[label.flatten()].reshape(proc.shape)
        
        h, w = quantized.shape[:2]
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        
        # Arka plan (en büyük alan)
        bg_color = self.get_dominant_color(quantized)
        r, g, b = int(bg_color[2]), int(bg_color[1]), int(bg_color[0])
        svg += f'<rect width="{w}" height="{h}" fill="#{r:02x}{g:02x}{b:02x}"/>'
        
        # Her renk için (dominant rengi atla)
        colors = np.unique(quantized.reshape(-1, 3), axis=0)
        for color in colors:
            # Dominant rengi atla (arka plan zaten çizildi)
            if np.array_equal(color, bg_color):
                continue
                
            b, g, r = color.astype(int)
            mask = cv2.inRange(quantized, color, color)
            
            # Gürültüyü temizle
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 50:  # Küçük detayları atla
                    continue
                
                # Daha az nokta ile yaklaş (sanatsal basitleştirme)
                epsilon = simplify_factor * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Çokgen oluştur
                points = approx.reshape(-1, 2)
                if len(points) < 3:
                    continue
                
                path_data = f"M {points[0][0]} {points[0][1]}"
                for point in points[1:]:
                    path_data += f" L {point[0]} {point[1]}"
                path_data += " Z"
                
                # Kenar rengini biraz koyulaştır
                stroke_r = max(0, r - 40)
                stroke_g = max(0, g - 40)
                stroke_b = max(0, b - 40)
                
                svg += f'<path d="{path_data}" fill="#{r:02x}{g:02x}{b:02x}" stroke="#{stroke_r:02x}{stroke_g:02x}{stroke_b:02x}" stroke-width="{stroke_width}"/>'
        
        svg += "</svg>"
        return svg

    def get_dominant_color(self, img, k=1):
        """Dominant rengi bul"""
        # DÜZELTME BURADA
        pixels = np.float32(img).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # En çok görülen rengi bul
        unique, counts = np.unique(labels, return_counts=True)
        dominant_idx = unique[np.argmax(counts)]
        
        return centers[dominant_idx]

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
        
        # İşleme seçenekleri
        options = {
            "edge_thickness": edge_thickness,
            "color_count": color_count,
            "simplify": 0.003
        }
        
        if method == "outline":
            # Eskiz stili
            engine.img = engine.process_sketch_style()
        else:
            # Sanatsal stil
            engine.img = engine.process_artistic_style(style=style, options=options)
        
        # SVG oluştur
        svg = engine.generate_artistic_svg(
            num_colors=color_count,
            simplify_factor=0.003,
            stroke_width=max(1, edge_thickness // 2)
        )
        
        svg_b64 = base64.b64encode(svg.encode()).decode()

        # Preview görseli
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

