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

# --- MEDIAPIPE ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
    print("✅ MediaPipe Yüklendi.")
except ImportError:
    HAS_MEDIAPIPE = False
    print("⚠️ UYARI: 'mediapipe' eksik!")

# --- ONNX MODELS ---
gan_session = None
u2net_session = None
base_dir = os.path.dirname(os.path.abspath(__file__))

# Model Yolları (Container veya Local)
possible_wb = [os.path.join(base_dir, "models/whitebox_cartoon.onnx"), "whitebox_cartoon.onnx", "/app/models/whitebox_cartoon.onnx"]
wb_path = next((p for p in possible_wb if os.path.exists(p)), None)
if wb_path:
    try:
        gan_session = ort.InferenceSession(wb_path, providers=["CPUExecutionProvider"])
        print(f"🚀 Whitebox Model: {wb_path}")
    except: pass

possible_u2 = [os.path.join(base_dir, "models/u2net.onnx"), "u2net.onnx", "/app/models/u2net.onnx"]
u2_path = next((p for p in possible_u2 if os.path.exists(p)), None)
u2net_input_name = "input"
if u2_path:
    try:
        u2net_session = ort.InferenceSession(u2_path, providers=["CPUExecutionProvider"])
        u2net_input_name = u2net_session.get_inputs()[0].name
        print(f"✅ U2Net Model: {u2_path}")
    except: pass

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = "BOTLAB_SECRET_123"
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# --- AYARLAR ---
SETTINGS_FILE = "settings.json"
PREMIUM_FILE = "users.json"

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {"admin": {"email": "admin@botlab.com", "password": "admin"}, "limits": {"vector": {"default": {"free": 5}}}}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def check_user_status(email, tool, subtool):
    # Basit limit kontrolü (Geliştirilebilir)
    return {"allowed": True, "premium": False, "left": 99}

def increase_usage(email, tool, subtool):
    pass 

# --- ERC V3 MOTORU ---
class AdvancedVectorEngine:
    def __init__(self, image_stream):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        self.vector_base = None
        if self.original_img is None: raise ValueError("Görüntü okunamadı.")
        
        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            alpha = self.original_img[:, :, 3]
            rgb = self.original_img[:, :, :3]
            white = np.ones_like(rgb) * 255
            self.img = (rgb * (alpha[:, :, None] / 255.0) + white * (1 - alpha[:, :, None] / 255.0)).astype(np.uint8)
        else:
            self.img = self.original_img[:, :, :3]

        h, w = self.img.shape[:2]
        if max(h, w) > 1000:
            scale = 1000 / max(h, w)
            self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.h, self.w = self.img.shape[:2]

    def get_xdog_edges(self, img, gamma=0.97, phi=200, epsilon=0.01):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g1 = cv2.GaussianBlur(gray, (0, 0), 0.8)
        g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)
        diff = g1 - gamma * g2
        edges = np.tanh(phi * (diff - epsilon))
        edges = (edges + 1) / 2.0
        edges = (edges * 255).astype(np.uint8)
        _, bin_edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
        return cv2.medianBlur(bin_edges, 3)

    def get_face_mask(self, img):
        if not HAS_MEDIAPIPE: return None
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                contour = mp_face_mesh.FACEMESH_FACE_OVAL
                pts = []
                for source, target in contour:
                    pt = lm[source]
                    pts.append([int(pt.x * w), int(pt.y * h)])
                if pts:
                    hull = cv2.convexHull(np.array(pts))
                    cv2.fillPoly(mask, [hull], 255)
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    def process_hybrid_cartoon(self, edge_thickness=2, color_count=16):
        # 1. Yüz Maskeleme
        face_mask_gray = self.get_face_mask(self.img)
        if face_mask_gray is None:
            # Yüz yoksa basit işlem
            filtered = cv2.bilateralFilter(self.img, 9, 75, 75)
            data = np.float32(filtered).reshape((-1, 3))
            _, label, center = cv2.kmeans(data, color_count, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            quantized = np.uint8(center)[label.flatten()].reshape(filtered.shape)
            edges = self.get_xdog_edges(self.img)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result = cv2.bitwise_and(quantized, edges_bgr)
            self.vector_base = result
            return result

        # 2. Bölgesel İşleme (ERC V3 Logic)
        face_mask = cv2.cvtColor(face_mask_gray, cv2.COLOR_GRAY2BGR)
        inv_mask = cv2.bitwise_not(face_mask)
        face_mask_f = face_mask.astype(float) / 255.0
        inv_mask_f = inv_mask.astype(float) / 255.0

        # YÜZ: Oil Paint
        try: face_part = cv2.xphoto.oilPainting(self.img, 5, 1)
        except: face_part = cv2.bilateralFilter(self.img, 7, 75, 75)
        
        # VÜCUT: Bilateral
        body_part = cv2.bilateralFilter(self.img, 9, 100, 100)

        # Quantize
        def quant(im, k):
            d = np.float32(im).reshape((-1, 3))
            _, l, c = cv2.kmeans(d, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            return np.uint8(c)[l.flatten()].reshape(im.shape)

        face_quant = quant(face_part, max(16, color_count))
        body_quant = quant(body_part, min(12, color_count - 4))

        final_color = (face_quant * face_mask_f + body_quant * inv_mask_f).astype(np.uint8)

        # 3. Çizgiler (XDoG)
        xdog = self.get_xdog_edges(self.img)
        if edge_thickness > 1:
            xdog = cv2.erode(xdog, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
        
        edges_bgr = cv2.cvtColor(xdog, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.GaussianBlur(edges_bgr, (3,3), 0)

        final_result = cv2.bitwise_and(final_color, edges_bgr)
        self.vector_base = final_result
        return final_result

    def generate_artistic_svg(self, num_colors=16, simplify_factor=0.002, stroke_width=1):
        img = self.vector_base if self.vector_base is not None else self.img
        # Sertleştirme (Clean Blur)
        img = cv2.pyrMeanShiftFiltering(img, 10, 40)
        
        data = np.float32(img).reshape((-1, 3))
        k = max(4, min(num_colors, 32))
        _, label, center = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(img.shape)
        
        h, w = img.shape[:2]
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
        
        unique_colors = np.unique(np.uint8(center), axis=0)
        layers = []
        for color in unique_colors:
            mask = cv2.inRange(quantized, color, color)
            area = cv2.countNonZero(mask)
            hex_c = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            is_black = (color[0]<40 and color[1]<40 and color[2]<40)
            layers.append({"hex": hex_c, "area": area, "is_black": is_black, "mask": mask})
        
        layers.sort(key=lambda x: x["area"], reverse=True)
        
        for layer in layers:
            if layer["is_black"]: continue
            svg.append(self._get_path(layer["mask"], layer["hex"], simplify_factor))
            
        for layer in layers:
            if not layer["is_black"]: continue
            svg.append(self._get_path(layer["mask"], layer["hex"], simplify_factor))
            
        svg.append('</svg>')
        return "".join(svg)

    def _get_path(self, mask, hex_c, eps_fac):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path = ""
        for c in cnts:
            if cv2.contourArea(c) < 25: continue
            approx = cv2.approxPolyDP(c, eps_fac * cv2.arcLength(c, True), True)
            if len(approx) < 3: continue
            pts = " ".join([f"{p[0][0]},{p[0][1]}" for p in approx])
            path += f"M {pts} Z "
        return f'<path d="{path}" fill="{hex_c}" stroke="none"/>' if path else ""

# --- ROUTES ---
@app.route("/")
def home(): return render_template("index.html")

@app.route("/vektor")
def vektor_page(): return render_template("vektor.html")

@app.route("/api/check_tool_status/<tool>/<subtool>")
def check_status(tool, subtool): return jsonify({"allowed": True, "premium": True, "left": 999})

@app.route("/api/vectorize", methods=["POST"])
def api_vectorize():
    if "image" not in request.files: return jsonify({"success": False}), 400
    file = request.files["image"]
    method = request.form.get("method", "cartoon")
    
    thick = int(request.form.get("edge_thickness", 2))
    colors = int(request.form.get("color_count", 16))

    try:
        engine = AdvancedVectorEngine(file)
        
        if method == "cartoon":
            engine.img = engine.process_hybrid_cartoon(edge_thickness=thick, color_count=colors)
        
        svg = engine.generate_artistic_svg(num_colors=colors, simplify_factor=0.002)
        svg_b64 = base64.b64encode(svg.encode()).decode()
        
        _, buf = cv2.imencode(".png", engine.img)
        prev_b64 = base64.b64encode(buf).decode()

        return jsonify({"success": True, "file": svg_b64, "preview_img": prev_b64})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
