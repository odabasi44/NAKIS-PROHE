import os
import numpy as np
import base64

# MediaPipe kontrolü
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

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
    
