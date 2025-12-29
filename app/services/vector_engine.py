import cv2
import numpy as np
import base64
import json

# MediaPipe (ZORUNLU)
# Not: Coolify logundaki "AttributeError: module 'mediapipe' has no attribute 'solutions'"
# genelde yanlış/uyumsuz mediapipe paketinden kaynaklanır. Burada fail-fast yapıyoruz.
try:
    import mediapipe as mp
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
        raise RuntimeError(
            "MediaPipe yüklü görünüyor ama 'mp.solutions.face_mesh' bulunamadı. "
            "Bu genelde uyumsuz/bozuk mediapipe kurulumu demektir. "
            "Docker imajında Python 3.10 + mediapipe==0.10.14 kullanın."
        )
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except ImportError as e:
    raise RuntimeError(
        "MediaPipe zorunludur ama bulunamadı. requirements.txt içinde mediapipe kurulu olmalı."
    ) from e
except Exception as e:
    raise RuntimeError(
        f"MediaPipe başlatılamadı: {e}. "
        "Docker imajında Python 3.10 + mediapipe==0.10.14 kullanın."
    ) from e

class AdvancedVectorEngine:
    """
    ERC V4 PRO MAX (OPTIMIZED)
    - XDoG & FDoG Hybrid
    - Smart Region Masking (Yüz, Saç, Vücut ayrımı)
    - Wilcom Ready Cleanup
    """

    def __init__(self, image_stream, target_width=None, target_height=None, lock_aspect=True, max_dim=1000):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        self.vector_base = None
        self.alpha = None  # varsa 0-255
        self.fg_mask = None  # varsa 0/255 tek kanal (foreground)
        self.color_ref = None

        if self.original_img is None:
            raise ValueError("Görüntü okunamadı.")

        # Saydamlık düzeltme
        if len(self.original_img.shape) == 3 and self.original_img.shape[2] == 4:
            alpha = self.original_img[:, :, 3]
            self.alpha = alpha.copy()
            # fg mask (çok düşük alpha değerlerini background say)
            self.fg_mask = ((alpha.astype(np.uint16)) > 10).astype(np.uint8) * 255

            rgb = self.original_img[:, :, :3]
            white = np.ones_like(rgb) * 255
            self.img = (rgb * (alpha[:, :, None] / 255.0) + white * (1 - alpha[:, :, None] / 255.0)).astype(np.uint8)
        else:
            self.img = self.original_img[:, :, :3]

        # Boyutlandırma:
        # - UI'dan hedef genişlik/yükseklik verilmişse ona göre resize
        # - verilmemişse eski davranış: max_dim (varsayılan 1000px) üstünü küçült
        h, w = self.img.shape[:2]
        try:
            tw = int(target_width) if target_width is not None else 0
        except Exception:
            tw = 0
        try:
            th = int(target_height) if target_height is not None else 0
        except Exception:
            th = 0
        lock = bool(lock_aspect)

        # güvenlik: çok büyük boyutlar performansı öldürmesin
        max_allowed = 3000
        max_dim = int(max_dim) if max_dim else 1000

        def _resize_masks(new_w: int, new_h: int):
            # alpha/fg_mask varsa görselle aynı boyuta getir (OpenCV binary_op size mismatch fix)
            if self.alpha is not None and (self.alpha.shape[1] != new_w or self.alpha.shape[0] != new_h):
                self.alpha = cv2.resize(self.alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if self.fg_mask is not None and (self.fg_mask.shape[1] != new_w or self.fg_mask.shape[0] != new_h):
                self.fg_mask = cv2.resize(self.fg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                # normalize back to 0/255
                self.fg_mask = ((self.fg_mask.astype(np.uint16)) > 10).astype(np.uint8) * 255

        if tw > 0 or th > 0:
            if lock:
                # tek değer verildiyse diğerini hesapla
                if tw > 0 and th <= 0:
                    th = int(round((tw * h) / w)) if w > 0 else tw
                elif th > 0 and tw <= 0:
                    tw = int(round((th * w) / h)) if h > 0 else th
                # ikisi de verildiyse oranı korumak için en yakın scale ile ayarla
                if tw > 0 and th > 0 and w > 0 and h > 0:
                    s = min(tw / w, th / h)
                    tw = int(round(w * s))
                    th = int(round(h * s))

            # clamp
            tw = max(1, min(tw, max_allowed)) if tw > 0 else 0
            th = max(1, min(th, max_allowed)) if th > 0 else 0

            if tw > 0 and th > 0 and (tw != w or th != h):
                interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_CUBIC
                self.img = cv2.resize(self.img, (tw, th), interpolation=interp)
                _resize_masks(tw, th)
        else:
            # default optimize
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                nh, nw = self.img.shape[:2]
                _resize_masks(nw, nh)

        self.h, self.w = self.img.shape[:2]

        self._face_landmarks = None

    def _ensure_face_landmarks(self):
        if self._face_landmarks is not None:
            return self._face_landmarks
        if (not HAS_MEDIAPIPE) or (mp_face_mesh is None):
            raise RuntimeError("MediaPipe face_mesh hazır değil. Kurulum/uyumluluk kontrol edin.")
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                self._face_landmarks = results.multi_face_landmarks[0].landmark
            else:
                self._face_landmarks = []
        return self._face_landmarks

    # --- YARDIMCI: thinning (opencv-contrib yoksa fallback) ---
    def _thinning(self, binary_img):
        """
        binary_img: 0/255 tek kanallı görüntü.
        Önce cv2.ximgproc.thinning denenir; yoksa morfolojik skeleton fallback kullanılır.
        """
        try:
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                return cv2.ximgproc.thinning(binary_img)
        except Exception:
            pass

        # Fallback: Morphological skeletonization
        img = (binary_img > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        max_iter = 200  # güvenlik
        it = 0
        while not done and it < max_iter:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            done = cv2.countNonZero(img) == 0
            it += 1
        return skel

    # --- YARDIMCI: Yüz Maskesi ---
    def get_face_mask(self, img):
        # MediaPipe zorunlu olduğu için burada None dönmek yerine net hata verelim.
        if (not HAS_MEDIAPIPE) or (mp_face_mesh is None):
            raise RuntimeError("MediaPipe face_mesh hazır değil. Kurulum/uyumluluk kontrol edin.")
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        lm = self._ensure_face_landmarks()
        if lm:
            contour = mp_face_mesh.FACEMESH_FACE_OVAL
            pts = []
            for source_idx, target_idx in contour:
                pt = lm[source_idx]
                pts.append([int(pt.x * w), int(pt.y * h)])

            if pts:
                pts = np.array(pts)
                hull = cv2.convexHull(pts)
                cv2.fillPoly(mask, [hull], 255)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    def _feature_poly_mask(self, contour, *, blur_ksize: int = 7):
        h, w = self.img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        lm = self._ensure_face_landmarks()
        if not lm:
            return mask
        pts = []
        for source_idx, target_idx in contour:
            pt = lm[source_idx]
            pts.append([int(pt.x * w), int(pt.y * h)])
        if pts:
            pts = np.array(pts)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 255)
            k = int(max(0, blur_ksize))
            if k >= 3:
                if k % 2 == 0:
                    k += 1
                mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    def get_lips_mask(self):
        if not hasattr(mp_face_mesh, "FACEMESH_LIPS"):
            return np.zeros((self.h, self.w), dtype=np.uint8)
        return self._feature_poly_mask(mp_face_mesh.FACEMESH_LIPS, blur_ksize=5)

    def get_eyes_mask(self):
        if not hasattr(mp_face_mesh, "FACEMESH_LEFT_EYE") or not hasattr(mp_face_mesh, "FACEMESH_RIGHT_EYE"):
            return np.zeros((self.h, self.w), dtype=np.uint8)
        ml = self._feature_poly_mask(mp_face_mesh.FACEMESH_LEFT_EYE, blur_ksize=3)
        mr = self._feature_poly_mask(mp_face_mesh.FACEMESH_RIGHT_EYE, blur_ksize=3)
        return cv2.bitwise_or(ml, mr)

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
        hair_lines = self._thinning(hair_lines)
        
        # SİYAH çizgi (0), BEYAZ zemin (255) formatına çevir
        hair_lines_inv = cv2.bitwise_not(hair_lines)

        # MASK ELEME: Yüz bölgesindeki saç çizgilerini sil (Beyaz yap)
        if face_mask is not None:
            hair_lines_inv = cv2.bitwise_or(hair_lines_inv, face_mask)

        return hair_lines_inv

    def _cleanup_small_regions(self, img, min_ratio=0.0005):
        """
        Portrait mode için küçük izole renk bölgelerini temizler.
        """
        h, w = img.shape[:2]
        min_area = max(50, int(min_ratio * h * w))

        out = img.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask = (labels == i).astype(np.uint8) * 255
                dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
                ring = cv2.bitwise_and(dil, cv2.bitwise_not(mask))
                ys, xs = np.where(ring > 0)
                if ys.size == 0:
                    continue
                color = np.median(out[ys, xs], axis=0).astype(np.uint8)
                out[labels == i] = color

        return out

   

    # --- HAIR FLOW FOR PORTRAIT MODE ---
    def get_hair_flow_portrait(self, img, face_mask):
        """Saç şeklini basitleştirir, ince telleri baskılar."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Strong Gaussian blur to suppress fine hair strands
        blurred = cv2.GaussianBlur(gray, (15, 15), 3.0)
        
        # Simple edge detection for major hair shapes only
        edges = cv2.Canny(blurred, 30, 60)
        
        # Morphological closing to merge nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Thinning for clean lines
        edges = self._thinning(edges)
        
        # Invert: black lines on white background
        hair_lines_inv = cv2.bitwise_not(edges)
        
        # Remove hair from face region
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
        wr = self._thinning(wr)
        
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
        bin_edges = cv2.medianBlur(bin_edges, 3)
        return cv2.bitwise_not(bin_edges) # Siyah çizgi, Beyaz zemin

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
        fdog = self._thinning(fdog)
        return cv2.bitwise_not(fdog) # Siyah çizgi, Beyaz zemin

    # --- QUANTIZATION ---
    def quantize(self, img, k):
        """
        Geriye sadece quantize edilmiş BGR döner (eski davranış).
        """
        q, _, _ = self._quantize_kmeans(img, k, return_labels=False)
        return q

    def _quantize_kmeans(self, img, k, return_labels: bool = False):
        """
        K-means quantization.
        - return_labels=False: (quant, None, centers)
        - return_labels=True: (quant, labels(H,W) uint8, centers)
        """
        k = int(max(2, min(int(k), 64)))
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, lab, cen = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        cen = np.uint8(cen)
        quant = cen[lab.flatten()].reshape(img.shape)
        if not return_labels:
            return quant, None, cen
        labels = lab.reshape((img.shape[0], img.shape[1])).astype(np.uint8)
        return quant, labels, cen

    # ---------------- ANA MOTOR (ERC V4) ----------------
    def process_hybrid_cartoon(self, edge_thickness=2, color_count=18, portrait_mode=False):
        # Güvenlik: çok düşük color_count değerleri (3-5 gibi) bazı iç hesaplarda negatif/0 k üretip OpenCV hatalarına yol açabiliyor.
        # Bu motor için minimum 4 renk gerekir; daha düşük isteniyorsa FastAPI EPS/BOT motoru kullanılmalı.
        try:
            color_count = int(color_count)
        except Exception:
            color_count = 18
            color_count = max(4, min(color_count, 32))
        
        # 1. MASKELEME
        face_mask = self.get_face_mask(self.img)
        if face_mask is None:
            face_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        # Portrait mode: sharper mask with skin-tone expansion
        if portrait_mode:
            # Remove Gaussian blur for sharper edges
            _, face_mask = cv2.threshold(face_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Skin-tone expansion using YCrCb
            ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            skin[y < 55] = 0  # Remove dark areas
            
            # Combine MediaPipe mask with skin-tone mask
            face_mask = cv2.bitwise_or(face_mask, skin)
            
            # Morphological operations for clean edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        face_mask_f = face_mask.astype(float) / 255.0
        # Genişletilmiş mask (inv)
        inv_mask_f = 1.0 - face_mask_f

        # 2. RENK İŞLEME (Color Flattening)
        if portrait_mode:
            # Portrait mode: Strong bilateral filter for skin smoothing
            face_part = cv2.bilateralFilter(self.img, 15, 120, 120)
            body_part = cv2.bilateralFilter(self.img, 9, 100, 100)
            
            # Face quantization: 3-4 flat colors using LAB space
            face_k = 3 if color_count <= 8 else 4
            
            # Extract face region
            if cv2.countNonZero(face_mask) > 0:
                ys, xs = np.where(face_mask > 0)
                face_pixels = face_part[ys, xs]
                
                if face_pixels.shape[0] > 200:
                    # Convert to LAB for better skin tone clustering
                    face_lab = cv2.cvtColor(face_part, cv2.COLOR_BGR2LAB)
                    face_lab_pixels = face_lab[ys, xs]
                    
                    # K-means in LAB space
                    data = np.float32(face_lab_pixels).reshape((-1, 3))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
                    _, labels, centers = cv2.kmeans(data, face_k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                    centers = np.uint8(centers)
                    
                    # Convert centers back to BGR
                    centers_bgr = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
                    
                    # Apply flat colors to face region
                    mapped = centers_bgr[labels.flatten()].reshape(-1, 3)
                    face_quant = face_part.copy()
                    face_quant[ys, xs] = mapped
                    
                    # Smooth edges between color regions
                    face_quant = cv2.bilateralFilter(face_quant, 5, 80, 80)
                else:
                    face_quant = self.quantize(face_part, face_k)
            else:
                face_quant = self.quantize(face_part, face_k)
                
            body_k = max(2, min(8, color_count - face_k))
            body_quant = self.quantize(body_part, body_k)
        else:
            # Original behavior
            # YÜZ: Oil Painting (Yumuşak geçiş)
            try:
                face_part = cv2.xphoto.oilPainting(self.img, 5, 1) # Boyutu küçülttüm
            except:
                # Fallback: opencv-contrib yoksa
                face_part = cv2.bilateralFilter(self.img, 7, 75, 75)
            
            # VÜCUT: Bilateral (Doku yok etme)
            body_part = cv2.bilateralFilter(self.img, 9, 100, 100)

            # Quantization (Ayrı Ayrı)
            face_k = max(8, min(32, max(16, color_count)))
            body_k = max(2, min(12, color_count - 4))
            face_quant = self.quantize(face_part, face_k)
            body_quant = self.quantize(body_part, body_k)
        # Birleştirme (Blending)
        final_color = (face_quant * face_mask_f[..., None] + body_quant * inv_mask_f[..., None]).astype(np.uint8)

        if portrait_mode:
            ref = self.color_ref if (self.color_ref is not None and hasattr(self.color_ref, "shape")) else self.img
            try:
                hsv = cv2.cvtColor(final_color, cv2.COLOR_BGR2HSV)
                hch, sch, vch = cv2.split(hsv)
                sch = np.clip((sch.astype(np.float32) * 1.25), 0, 255).astype(np.uint8)
                final_color = cv2.cvtColor(cv2.merge([hch, sch, vch]), cv2.COLOR_HSV2BGR)
            except Exception:
                pass

            def _median_bgr(src, m):
                try:
                    if m is None:
                        return None
                    mm = (m.astype(np.uint8) > 10)
                    if mm.ndim != 2:
                        return None
                    pix = src[mm]
                    if pix.size == 0:
                        return None
                    return np.median(pix.reshape(-1, 3), axis=0).astype(np.uint8)
                except Exception:
                    return None

            try:
                ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
                skin[y < 55] = 0
                skin = cv2.bitwise_and(skin, face_mask)
                skin_c = _median_bgr(ref, skin)
                if skin_c is not None:
                    final_color[skin > 10] = skin_c
            except Exception:
                pass

            try:
                lips = self.get_lips_mask()
                lips = cv2.bitwise_and(lips, face_mask)
                lips_c = _median_bgr(ref, lips)
                if lips_c is not None:
                    lab = cv2.cvtColor(lips_c.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3)
                    lab[1] = np.clip(int(lab[1]) + 14, 0, 255)
                    lab[0] = np.clip(int(lab[0]) + 6, 0, 255)
                    lips_c2 = cv2.cvtColor(lab.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(3)
                    final_color[lips > 10] = lips_c2
            except Exception:
                pass

            try:
                eyes = self.get_eyes_mask()
                eyes = cv2.bitwise_and(eyes, face_mask)
                hsv_o = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
                hh, ss, vv = cv2.split(hsv_o)
                sclera = ((vv > 170) & (ss < 70)).astype(np.uint8) * 255
                sclera = cv2.bitwise_and(sclera, eyes)
                if cv2.countNonZero(sclera) > 10:
                    final_color[sclera > 10] = np.array([245, 245, 245], dtype=np.uint8)
            except Exception:
                pass

            try:
                hsv_o = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
                hh, ss, vv = cv2.split(hsv_o)
                dark = (vv < 75).astype(np.uint8) * 255
                ring = cv2.dilate(face_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
                ring = cv2.bitwise_and(ring, cv2.bitwise_not(face_mask))
                hair = cv2.bitwise_and(dark, ring)
                hair_c = _median_bgr(ref, hair)
                if hair_c is None:
                    hair_c = np.array([15, 15, 15], dtype=np.uint8)
                final_color[hair > 10] = hair_c
            except Exception:
                pass

        if portrait_mode:
            # Portrait mode: Simplified XDoG for major contours only
            xdog = self.get_xdog(self.img, gamma=0.98, phi=150, epsilon=0.02)  # Higher thresholds, fewer lines
            fdog = np.ones_like(xdog) * 255  # Disable FDoG (white background)
            
            # Simplified hair flow for portrait mode
            hair = self.get_hair_flow_portrait(self.img, face_mask) 
            wrinkle = np.ones_like(xdog) * 255  # Disable wrinkles (white background)
        else:
            # Original behavior
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
        # Portrait mode: Remove small isolated regions for cleaner output
        if portrait_mode:
            result = self._cleanup_small_regions(result)
        self.vector_base = result.copy()

        return result

    def process_flat_cartoon(self, edge_thickness=2, color_count=10, border_sensitivity=12):
        """
        Nakış/Wilcom için optimize edilmiş 'flat' görünüm:
        - Az renk (posterize)
        - Bölge sınırlarından temiz kontur (renk geçişi sınırı)
        - Saç/kırışıklık gibi mikro çizgiler yok (daha az node, daha temiz SVG)
        - Nakış için en temiz, en vektörel, en düzgün çıktı
        """
        # 1) yumuşatma + renk azaltma (BÖLGE TABANLI) - Nakış için daha agresif yumuşatma
        base = cv2.bilateralFilter(self.img, 11, 120, 120)  # daha yumuşak, daha az doku
        k = int(max(2, min(int(color_count), 24)))
        quant, labels, centers = self._quantize_kmeans(base, k, return_labels=True)

        # 1.0) Label smoothing (nakış için daha agresif speckle azaltma)
        if labels is not None:
            labels = cv2.medianBlur(labels, 5)  # 5x5 daha temiz alanlar

        # 1.1) Bölge/adacık temizliği: küçük label bileşenlerini komşu baskın label'a birleştir
        try:
            if labels is not None:
                hq, wq = labels.shape[:2]
                img_area = float(hq * wq)
                min_pix = max(80, int(0.0001 * img_area))  # Nakış için daha agresif temizlik
                bright_min_pix = max(min_pix * 15, int(0.005 * img_area))  # Parlak adacıklar için daha agresif
                kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                # Quantized BGR üzerinden "parlak" label'ları tespit et (beyaz highlight için daha agresif)
                label_brightness = np.zeros((k,), dtype=np.float32)
                for li in range(k):
                    c = centers[li]
                    label_brightness[li] = (float(c[0]) + float(c[1]) + float(c[2])) / 3.0

                lab2 = labels.copy()
                for li in range(k):
                    mask = (lab2 == li).astype(np.uint8) * 255
                    if cv2.countNonZero(mask) == 0:
                        continue
                    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    if num <= 1:
                        continue
                    is_bright = (label_brightness[li] > 235.0)
                    thr = bright_min_pix if is_bright else min_pix
                    for i in range(1, num):
                        area = int(stats[i, cv2.CC_STAT_AREA])
                        if area >= thr:
                            continue
                        comp = (cc == i).astype(np.uint8) * 255
                        ring = cv2.dilate(comp, kernel3, iterations=2)
                        ring = cv2.bitwise_and(ring, cv2.bitwise_not(comp))
                        ry, rx = np.where(ring > 0)
                        if ry.size == 0:
                            continue
                        neigh_labels = lab2[ry, rx].reshape(-1)
                        # kendi label'ını çıkar
                        neigh_labels = neigh_labels[neigh_labels != li]
                        if neigh_labels.size == 0:
                            continue
                        vals, counts = np.unique(neigh_labels, return_counts=True)
                        newl = int(vals[int(np.argmax(counts))])
                        yy, xx = np.where(cc == i)
                        lab2[yy, xx] = newl
                labels = lab2
                quant = centers[labels.reshape(-1)].reshape(hq, wq, 3)
        except Exception:
            pass

        # 1.2) Yüz/ten bölgesi düzeltmesi:
        # - Alındaki beyaz highlight gibi "parlak adacıkları" ten rengine yaklaştır
        # - Dengeli/Yüksek preset'lerinde yüz paletini SABİT tut (ör: 3 renk)
        # - Ten rengini bir tık aç (LAB L kanalını artır)
        try:
            h0, w0 = self.img.shape[:2]
            # Skin mask (YCrCb) - kaba ama hızlı
            ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            # çok karanlık alanları çıkar
            skin[y < 55] = 0
            # maske temizliği
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k5, iterations=1)
            skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k5, iterations=2)

            # Yüz seçimi (geliştirilmiş):
            # - Sadece "en büyük" komponent değil; üst bölgede, makul boyutta ve merkeze yakın olan komponent(ler) seçilir.
            # - Grup fotoğraflarında 1-3 yüz komponenti birleştirilebilir.
            num, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
            face_mask = np.zeros_like(skin)

            if num > 1:
                img_cx, img_cy = (w0 * 0.5), (h0 * 0.38)  # yüzler genelde üst-orta bölgede
                img_area = float(w0 * h0)
                candidates = []

                for i in range(1, num):
                    x = int(stats[i, cv2.CC_STAT_LEFT])
                    yb = int(stats[i, cv2.CC_STAT_TOP])
                    ww = int(stats[i, cv2.CC_STAT_WIDTH])
                    hh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])

                    if ww <= 0 or hh <= 0:
                        continue

                    # Boyut filtreleri:
                    # - çok küçük benekleri at
                    # - çok büyük (vücut/kol) bölgelerini at
                    if area < max(500, int(0.002 * img_area)):
                        continue
                    if area > int(0.25 * img_area):
                        continue

                    # Konum: üst bölgede olması daha olası
                    cy = yb + hh * 0.5
                    if cy > (h0 * 0.80):
                        continue

                    # En-boy oranı: yüz çoğunlukla 0.6–1.8
                    ar = float(ww) / float(hh)
                    if ar < 0.55 or ar > 1.9:
                        continue

                    cx = x + ww * 0.5
                    # Merkeze yakınlık skoru
                    dx = (cx - img_cx) / max(1.0, w0)
                    dy = (cy - img_cy) / max(1.0, h0)
                    dist = (dx * dx + dy * dy)

                    # Üstte olma bonusu
                    top_bonus = 1.0 - min(1.0, max(0.0, cy / (h0 * 0.85)))

                    # Alanı normalize edip çok büyükleri cezalandır
                    area_n = float(area) / img_area
                    size_score = min(1.0, area_n / 0.08)  # ~%8'e kadar iyi

                    score = (2.2 * size_score) + (1.2 * top_bonus) - (2.0 * dist)
                    candidates.append((score, i))

                candidates.sort(key=lambda t: t[0], reverse=True)
                max_faces = 3
                picked = [idx for (sc, idx) in candidates[:max_faces] if sc > 0.15]
                if not picked and candidates:
                    picked = [candidates[0][1]]

                for idx in picked:
                    face_mask[labels == idx] = 255
            else:
                face_mask = skin

            # Maske genişletme: alın/yanak gibi sınırları biraz kapsasın
            if cv2.countNonZero(face_mask) > 0:
                face_mask = cv2.dilate(face_mask, k5, iterations=1)

            face_area = int(cv2.countNonZero(face_mask))
            if face_area > max(400, int(0.01 * h0 * w0)):
                # yüz piksellerinden sabit palet çıkar (3 renk)
                k_face = 3
                ys, xs = np.where(face_mask > 0)
                face_pixels = quant[ys, xs].reshape(-1, 3)
                if face_pixels.shape[0] > 200:
                    data = np.float32(face_pixels)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
                    # unique renk < k_face ise k'yi düşür
                    uniq = np.unique(face_pixels, axis=0)
                    k_use = int(max(2, min(k_face, len(uniq))))
                    _, f_labels, centers = cv2.kmeans(data, k_use, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                    centers = np.uint8(centers)

                    # aşırı beyaz merkezleri ten rengine yaklaştır (diğer merkezlerin medyanı)
                    meanv = centers.mean(axis=1)
                    ok_idx = np.where(meanv < 230)[0]
                    if ok_idx.size == 0:
                        ok_idx = np.arange(len(centers))
                    fallback = np.median(centers[ok_idx], axis=0).astype(np.uint8)
                    for i in range(len(centers)):
                        if float(meanv[i]) > 235:
                            centers[i] = fallback

                    # ten rengini bir tık aç (LAB L +8)
                    lab = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
                    lab[:, 0] = np.clip(lab[:, 0].astype(np.int16) + 8, 0, 255).astype(np.uint8)
                    centers = cv2.cvtColor(lab.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)

                    # yüz bölgesini bu sabit palete map'le
                    mapped = centers[f_labels.flatten()].reshape(-1, 3)
                    quant[ys, xs] = mapped
        except Exception:
            pass

        # 2) Kontur: SADECE bölge sınırlarından üret (doku/ışık çizgisi üretmez)
        edges = None
        if labels is not None:
            lab = labels.astype(np.int16)
            # komşu label farkı => boundary
            e = np.zeros_like(lab, dtype=np.uint8)
            e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8) * 255
            e[1:, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8) * 255
            edges = e
        else:
            # fallback (eski): Laplacian
            gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
            lap = cv2.convertScaleAbs(lap)
            _, edges = cv2.threshold(lap, int(max(1, border_sensitivity)), 255, cv2.THRESH_BINARY)

        if edge_thickness > 1:
            ksz = int(edge_thickness)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            edges = cv2.dilate(edges, kernel, iterations=1)

        # 2.1) Edge temizliği: küçük benekleri azalt (nakış için daha agresif)
        try:
            h, w = edges.shape[:2]
            kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel3, iterations=2)  # 2 iterasyon daha temiz

            # Küçük connected-component'ları at (nakış için daha büyük threshold)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            min_pixels = max(50, int(0.00005 * h * w))  # Daha agresif temizlik
            cleaned = np.zeros_like(edges)
            for i in range(1, num):
                if int(stats[i, cv2.CC_STAT_AREA]) >= min_pixels:
                    cleaned[labels == i] = 255
            edges = cleaned
        except Exception:
            pass

        # 3) black lines on top of quantized color
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = quant.copy()
        # edge == 255 -> set pixel to black
        result[edges_bgr[:, :, 0] > 0] = (0, 0, 0)

        self.vector_base = result.copy()
        return result

    # --- DİĞERLERİ ---
    def process_sketch_style(self):
        xdog = self.get_xdog(self.img, phi=250)
        res = cv2.cvtColor(xdog, cv2.COLOR_GRAY2BGR)
        self.vector_base = res
        return res

    def process_lineart(self, edge_thickness=2):
        face_mask = self.get_face_mask(self.img)
        if face_mask is None:
            face_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        xdog = self.get_xdog(self.img, gamma=0.98, phi=160, epsilon=0.02)
        hair = self.get_hair_flow_portrait(self.img, face_mask)

        combined = cv2.min(xdog, hair)

        if edge_thickness > 1:
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            combined = cv2.erode(combined, kernel, iterations=1)

        res = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        self.vector_base = res.copy()
        return res

    def process_logo_style(self, color_count=8):
        """
        Logo/Grafik için daha 'temiz' taban üretir:
        - Doku/gürültü azaltma
        - Posterize / renk azaltma
        Not: Bu çıktı, SVG motoru ile katmanlara ayrılıp path'e çevrilir.
        """
        # Keskinlik için hafif median + bilateral
        base = cv2.medianBlur(self.img, 3)
        base = cv2.bilateralFilter(base, 9, 75, 75)

        k = int(max(2, min(color_count, 24)))
        q = self.quantize(base, k)
        self.vector_base = q
        return q

    def process_artistic_style(self, style="cartoon", options=None):
        if options is None: options = {}
        edge = options.get("edge_thickness", 2)
        colors = options.get("color_count", 16)
        border_sens = options.get("border_sensitivity", 12)
        
        # Nakışa uygun flat preset
        if style == "flat":
            base = self.process_flat_cartoon(edge_thickness=edge, color_count=colors, border_sensitivity=border_sens)
        else:
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
    def generate_artistic_svg(
        self,
        num_colors=16,
        simplify_factor=0.003,
        stroke_width=1,
        min_area=20,
        cleanup_kernel=3,
        alpha_threshold=10,
        ignore_background=True,
    ):
        """
        ERC V4 için optimize edilmiş SVG motoru.
        İyileştirmeler:
        - Delik/boşluk (holes) desteği: RETR_CCOMP + fill-rule=evenodd
        - Maske temizleme (open/close) ile gürültü azaltma
        - Alpha/foreground mask varsa background'u dışarıda tutma
        """
        img = self.vector_base if self.vector_base is not None else self.img
        # Sertleştirme (Blur Temizliği)
        img = cv2.pyrMeanShiftFiltering(img, 10, 40)

        k = max(2, min(int(num_colors), 32))
        h, w = img.shape[:2]
        svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']

        # Eğer img zaten quantize edilmişse (az sayıda renk), tekrar kmeans yapma.
        try:
            uniq = np.unique(img.reshape(-1, 3), axis=0)
        except Exception:
            uniq = None

        if uniq is not None and len(uniq) <= k:
            quantized = img
            unique_colors = uniq
        else:
            data = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            quantized = np.uint8(center)[label.flatten()].reshape(img.shape)
            unique_colors = np.unique(np.uint8(center), axis=0)

        # Arka plan rengi (alpha yoksa) -> en sık görülen border rengi
        bg_color = None
        if ignore_background and (self.alpha is None or self.fg_mask is None):
            try:
                border = np.concatenate([
                    quantized[0, :, :],
                    quantized[-1, :, :],
                    quantized[:, 0, :],
                    quantized[:, -1, :],
                ], axis=0)
                # border shape: (n, 3) BGR
                vals, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
                if vals is not None and len(vals) > 0:
                    bg_color = vals[int(np.argmax(counts))]
            except Exception:
                bg_color = None
        layers = []
        for color in unique_colors:
            mask = cv2.inRange(quantized, color, color)
            # bg rengi ise layer'ı tamamen atla (şeffaf arka plan)
            if bg_color is not None and np.array_equal(color, bg_color):
                continue
            # foreground mask varsa background'u hariç tut
            if self.alpha is not None and self.fg_mask is not None:
                # alpha threshold'u aşmayan bölgeleri temizle
                fg = ((self.alpha.astype(np.uint16)) > int(alpha_threshold)).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask, fg)

            # maske temizliği (gürültü)
            ksz = int(max(0, cleanup_kernel))
            if ksz >= 2:
                if ksz % 2 == 0:
                    ksz += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            area = cv2.countNonZero(mask)
            hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
            is_black = (color[0] < 45 and color[1] < 45 and color[2] < 45)
            layers.append({"hex": hex_color, "area": area, "is_black": is_black, "mask": mask})

        # Büyükten küçüğe, siyahlar en son
        layers.sort(key=lambda x: x["area"], reverse=True)

        for layer in layers:
            if layer["is_black"]: continue
            svg_output.append(self._get_svg_path(layer["mask"], layer["hex"], simplify_factor, min_area=min_area))
        
        for layer in layers:
            if not layer["is_black"]: continue
            svg_output.append(self._get_svg_path(layer["mask"], layer["hex"], simplify_factor, min_area=min_area))

        svg_output.append('</svg>')
        return "".join(svg_output)

    def generate_eps(
        self,
        num_colors=16,
        simplify_factor=0.003,
        min_area=20,
        cleanup_kernel=3,
        alpha_threshold=10,
        ignore_background=True,
    ):
        """
        Basit EPS üretimi (polygon path).
        Not: Bezier fitting/potrace seviyesinde değil; ama çoğu iş akışında temiz, katmanlı EPS verir.
        """
        img = self.vector_base if self.vector_base is not None else self.img
        img = cv2.pyrMeanShiftFiltering(img, 10, 40)

        # EPS katmanları için 2 renge kadar izin ver (flat/logo kullanımında).
        k = max(2, min(int(num_colors), 32))
        try:
            uniq = np.unique(img.reshape(-1, 3), axis=0)
        except Exception:
            uniq = None

        if uniq is not None and len(uniq) <= k:
            quantized = img
            center = uniq
        else:
            data = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            quantized = np.uint8(center)[label.flatten()].reshape(img.shape)
            center = np.uint8(center)

        h, w = img.shape[:2]

        # bg tespiti (alpha yoksa)
        bg_color = None
        if ignore_background and (self.alpha is None or self.fg_mask is None):
            try:
                border = np.concatenate([
                    quantized[0, :, :],
                    quantized[-1, :, :],
                    quantized[:, 0, :],
                    quantized[:, -1, :],
                ], axis=0)
                vals, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
                if vals is not None and len(vals) > 0:
                    bg_color = vals[int(np.argmax(counts))]
            except Exception:
                bg_color = None

        unique_colors = np.unique(np.uint8(center), axis=0)
        layers = []
        for color in unique_colors:
            if bg_color is not None and np.array_equal(color, bg_color):
                continue
            mask = cv2.inRange(quantized, color, color)
            if self.alpha is not None and self.fg_mask is not None:
                fg = ((self.alpha.astype(np.uint16)) > int(alpha_threshold)).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask, fg)

            ksz = int(max(0, cleanup_kernel))
            if ksz >= 2:
                if ksz % 2 == 0:
                    ksz += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            area = cv2.countNonZero(mask)
            is_black = (color[0] < 45 and color[1] < 45 and color[2] < 45)
            layers.append({"color": color, "area": area, "is_black": is_black, "mask": mask})

        layers.sort(key=lambda x: x["area"], reverse=True)

        out = []
        out.append(b"%!PS-Adobe-3.0 EPSF-3.0\n")
        out.append(f"%%BoundingBox: 0 0 {w} {h}\n".encode("ascii"))
        out.append(b"%%LanguageLevel: 2\n")
        out.append(b"%%EndComments\n")
        out.append(b"/m {moveto} bind def\n/l {lineto} bind def\n/cp {closepath} bind def\n")
        out.append(b"gsave\n")

        # renkli katmanlar önce, siyah kontur sonra
        for pass_black in (False, True):
            for layer in layers:
                if layer["is_black"] != pass_black:
                    continue
                bgr = layer["color"]
                r, g, b = float(bgr[2]) / 255.0, float(bgr[1]) / 255.0, float(bgr[0]) / 255.0
                cmds = self._mask_to_eps(layer["mask"], simplify_factor, min_area=min_area, canvas_h=h)
                if not cmds:
                    continue
                out.append(f"gsave {r:.4f} {g:.4f} {b:.4f} setrgbcolor\n".encode("ascii"))
                out.append(cmds)
                out.append(b"grestore\n")

        out.append(b"grestore\nshowpage\n%%EOF\n")
        return b"".join(out)

    def _mask_to_eps(self, mask, epsilon_factor, min_area=20, canvas_h=0):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        if hierarchy is None or len(contours) == 0:
            return b""
        h = hierarchy[0]

        def approx_pts(cnt):
            if cv2.contourArea(cnt) < float(min_area):
                return None
            epsilon = float(epsilon_factor) * cv2.arcLength(cnt, True)
            ap = cv2.approxPolyDP(cnt, epsilon, True)
            if ap is None or len(ap) < 3:
                return None
            pts = [(int(p[0][0]), int(p[0][1])) for p in ap]
            return pts

        parts = []
        for i, cnt in enumerate(contours):
            if int(h[i][3]) != -1:
                continue
            outer = approx_pts(cnt)
            if not outer:
                continue
            subpaths = [outer]
            child = int(h[i][2])
            while child != -1:
                hole = approx_pts(contours[child])
                if hole:
                    subpaths.append(hole)
                child = int(h[child][0])

            # EPS: tek path içinde outer+holes ve even-odd fill => eofill
            parts.append(b"newpath\n")
            for pts in subpaths:
                # coordinate flip (top-left -> bottom-left)
                x0, y0 = pts[0]
                y0 = (canvas_h - y0) if canvas_h else y0
                parts.append(f"{x0} {y0} m\n".encode("ascii"))
                for x, y in pts[1:]:
                    y = (canvas_h - y) if canvas_h else y
                    parts.append(f"{x} {y} l\n".encode("ascii"))
                parts.append(b"cp\n")
            parts.append(b"eofill\n")

        return b"".join(parts)

    def generate_bot_json(
        self,
        num_colors=4,
        simplify_factor=0.003,
        min_area=20,
        cleanup_kernel=3,
        alpha_threshold=10,
        ignore_background=True,
    ):
        """
        BOT v1 (JSON). Bu, Wilcom gibi programlarda direkt açılmaz; Botlab içinde editlenebilir temel objeler üretir.
        """
        img = self.vector_base if self.vector_base is not None else self.img
        img = cv2.pyrMeanShiftFiltering(img, 10, 40)

        # BOT katmanları için 2 renge kadar izin ver (flat/logo kullanımında).
        k = max(2, min(int(num_colors), 16))
        try:
            uniq = np.unique(img.reshape(-1, 3), axis=0)
        except Exception:
            uniq = None

        if uniq is not None and len(uniq) <= k:
            quantized = img
            center = uniq
        else:
            data = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            quantized = np.uint8(center)[label.flatten()].reshape(img.shape)
            center = np.uint8(center)

        h, w = img.shape[:2]

        bg_color = None
        if ignore_background and (self.alpha is None or self.fg_mask is None):
            try:
                border = np.concatenate([
                    quantized[0, :, :],
                    quantized[-1, :, :],
                    quantized[:, 0, :],
                    quantized[:, -1, :],
                ], axis=0)
                vals, counts = np.unique(border.reshape(-1, 3), axis=0, return_counts=True)
                if vals is not None and len(vals) > 0:
                    bg_color = vals[int(np.argmax(counts))]
            except Exception:
                bg_color = None

        def clean_mask(mask):
            ksz = int(max(0, cleanup_kernel))
            if ksz >= 2:
                if ksz % 2 == 0:
                    ksz += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            return mask

        objects = []
        obj_i = 1
        unique_colors = np.unique(np.uint8(center), axis=0)
        for color in unique_colors:
            if bg_color is not None and np.array_equal(color, bg_color):
                continue
            mask = cv2.inRange(quantized, color, color)
            if self.alpha is not None and self.fg_mask is not None:
                fg = ((self.alpha.astype(np.uint16)) > int(alpha_threshold)).astype(np.uint8) * 255
                mask = cv2.bitwise_and(mask, fg)
            mask = clean_mask(mask)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            if hierarchy is None or len(contours) == 0:
                continue
            hh = hierarchy[0]

            rgb = [int(color[2]), int(color[1]), int(color[0])]

            for idx, cnt in enumerate(contours):
                if int(hh[idx][3]) != -1:
                    continue
                if cv2.contourArea(cnt) < float(min_area):
                    continue
                eps = float(simplify_factor) * cv2.arcLength(cnt, True)
                ap = cv2.approxPolyDP(cnt, eps, True)
                if ap is None or len(ap) < 3:
                    continue
                pts = [[int(p[0][0]), int(p[0][1])] for p in ap]

                objects.append({
                    "id": f"obj_{obj_i:03d}",
                    "type": "tatami",
                    "points": pts,
                    "angle": 45,
                    "density": 4.0,
                    "pull_comp": 0.3,
                    "underlay": {"type": "zigzag", "density": 1.2},
                    "thread": {"brand": "Generic", "color_code": None, "rgb": rgb},
                })
                obj_i += 1

        bot = {
            "version": "1.0",
            "metadata": {"width": int(w), "height": int(h), "unit": "px"},
            "objects": objects,
        }
        return json.dumps(bot, ensure_ascii=False)

    def _get_svg_path(self, mask, hex_color, epsilon_factor, min_area=20):
        """
        Delikleri koruyacak şekilde path üretir.
        - RETR_CCOMP: dış kontur + iç delikler hiyerarşi ile gelir
        - fill-rule=evenodd: deliklerin otomatik boş kalmasını sağlar
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        if hierarchy is None or len(contours) == 0:
            return ""

        h = hierarchy[0]

        def contour_to_subpath(cnt):
            if cv2.contourArea(cnt) < float(min_area):
                return ""
            epsilon = float(epsilon_factor) * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if approx is None or len(approx) < 3:
                return ""
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            # "M x y L x y ... Z"
            d = f"M {pts[0][0]} {pts[0][1]}"
            for x, y in pts[1:]:
                d += f" L {x} {y}"
            d += " Z"
            return d

        path_parts = []
        for i, cnt in enumerate(contours):
            # parent == -1 => dış kontur
            if int(h[i][3]) != -1:
                continue
            outer = contour_to_subpath(cnt)
            if not outer:
                continue
            d_parts = [outer]

            # child chain: delikler (ve onların sibling'ları)
            child = int(h[i][2])
            while child != -1:
                hole = contour_to_subpath(contours[child])
                if hole:
                    d_parts.append(hole)
                child = int(h[child][0])  # next sibling

            path_parts.append(f'<path d="{" ".join(d_parts)}" fill="{hex_color}" stroke="none" fill-rule="evenodd"/>')

        return "".join(path_parts)
    
    def get_dominant_color(self, img, k=1):
        return [0,0,0]
