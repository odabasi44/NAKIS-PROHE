import cv2
import numpy as np
import base64
import json

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

    def __init__(self, image_stream, target_width=None, target_height=None, lock_aspect=True, max_dim=1000):
        file_bytes = np.frombuffer(image_stream.read(), np.uint8)
        self.original_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        self.vector_base = None
        self.alpha = None  # varsa 0-255
        self.fg_mask = None  # varsa 0/255 tek kanal (foreground)

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
        else:
            # default optimize
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                self.img = cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.h, self.w = self.img.shape[:2]

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
        hair_lines = self._thinning(hair_lines)
        
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
        fdog = self._thinning(fdog)
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

    def process_flat_cartoon(self, edge_thickness=2, color_count=10, border_sensitivity=12):
        """
        Nakış/Wilcom için daha uygun 'flat' görünüm:
        - Az renk (posterize)
        - Bölge sınırlarından temiz kontur (renk geçişi sınırı)
        - Saç/kırışıklık gibi mikro çizgiler yok (daha az node, daha temiz SVG)
        """
        # 1) yumuşatma + renk azaltma
        base = cv2.bilateralFilter(self.img, 9, 90, 90)
        k = int(max(4, min(color_count, 24)))
        quant = self.quantize(base, k)

        # 2) sınır/contour: komşu piksellerin renk farkından edge mask
        gray = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
        # Laplacian sınırları yakalar; threshold ile ikili maske
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        lap = cv2.convertScaleAbs(lap)
        _, edges = cv2.threshold(lap, int(max(1, border_sensitivity)), 255, cv2.THRESH_BINARY)

        if edge_thickness > 1:
            ksz = int(edge_thickness)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            edges = cv2.dilate(edges, kernel, iterations=1)

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

        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = max(4, min(num_colors, 32))
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(img.shape)

        h, w = img.shape[:2]
        svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']

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

        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = max(3, min(int(num_colors), 32))
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(img.shape)

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

        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = max(3, min(int(num_colors), 16))
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(img.shape)

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
