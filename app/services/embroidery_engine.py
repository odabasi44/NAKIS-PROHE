import io
import cv2
import numpy as np
from app.services.vector_engine import AdvancedVectorEngine

try:
    import pyembroidery
    HAS_EMBROIDERY = True
except ImportError:
    HAS_EMBROIDERY = False

class EmbroideryGenerator:
    def __init__(self, image_stream):
        # Vektör motorunu kullanarak görüntüyü temizle
        vec_engine = AdvancedVectorEngine(image_stream)
        # Nakış için renk sayısını düşük tutuyoruz (max 12)
        # Main.py'deki mantıkla aynı metodları çağırıyoruz
        # Ancak vector_engine içindeki metodları kopyaladığınızdan emin olun
        
        # process_hybrid_cartoon metodunun vector_engine içinde olduğunu varsayıyoruz
        if hasattr(vec_engine, 'process_hybrid_cartoon'):
            vec_engine.process_hybrid_cartoon(edge_thickness=2, color_count=12)
            self.img = vec_engine.vector_base 
        else:
            self.img = vec_engine.img

        self.img = cv2.pyrMeanShiftFiltering(self.img, 15, 50)

    def generate_pattern(self, file_format="dst"):
        if not HAS_EMBROIDERY:
            raise ImportError("pyembroidery yüklü değil.")

        pattern = pyembroidery.EmbPattern()
        
        data = np.float32(self.img).reshape((-1, 3))
        k = 10 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        quantized = np.uint8(center)[label.flatten()].reshape(self.img.shape)
        unique_colors = np.unique(np.uint8(center), axis=0)

        SCALE = 1.0 
        
        color_change_cmd = getattr(pyembroidery, "COLOR_CHANGE", None)
        first_color = True

        for color in unique_colors:
            # Siyah renkleri (arkaplan olabilir) atla veya işle
            # is_black = (color[0] < 40 and color[1] < 40 and color[2] < 40)
            
            mask = cv2.inRange(quantized, color, color)
            # Renk değişimi: ilk thread'de COLOR_CHANGE eklemeyelim
            if (not first_color) and (color_change_cmd is not None):
                pattern.add_command(color_change_cmd)
            pattern.add_thread(pyembroidery.EmbThread(color[2], color[1], color[0])) # RGB
            first_color = False
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < 50: continue 
                
                approx = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)
                for point in approx:
                    x, y = point[0]
                    pattern.add_stitch_absolute(pyembroidery.STITCH, x * SCALE, y * SCALE)
                
                x, y = approx[0][0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, x * SCALE, y * SCALE)
                pattern.add_command(pyembroidery.JUMP)

        out_stream = io.BytesIO()
        fmt = file_format.lower()
        if fmt == "emb": fmt = "dst"
        
        # Format yazma
        if fmt == "dst": pyembroidery.write_dst(pattern, out_stream)
        elif fmt == "pes": pyembroidery.write_pes(pattern, out_stream)
        elif fmt == "exp": pyembroidery.write_exp(pattern, out_stream)
        elif fmt == "jef": pyembroidery.write_jef(pattern, out_stream)
        elif fmt == "vp3": pyembroidery.write_vp3(pattern, out_stream)
        elif fmt == "xxx": pyembroidery.write_xxx(pattern, out_stream)
        else: pyembroidery.write_dst(pattern, out_stream)

        return out_stream.getvalue()
