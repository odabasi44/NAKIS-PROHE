import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
from PyPDF2 import PdfMerger

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def allowed_image(filename):
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed_extensions

def load_image_from_file(file_storage):
    image_bytes = file_storage.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Resim okunamadı.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def cartoon_vectorize(image_rgb, k=8, edge_th1=50, edge_th2=150):
    blurred = cv2.medianBlur(image_rgb, 5)
    z = blurred.reshape((-1, 3))
    z = np.float32(z)

    k = max(2, min(12, k))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(blurred.shape)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, edge_th1, edge_th2)
    edges_dilated = cv2.dilate(edges, None, iterations=1)
    edges_inv = cv2.bitwise_not(edges_dilated)
    edges_rgb = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2RGB)

    result = cv2.bitwise_and(quantized, edges_rgb)
    return result

def normal_vector(image_rgb, k=8):
    blurred = cv2.medianBlur(image_rgb, 5)
    z = blurred.reshape((-1, 3))
    z = np.float32(z)

    k = max(2, min(12, k))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(blurred.shape)
    return quantized

def line_art_vector(image_rgb, edge_th1=50, edge_th2=150):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray_blur, edge_th1, edge_th2)
    edges_dilated = cv2.dilate(edges, None, iterations=1)

    h, w = gray.shape
    line_img = np.full((h, w), 255, dtype=np.uint8)
    line_img[edges_dilated > 0] = 0

    result_rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
    return result_rgb

def convert_to_svg(image_rgb, threshold=128):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = thresh.shape

    svg_paths = []
    for cnt in contours:
        if len(cnt) < 3:
            continue

        approx = cv2.approxPolyDP(cnt, 1.0, True)

        d_commands = []
        for i, point in enumerate(approx):
            x, y = point[0]
            if i == 0:
                d_commands.append(f"M {x} {y}")
            else:
                d_commands.append(f"L {x} {y}")
        d_commands.append("Z")

        path_data = " ".join(d_commands)
        svg_paths.append(path_data)

    svg_path_elements = "\n".join(
        f'<path d="{path_data}" fill="black" stroke="none" />' for path_data in svg_paths
    )

    svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}"
     viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg">
{svg_path_elements}
</svg>
'''
    return svg_template

def encode_image_to_base64(image_rgb):
    image_pil = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    return base64.b64encode(png_bytes).decode("utf-8")

# ============================================================
# ANASAYFA
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# PDF BİRLEŞTİRME ARAÇLARI
# ============================================================

# ============================================================
# PDF ARAÇLARI API
# ============================================================


@app.route("/pdf/merge")
def pdf_merge_page():
    return render_template("pdf_merge.html")



@app.route("/api/pdf/merge", methods=["POST"])
def api_pdf_merge():
    """
    Birden fazla PDF dosyasını tek PDF'te birleştirir.
    Çıktıyı base64 olarak JSON içinde döner.
    """
    try:
        files = request.files.getlist("pdf_files")
        pdf_files = [f for f in files if f and f.filename.lower().endswith(".pdf")]

        if len(pdf_files) < 2:
            return (
                jsonify({
                    "success": False,
                    "message": "En az iki PDF dosyası seçmelisiniz.",
                }),
                400,
            )

        r = Pdfr()

        for f in pdf_files:
            r.append(io.BytesIO(f.read()))

        out_buffer = io.BytesIO()
        r.write(out_buffer)
        r.close()
        out_buffer.seek(0)

        pdf_bytes = out_buffer.read()
        encoded = base64.b64encode(pdf_bytes).decode("utf-8")

        return jsonify({
            "success": True,
            "file": encoded,          # base64 içerik
            "filename": "birlesik.pdf",
        })

    except Exception as e:
        print("api_pdf_ error:", e)
        return (
            jsonify({
                "success": False,
                "message": "Birleştirme sırasında sunucu hatası oluştu.",
            }),
            500,
        )



# ============================================================
# VEKTÖRLEŞTİRME API
# ============================================================

@app.route('/vectorize_style', methods=['POST'])
def vectorize_with_style():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Resim dosyası bulunamadı.'}), 400

        file = request.files['image']
        if not allowed_image(file.filename):
            return jsonify({'success': False, 'error': 'Geçersiz dosya türü.'}), 400

        image = load_image_from_file(file)

        style = request.form.get('style', 'cartoon')
        mode = request.form.get('mode', 'color')
        k = int(request.form.get('colors', 8))

        if style == 'cartoon':
            vector_image = cartoon_vectorize(image, k=k)
        elif style == 'normal':
            vector_image = normal_vector(image, k=k)
        elif style == 'lines':
            vector_image = line_art_vector(image)
        else:
            return jsonify({'success': False, 'error': 'Geçersiz stil seçildi.'}), 400

        if mode == 'bw':
            gray = cv2.cvtColor(vector_image, cv2.COLOR_RGB2GRAY)
            vector_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        vector_base64 = encode_image_to_base64(vector_image)

        svg_data = None
        if style in ['lines', 'normal', 'cartoon']:
            try:
                svg_data = convert_to_svg(vector_image)
                svg_base64 = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
            except Exception as e:
                print("SVG oluşturulurken hata:", e)
                svg_base64 = None
        else:
            svg_base64 = None

        return jsonify({
            'success': True,
            'image_data': vector_base64,
            'svg_data': svg_base64
        })

    except Exception as e:
        print("vectorize_style hata:", e)
        return jsonify({'success': False, 'error': 'İşleme sırasında hata oluştu.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




