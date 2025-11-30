from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
from collections import Counter

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze_colors', methods=['POST'])
def analyze_colors():
    """Görseldeki renkleri analiz et"""
    try:
        file = request.files['image']
        max_colors = int(request.form.get('max_colors', 20))
        
        file_stream = io.BytesIO(file.read())
        image = Image.open(file_stream)
        image = image.convert('RGB')
        
        # Görsel boyutunu küçült (performans için)
        width, height = image.size
        if width > 400 or height > 400:
            ratio = min(400/width, 400/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        
        # Tüm renkleri topla
        pixels = img_array.reshape(-1, 3)
        
        # En çok kullanılan renkleri bul
        color_counts = Counter(map(tuple, pixels))
        dominant_colors = color_counts.most_common(max_colors)
        
        # Renkleri formatla
        colors = []
        for color, count in dominant_colors:
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            percentage = (count / len(pixels)) * 100
            colors.append({
                'rgb': color,
                'hex': hex_color,
                'percentage': round(percentage, 2),
                'count': count
            })
        
        return jsonify({
            'success': True,
            'colors': colors,
            'total_colors': len(dominant_colors)
        })
        
    except Exception as e:
        print(f"Renk analiz hatası: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vectorize_custom', methods=['POST'])
def vectorize_custom():
    """Özel renk paleti ile vektörleştir"""
    try:
        file = request.files['image']
        selected_colors = json.loads(request.form.get('selected_colors', '[]'))
        style = request.form.get('style', 'sharp')
        output_format = request.form.get('format', 'png')
        
        file_stream = io.BytesIO(file.read())
        image = Image.open(file_stream)
        image = image.convert('RGB')
        
        # Seçilen renkleri numpy array'e çevir
        custom_colors = np.array([color['rgb'] for color in selected_colors], dtype=np.uint8)
        
        if output_format == 'png':
            result_data = vectorize_with_custom_colors(image, custom_colors, style)
            return jsonify({
                'success': True, 
                'image_data': result_data,
                'format': 'png'
            })
        else:
            svg = vectorize_with_custom_colors_svg(image, custom_colors, style)
            return jsonify({
                'success': True, 
                'svg': svg,
                'format': 'svg'
            })
        
    except Exception as e:
        print(f"Özel vektörleştirme hatası: {e}")
        return jsonify({'error': str(e)}), 500

def vectorize_with_custom_colors(image, custom_colors, style):
    """Özel renk paleti ile PNG oluştur"""
    try:
        width, height = image.size
        max_size = 600
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        
        # Özel renklerle segmentasyon
        data = img_array.reshape((-1, 3))
        data = np.float32(data)
        
        # En yakın renkleri bul
        segmented_data = []
        for pixel in data:
            distances = np.sqrt(np.sum((custom_colors - pixel) ** 2, axis=1))
            closest_color_idx = np.argmin(distances)
            segmented_data.append(custom_colors[closest_color_idx])
        
        segmented_image = np.array(segmented_data, dtype=np.uint8).reshape(img_array.shape)
        
        if style == 'sharp':
            segmented_image = sharpen_edges_improved(segmented_image)
        elif style == 'cartoon':
            # Özel paletli cartoon için eski efekti kullanmaya devam edelim
            segmented_image = apply_cartoon_effect(segmented_image)
        
        png_image = Image.fromarray(segmented_image)
        buffered = io.BytesIO()
        png_image.save(buffered, format="PNG", optimize=True, quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Özel PNG vektörleştirme hatası: {e}")
        raise e

def vectorize_with_custom_colors_svg(image, custom_colors, style):
    """Özel renk paleti ile SVG oluştur"""
    try:
        width, height = image.size
        max_size = 600
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = new_width, new_height
        
        img_array = np.array(image)
        
        # Özel renklerle segmentasyon
        data = img_array.reshape((-1, 3))
        data = np.float32(data)
        
        segmented_data = []
        for pixel in data:
            distances = np.sqrt(np.sum((custom_colors - pixel) ** 2, axis=1))
            closest_color_idx = np.argmin(distances)
            segmented_data.append(custom_colors[closest_color_idx])
        
        segmented_image = np.array(segmented_data, dtype=np.uint8).reshape(img_array.shape)
        
        if style == 'sharp':
            segmented_image = sharpen_edges_improved(segmented_image)
        
        return create_sharp_svg(segmented_image, custom_colors, width, height)
        
    except Exception as e:
        print(f"Özel SVG vektörleştirme hatası: {e}")
        raise e

def apply_cartoon_effect(image):
    """ESKİ cartoon efekti (custom renk paleti için kullanılıyor)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 9, 2
    )
    
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(image, edges_colored)
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=10)
    
    return cartoon

def sharpen_edges_improved(image):
    """Kenar keskinleştirme"""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.7, blurred, -0.7, 0)
    
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
    sharpened = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def create_sharp_svg(segmented_image, colors, width, height):
    """SVG oluştur"""
    svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
    svg += f'<rect width="100%" height="100%" fill="white"/>\n'
    
    for i, color in enumerate(colors):
        color_bgr = color.tolist()
        mask = np.all(segmented_image == color_bgr, axis=-1).astype(np.uint8) * 255
        
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        
        for contour in contours:
            if len(contour) > 2:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) > 2:
                    path_data = "M "
                    for j, point in enumerate(approx):
                        x, y = point[0]
                        if j == 0:
                            path_data += f"{x} {y}"
                        else:
                            path_data += f" L {x} {y}"
                    path_data += " Z"
                    
                    svg += f'<path d="{path_data}" fill="{hex_color}" stroke="none"/>\n'
    
    svg += '</svg>'
    return svg

# ============================================================
# YENİ CARTOON VECTOR ALGORİTMASI (PNG için)
# ============================================================

def cartoon_vectorize(image_rgb, k=4):
    """
    Renkleri k kümeye düşürüp keskin kenarlı cartoon vector görünümü üretir.
    image_rgb: RGB numpy array
    k: renk sayısı (2–6 arası ideal)
    """
    # Hafif blur
    img_color = cv2.medianBlur(image_rgb, 7)

    # Renk kuantizasyonu (k-means)
    Z = img_color.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    img_reduced = res.reshape(img_color.shape)

    # Kenar çıkarma
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 140)
    edges = cv2.dilate(edges, None)
    edges = 255 - edges  # çizgileri beyaz yap
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Renk + kenar maskesi ile combine
    cartoon = cv2.bitwise_and(img_reduced, edges_colored)

    return cartoon

# ============================================================
# Ana endpoint: /vectorize_style
# ============================================================

@app.route('/vectorize_style', methods=['POST'])
def vectorize_with_style():
    try:
        file = request.files['image']
        colors = int(request.form.get('colors', 8))
        style = request.form.get('style', 'sharp')
        output_format = request.form.get('format', 'png')
        
        file_stream = io.BytesIO(file.read())
        image = Image.open(file_stream)
        image = image.convert('RGB')
        
        if output_format == 'png':
            if style == 'sharp':
                result_data = vectorize_to_png(image, colors)
            elif style == 'line_art':
                result_data = create_line_art(image)
            else:  # cartoon
                # Cartoon için renk sayısını 2–6 aralığına sıkıştırıyoruz
                cartoon_colors = max(2, min(colors, 6))
                result_data = create_cartoon_effect_vector(image, cartoon_colors)
            return jsonify({
                'success': True, 
                'image_data': result_data,
                'format': 'png'
            })
        else:
            if style == 'sharp':
                svg = vectorize_with_colors_sharp(image, colors)
            elif style == 'line_art':
                svg = create_line_art_svg(image)
            else:  # cartoon
                cartoon_colors = max(2, min(colors, 6))
                svg = create_cartoon_svg(image, cartoon_colors)
            return jsonify({
                'success': True, 
                'svg': svg,
                'format': 'svg'
            })
        
    except Exception as e:
        print(f"Vectorize style error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# Yardımcı / fallback fonksiyonlar
# ============================================================

def _image_to_data_url(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def vectorize_to_png(image, colors=8):
    """Hafif bir renk-kuantizasyonu yapıp base64 PNG döndürür."""
    try:
        width, height = image.size
        max_size = 600
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        arr = np.array(image)
        data = arr.reshape((-1, 3)).astype(np.float32)

        # OpenCV k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(data, colors, None, criteria, 10, flags)
        centers = np.uint8(centers)
        quant = centers[labels.flatten()]
        quant_image = quant.reshape(arr.shape)

        quant_image = sharpen_edges_improved(quant_image)

        pil = Image.fromarray(quant_image)
        return _image_to_data_url(pil)
    except Exception as e:
        print(f"vectorize_to_png error: {e}")
        raise


def create_line_art(image):
    """Canny kenar algılama ile siyah-beyaz line-art PNG data URL döndürür."""
    try:
        arr = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        inv = 255 - edges
        img_rgb = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(img_rgb)
        return _image_to_data_url(pil)
    except Exception as e:
        print(f"create_line_art error: {e}")
        raise


def create_cartoon_effect_vector(image, colors=4):
    """
    Yeni cartoon vector algoritmasını kullanıp PNG data URL döndürür.
    colors parametresi 2–6 arası k-means renk sayısıdır.
    """
    try:
        arr = np.array(image.convert('RGB'))

        # Renk sayısını güvenli aralıkta tut
        k = max(2, min(colors, 6))
        cartoon = cartoon_vectorize(arr, k=k)

        pil = Image.fromarray(cartoon)
        return _image_to_data_url(pil)
    except Exception as e:
        print(f"create_cartoon_effect_vector error: {e}")
        raise


def vectorize_with_colors_sharp(image, colors=8):
    """Renk sayısına göre kuantize edip `create_sharp_svg` çağırır."""
    try:
        width, height = image.size
        max_size = 600
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = new_width, new_height

        arr = np.array(image)
        data = arr.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(data, colors, None, criteria, 10, flags)
        centers = np.uint8(centers)

        segmented = centers[labels.flatten()].reshape(arr.shape)
        return create_sharp_svg(segmented, centers, width, height)
    except Exception as e:
        print(f"vectorize_with_colors_sharp error: {e}")
        raise


def create_line_art_svg(image):
    """Fallback SVG: raster line-art PNG'yi gömerek SVG string döndürür."""
    try:
        png_data = create_line_art(image)
        if png_data.startswith('data:image/png;base64,'):
            b64 = png_data.split(',', 1)[1]
        else:
            b64 = png_data
        width, height = image.size
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        svg += f'<image href="data:image/png;base64,{b64}" width="{width}" height="{height}"/>'
        svg += '</svg>'
        return svg
    except Exception as e:
        print(f"create_line_art_svg error: {e}")
        raise


def create_cartoon_svg(image, colors=4):
    """Yeni cartoon vector PNG'yi SVG içine gömerek SVG string döndürür."""
    try:
        png_data = create_cartoon_effect_vector(image, colors)
        if png_data.startswith('data:image/png;base64,'):
            b64 = png_data.split(',', 1)[1]
        else:
            b64 = png_data
        width, height = image.size
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        svg += f'<image href="data:image/png;base64,{b64}" width="{width}" height="{height}"/>'
        svg += '</svg>'
        return svg
    except Exception as e:
        print(f"create_cartoon_svg error: {e}")
        raise

if __name__ == '__main__':
    # Disable the reloader to keep a single process (helps automated runs)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
