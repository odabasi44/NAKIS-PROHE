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
            closes
