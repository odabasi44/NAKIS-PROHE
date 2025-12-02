import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from PyPDF2 import PdfMerger
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ============================================================
# FLASK UYGULAMASI
# ============================================================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vektor')
def vektor():
    return render_template('vector.html')

@app.route('/pdf/merge')
def pdf_merge():
    return render_template('pdf_merge.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')

# ============================================================
# PDF BİRLEŞTİRME API
# ============================================================

@app.route('/api/pdf/merge', methods=['POST', 'OPTIONS'])
def api_pdf_merge():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        print("PDF Merge endpoint çağrıldı")
        
        # Kullanıcı kontrolü (basit implementasyon)
        # Gerçek implementasyonda session/cookie/database kontrolü
        user_email = request.cookies.get('user_email') or 'guest'
        is_premium = check_if_premium(user_email)
        
        print(f"Kullanıcı: {user_email}, Premium: {is_premium}")
        
        if 'pdf_files' not in request.files:
            return jsonify({
                "success": False,
                "message": "Lütfen PDF dosyaları seçin.",
            }), 400
        
        files = request.files.getlist("pdf_files")
        print(f"Toplam dosya: {len(files)}")
        
        # LIMIT KONTROLLERİ
        if not is_premium:
            # ÜCRETSİZ KULLANICI LİMİTLERİ
            MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB
            MAX_FILES = 10
            MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB per file
            
            # TODO: Günlük 2 kullanım limiti için database gerekli
            # daily_usage = get_daily_usage(user_email)
            # if daily_usage >= 2:
            #     return jsonify({
            #         "success": False,
            #         "message": "Günlük ücretsiz limitiniz doldu (2/2). Premium'a yükseltin.",
            #         "upgrade_url": "/premium"
            #     }), 403
        else:
            # PREMIUM KULLANICI LİMİTLERİ
            MAX_TOTAL_SIZE = 150 * 1024 * 1024  # 150MB
            MAX_FILES = 25
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
        
        # Dosya sayısı kontrolü
        if len(files) > MAX_FILES:
            return jsonify({
                "success": False,
                "message": f"Çok fazla dosya seçtiniz. {'Premium' if not is_premium else ''} maksimum {MAX_FILES} dosya destekleniyor.",
                "upgrade_url": "/premium" if not is_premium else None
            }), 400
        
        pdf_files = []
        total_size = 0
        
        for f in files:
            if f and f.filename:
                # Dosya uzantısı kontrolü
                if not f.filename.lower().endswith('.pdf'):
                    continue
                
                # Tek dosya boyutu kontrolü
                f.seek(0, 2)  # Sona git
                file_size = f.tell()
                f.seek(0)  # Başa dön
                
                if file_size > MAX_FILE_SIZE:
                    size_mb = MAX_FILE_SIZE / (1024 * 1024)
                    return jsonify({
                        "success": False,
                        "message": f"{f.filename} çok büyük. Tek dosya limiti: {size_mb}MB",
                        "upgrade_url": "/premium" if not is_premium else None
                    }), 413
                
                # Toplam boyut kontrolü
                total_size += file_size
                if total_size > MAX_TOTAL_SIZE:
                    total_mb = MAX_TOTAL_SIZE / (1024 * 1024)
                    return jsonify({
                        "success": False,
                        "message": f"Toplam boyut limiti aşıldı. {'Premium' if not is_premium else ''} maksimum: {total_mb}MB",
                        "upgrade_url": "/premium" if not is_premium else None
                    }), 413
                
                pdf_files.append(f)
                print(f"Geçerli PDF: {f.filename} ({file_size/1024:.1f} KB)")
        
        if len(pdf_files) < 2:
            return jsonify({
                "success": False,
                "message": f"En az 2 PDF dosyası gerekiyor (seçilen: {len(pdf_files)}).",
            }), 400
        
        print(f"İşlenecek PDF'ler: {len(pdf_files)} dosya, Toplam: {total_size/1024/1024:.2f} MB")
        
        # BİRLEŞTİRME İŞLEMİ
        merger = PdfMerger()
        
        for pdf in pdf_files:
            pdf.seek(0)
            merger.append(io.BytesIO(pdf.read()))
        
        output = io.BytesIO()
        merger.write(output)
        merger.close()
        output.seek(0)
        
        pdf_data = output.read()
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        response = jsonify({
            "success": True,
            "file": pdf_base64,
            "filename": "birlesik.pdf",
            "message": f"{len(pdf_files)} PDF dosyası başarıyla birleştirildi.",
            "limits": {
                "is_premium": is_premium,
                "max_size_mb": MAX_TOTAL_SIZE / (1024 * 1024),
                "max_files": MAX_FILES
            }
        })
        
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
        
    except Exception as e:
        print("PDF Merge Error:", str(e))
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "message": f"Birleştirme hatası: {str(e)}"
        }), 500

# ============================================================
# DİĞER ENDPOINT'LER (VEKTÖRLEŞTİRME)
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

        # Vektörleştirme işlemleri burada...
        # (mevcut kodunuz)
        
        vector_base64 = "base64_encoded_image"  # Placeholder

        return jsonify({
            'success': True,
            'image_data': vector_base64,
        })

    except Exception as e:
        print("vectorize_style hata:", e)
        return jsonify({'success': False, 'error': 'İşleme sırasında hata oluştu.'}), 500

# ============================================================
# ÇALIŞTIRMA
# ============================================================

if __name__ == '__main__':
    # Port kontrolü
    port = int(os.environ.get('PORT', 5001))
    
    # Debug için tüm endpoint'leri listele
    print("\n" + "="*50)
    print("AVAILABLE ENDPOINTS:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.rule}")
    print("="*50 + "\n")
    
    print(f"Server starting on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

