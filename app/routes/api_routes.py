import io
import base64
import cv2
import numpy as np
import qrcode
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from flask import Blueprint, request, jsonify, session
from app.utils.helpers import check_user_status, increase_usage
from app.services.vector_engine import AdvancedVectorEngine
from app.services.embroidery_engine import EmbroideryGenerator
from app.services.ai_loader import AILoader

bp = Blueprint('api', __name__)

# --- VEKTÖR İŞLEMLERİ ---
@bp.route("/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    # Limit kontrolü
    if not check_user_status(email, "vector", "default")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403
        
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Görsel yüklenmedi"}), 400

    file = request.files["image"]
    method = request.form.get("method", "cartoon")
    style = request.form.get("style", "cartoon")
    edge_thickness = int(request.form.get("edge_thickness", 2))
    color_count = int(request.form.get("color_count", 16))

    try:
        engine = AdvancedVectorEngine(file)
        options = {"edge_thickness": edge_thickness, "color_count": color_count}
        
        if method == "outline":
            engine.img = engine.process_sketch_style()
        else:
            engine.img = engine.process_artistic_style(style=style, options=options)
        
        svg = engine.generate_artistic_svg(num_colors=color_count, simplify_factor=0.003)
        svg_b64 = base64.b64encode(svg.encode()).decode()
        
        _, buf = cv2.imencode(".png", engine.img)
        preview_b64 = base64.b64encode(buf).decode()
        
        increase_usage(email, "vector", "default")

        return jsonify({
            "success": True,
            "file": svg_b64,
            "preview_img": preview_b64,
            "info": {"style": style}
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/convert_embroidery", methods=["POST"])
def api_convert_embroidery():
    # Bu işlem genellikle vektör ile birlikte yapılır, ayrı limit koymuyoruz ama istenirse eklenebilir.
    if "image" not in request.files: 
        return jsonify({"success": False, "message": "Görsel yok"}), 400
    
    file = request.files["image"]
    fmt = request.form.get("format", "dst")

    try:
        emb_engine = EmbroideryGenerator(file)
        emb_data = emb_engine.generate_pattern(file_format=fmt)
        emb_b64 = base64.b64encode(emb_data).decode()
        
        return jsonify({
            "success": True, 
            "file": emb_b64, 
            "filename": f"design.{fmt}",
            "format": fmt
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- ARKA PLAN SİLME ---
@bp.route("/remove_bg", methods=["POST"])
def api_remove_bg():
    if not AILoader.u2net_session:
        return jsonify({"success": False, "reason": "AI Model Yüklenemedi"}), 503

    if "image" not in request.files: return jsonify({"success": False}), 400

    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403

    try:
        original = Image.open(request.files["image"]).convert("RGB")
        small = original.resize((320, 320))
        inp = np.transpose(np.array(small).astype(np.float32) / 255.0, (2, 0, 1))
        inp = np.expand_dims(inp, 0)

        out = AILoader.u2net_session.run(None, {AILoader.u2net_input_name: inp})[0].squeeze()
        mask = cv2.resize(out, original.size)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        rgba = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, "PNG")
        increase_usage(email, "image", "remove_bg")
        
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- RESİM VE PDF ARAÇLARI ---
@bp.route("/img/compress", methods=["POST"])
def api_img_compress():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "compress")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    try:
        quality = int(request.form.get("quality", 70))
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        
        new_size = buf.getbuffer().nbytes
        orig_size = request.content_length or new_size * 2 
        saving = int(100 - (new_size / orig_size * 100)) if orig_size > 0 else 0

        increase_usage(email, "image", "compress")
        return jsonify({
            "success": True, 
            "file": base64.b64encode(buf.getvalue()).decode(),
            "new_size": f"{new_size/1024:.1f} KB",
            "saving": str(max(0, saving))
        })
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "convert")["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    try:
        fmt = request.form.get("format", "jpeg").lower()
        fmt_map = {"jpg":"JPEG","jpeg":"JPEG","png":"PNG","webp":"WEBP","pdf":"PDF","ico":"ICO","bmp":"BMP","tiff":"TIFF","gif":"GIF"}
        
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        save_fmt = fmt_map.get(fmt, "JPEG")
        
        if save_fmt == "ICO":
             img.resize((256, 256)).save(buf, format="ICO")
        else:
             img.save(buf, save_fmt)
             
        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "pdf", "merge")["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403
    
    if "pdf_files" not in request.files: return jsonify({"success": False}), 400
    try:
        merger = PdfMerger()
        for f in request.files.getlist("pdf_files"): 
            merger.append(io.BytesIO(f.read()))
        out = io.BytesIO()
        merger.write(out)
        merger.close()
        increase_usage(email, "pdf", "merge")
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

# --- YENİ EKLENEN PDF FONKSİYONLARI ---

@bp.route("/pdf/split", methods=["POST"])
def api_pdf_split():
    email = session.get("user_email", "guest")
    # LİMİT KONTROLÜ EKLENDİ
    if not check_user_status(email, "pdf", "split")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403

    if "pdf_file" not in request.files: return jsonify({"success": False}), 400
    try:
        reader = PdfReader(request.files["pdf_file"])
        # Şimdilik sadece ilk sayfayı ayırır (Demo)
        writer = PdfWriter()
        if len(reader.pages) > 0:
            writer.add_page(reader.pages[0])
        
        out = io.BytesIO()
        writer.write(out)
        
        increase_usage(email, "pdf", "split")
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/pdf/compress", methods=["POST"])
def api_pdf_compress():
    email = session.get("user_email", "guest")
    # LİMİT KONTROLÜ EKLENDİ
    if not check_user_status(email, "pdf", "compress")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403

    if "pdf_file" not in request.files: return jsonify({"success": False}), 400
    try:
        reader = PdfReader(request.files["pdf_file"])
        writer = PdfWriter()
        for page in reader.pages:
            page.compress_content_streams() 
            writer.add_page(page)
        
        writer.add_metadata({}) 
        out = io.BytesIO()
        writer.write(out)
        
        increase_usage(email, "pdf", "compress")
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/word_to_pdf", methods=["POST"])
def api_word_to_pdf():
    # Bu özellik sunucu tarafında LibreOffice gerektirir. Şimdilik pasif.
    return jsonify({"success": False, "message": "Bu özellik için sunucu yapılandırması gerekiyor (LibreOffice)."}), 501

# --- QR & LOGO ---

@bp.route("/qr_generator", methods=["POST"])
def api_qr_generator():
    email = session.get("user_email", "guest")
    # LİMİT KONTROLÜ EKLENDİ
    if not check_user_status(email, "generator", "qr")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403

    data = request.json or request.form
    text = data.get("text", "https://botlab.tools")
    color = data.get("color", "black")
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color=color, back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    increase_usage(email, "generator", "qr")
    return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

@bp.route("/logo_generator", methods=["POST"])
def api_logo_generator():
    import random
    
    data = request.json or request.form
    text = data.get("text", "Marka").strip()
    # Markanın baş harfini al (ikon için)
    initial = text[0].upper() if text else "B"
    
    # Modern Renk Paletleri (Arka Plan, Şekil, Metin)
    palettes = [
        ((23, 23, 23), (59, 130, 246), (255, 255, 255)),   # Koyu Mod / Mavi
        ((255, 255, 255), (236, 72, 153), (30, 30, 30)),   # Beyaz / Pembe
        ((15, 23, 42), (16, 185, 129), (241, 245, 249)),   # Lacivert / Yeşil
        ((255, 251, 235), (245, 158, 11), (120, 53, 15)),  # Krem / Turuncu
        ((0, 0, 0), (255, 255, 255), (255, 255, 255))      # Siyah / Beyaz (Minimal)
    ]
    
    # Rastgele bir palet seç
    bg_color, shape_color, text_color = random.choice(palettes)
    
    # 512x512 Tuval Oluştur
    img = Image.new('RGB', (512, 512), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # --- Modern Arka Plan Şekilleri (Rastgele) ---
    shape_type = random.choice(['circle', 'square', 'rounded', 'outline'])
    
    # Merkez koordinatları
    cx, cy = 256, 220 
    size = 140 
    
    if shape_type == 'circle':
        d.ellipse([cx-size, cy-size, cx+size, cy+size], fill=shape_color)
    elif shape_type == 'square':
        d.rectangle([cx-size, cy-size, cx+size, cy+size], fill=shape_color)
    elif shape_type == 'rounded':
        # Köşeli yuvarlak (Basit simülasyon)
        d.ellipse([cx-size, cy-size, cx+size, cy+size], fill=shape_color)
        d.rectangle([cx-size+20, cy-size+20, cx+size-20, cy+size-20], fill=bg_color)
        d.text((cx-40, cy-60), initial, fill=shape_color, font=ImageFont.load_default()) # İç harf
    elif shape_type == 'outline':
        d.ellipse([cx-size, cy-size, cx+size, cy+size], outline=shape_color, width=15)

    # --- Baş Harf İkonu (Merkeze) ---
    if shape_type != 'rounded':
        # Harfi büyük çizmek için font yüklemeye çalış, yoksa default kullan
        try:
            # Linux sunucularda genellikle bulunan bir font
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
        except:
            # Windows veya font yoksa, varsayılanı kullan (küçük kalır ama çalışır)
            font_large = ImageFont.load_default()
            
        # Harfi şeklin ortasına yerleştirme hesabı (Basit ortalama)
        # Pillow default font ile boyut hesabı zordur, tahmini ortalıyoruz
        d.text((cx-50, cy-80), initial, fill=bg_color if shape_type != 'outline' else shape_color, font=font_large)

    # --- Marka İsmi (Alta) ---
    try:
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except:
        font_text = ImageFont.load_default()

    # Metni alta ortalayarak yaz
    text_width = len(text) * 20 # Tahmini genişlik
    d.text((256 - (text_width/2), 400), text, fill=text_color, font=font_text)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    
    increase_usage(session.get("user_email", "guest"), "generator", "logo")
    return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

# --- LİMİT KONTROL ---
@bp.route("/check_tool_status/<tool>/<subtool>")
def status_api(tool, subtool):
    return jsonify(check_user_status(session.get("user_email","guest"), tool, subtool))
