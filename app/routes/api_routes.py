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
        orig_size = request.content_length or new_size * 2 # Tahmini
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
    # PDF Bölme Mantığı (Basitleştirilmiş: Her sayfayı ayırır veya aralık alır)
    email = session.get("user_email", "guest")
    # Limit kontrolü eklenebilir
    
    if "pdf_file" not in request.files: return jsonify({"success": False}), 400
    try:
        reader = PdfReader(request.files["pdf_file"])
        # Örnek: İlk sayfayı döndür (Geliştirilebilir: ZIP olarak tüm sayfalar)
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        
        out = io.BytesIO()
        writer.write(out)
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/pdf/compress", methods=["POST"])
def api_pdf_compress():
    # PDF Sıkıştırma (PyPDF2 ile Metadata silerek basit sıkıştırma)
    if "pdf_file" not in request.files: return jsonify({"success": False}), 400
    try:
        reader = PdfReader(request.files["pdf_file"])
        writer = PdfWriter()
        for page in reader.pages:
            page.compress_content_streams() # İçeriği sıkıştır
            writer.add_page(page)
        
        writer.add_metadata({}) # Metadatayı silerek yer kazan
        out = io.BytesIO()
        writer.write(out)
        return jsonify({"success": True, "file": base64.b64encode(out.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/word_to_pdf", methods=["POST"])
def api_word_to_pdf():
    # Word to PDF (Python'da programsız zordur, bu bir simülasyondur)
    # Gerçek dönüşüm için LibreOffice kurulu olmalı.
    return jsonify({"success": False, "message": "Sunucuda LibreOffice kurulu değil. Bu özellik şu an demo modundadır."}), 501

# --- QR & LOGO ---

@bp.route("/qr_generator", methods=["POST"])
def api_qr_generator():
    data = request.json or request.form
    text = data.get("text", "https://botlab.tools")
    color = data.get("color", "black")
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color=color, back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

@bp.route("/logo_generator", methods=["POST"])
def api_logo_generator():
    # Basit "AI" Logo Simülasyonu (Metin + Şekil Çizimi)
    data = request.json or request.form
    text = data.get("text", "Marka")
    
    # 500x500 Boş Resim
    img = Image.new('RGB', (500, 500), color=(20, 20, 30))
    d = ImageDraw.Draw(img)
    
    # Basit bir geometrik şekil (AI Logosu gibi dursun)
    d.ellipse([100, 50, 400, 350], outline="cyan", width=10)
    d.rectangle([200, 200, 300, 300], fill="magenta")
    
    # Metin (Font dosyasını bulamazsa default kullanır)
    try:
        # Sunucuda font varsa: font = ImageFont.truetype("arial.ttf", 40)
        font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        
    d.text((150, 400), text, fill="white", font=font)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})

# --- LİMİT KONTROL ---
@bp.route("/check_tool_status/<tool>/<subtool>")
def status_api(tool, subtool):
    return jsonify(check_user_status(session.get("user_email","guest"), tool, subtool))
