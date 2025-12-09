import io
import base64
import cv2
import numpy as np
from PIL import Image
from PyPDF2 import PdfMerger
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
    # Limit kontrolü (İsteğe bağlı açılabilir)
    # status = check_user_status(email, "vector", "default")
    # if not status["allowed"]: return jsonify({"success": False, "reason": "limit"}), 403
    
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Görsel yüklenmedi"}), 400

    file = request.files["image"]
    method = request.form.get("method", "cartoon")
    style = request.form.get("style", "cartoon")
    edge_thickness = int(request.form.get("edge_thickness", 2))
    color_count = int(request.form.get("color_count", 16))

    try:
        engine = AdvancedVectorEngine(file)
        
        options = {
            "edge_thickness": edge_thickness,
            "color_count": color_count
        }
        
        if method == "outline":
            engine.img = engine.process_sketch_style()
        else:
            # ERC V4 Motoru
            engine.img = engine.process_artistic_style(style=style, options=options)
        
        # SVG Üret
        svg = engine.generate_artistic_svg(num_colors=color_count, simplify_factor=0.003)
        svg_b64 = base64.b64encode(svg.encode()).decode()
        
        # Önizleme (PNG)
        _, buf = cv2.imencode(".png", engine.img)
        preview_b64 = base64.b64encode(buf).decode()

        # increase_usage(email, "vector", "default")

        return jsonify({
            "success": True,
            "file": svg_b64,
            "preview_img": preview_b64,
            "info": {"style": style}
        })

    except Exception as e:
        print("VECTOR ERROR:", e)
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
        print(f"NAKIŞ HATASI: {e}")
        return jsonify({"success": False, "message": f"Nakış hatası: {str(e)}"}), 500

# --- ARKA PLAN SİLME ---
@bp.route("/remove_bg", methods=["POST"])
def api_remove_bg():
    if not AILoader.u2net_session:
        return jsonify({"success": False, "reason": "AI Model Yüklenemedi (Dosya Eksik)"}), 503

    if "image" not in request.files:
        return jsonify({"success": False}), 400

    email = session.get("user_email", "guest")
    status = check_user_status(email, "image", "remove_bg")
    if not status["allowed"]:
        return jsonify({"success": False, "reason": "limit"}), 403

    try:
        original = Image.open(request.files["image"]).convert("RGB")
        # Performans için küçültme
        small = original.resize((320, 320))

        inp = np.transpose(np.array(small).astype(np.float32) / 255.0, (2, 0, 1))
        inp = np.expand_dims(inp, 0)

        # AI Tahmini
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
        increase_usage(email, "image", "compress")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/img/convert", methods=["POST"])
def api_img_convert():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "image", "convert")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403
    
    if "image" not in request.files: return jsonify({"success": False}), 400
    try:
        fmt = request.form.get("format", "jpeg").lower()
        fmt_map = {"jpg":"JPEG","jpeg":"JPEG","png":"PNG","webp":"WEBP","pdf":"PDF"}
        img = Image.open(request.files["image"]).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, fmt_map.get(fmt, "JPEG"))
        increase_usage(email, "image", "convert")
        return jsonify({"success": True, "file": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e: return jsonify({"success": False, "message": str(e)}), 500

@bp.route("/pdf/merge", methods=["POST"])
def api_pdf_merge():
    email = session.get("user_email", "guest")
    if not check_user_status(email, "pdf", "merge")["allowed"]: 
        return jsonify({"success": False, "reason": "limit"}), 403
    
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

# --- LİMİT KONTROL ---
@bp.route("/check_tool_status/<tool>/<subtool>")
def status_api(tool, subtool):
    return jsonify(check_user_status(session.get("user_email","guest"), tool, subtool))
