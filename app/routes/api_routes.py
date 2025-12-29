import io
import base64
import json
import zipfile
import os
import cv2
import numpy as np
import qrcode
import requests
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from flask import Blueprint, request, jsonify, session
from app.utils.helpers import check_user_status, increase_usage
from app.services.vector_engine import AdvancedVectorEngine
from app.services.embroidery_engine import EmbroideryGenerator
from app.services.ai_loader import AILoader
# Optional: BOT -> embroidery converter (repo deploy senaryolarında dosya eksikse app boot etsin)
try:
    from app.services.bot_stitch_engine import bot_json_to_pattern, export_pattern  # type: ignore
except Exception:
    bot_json_to_pattern = None  # type: ignore
    export_pattern = None  # type: ignore
from app.models import Ticket, TicketMessage, UsageEvent
from app.extensions import db
from datetime import datetime

bp = Blueprint('api', __name__)

# --- VEKTÖR İŞLEMLERİ ---
@bp.route("/vectorize", methods=["POST"])
def api_vectorize():
    email = session.get("user_email", "guest")
    # Limit kontrolü
    st_before = check_user_status(email, "vector", "default")
    if not st_before["allowed"]:
        return jsonify({"success": False, "reason": "limit", "status": st_before}), 403
        
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Görsel yüklenmedi"}), 400

    file = request.files["image"]
    method = (request.form.get("method", "cartoon") or "cartoon").strip().lower()
    style = (request.form.get("style", "cartoon") or "cartoon").strip().lower()

    target_width = request.form.get("target_width")
    target_height = request.form.get("target_height")
    lock_aspect = request.form.get("lock_aspect")
    lock_aspect = (str(lock_aspect).strip().lower() not in ("0", "false", "no", "off")) if lock_aspect is not None else True

    # kalite parametreleri (UI'dan gelebilir)
    edge_thickness = int(request.form.get("edge_thickness", 2))
    color_count = int(request.form.get("color_count", 16))
    simplify_factor = float(request.form.get("simplify_factor", 0.003))
    min_area = int(request.form.get("min_area", 20))
    cleanup_kernel = int(request.form.get("cleanup_kernel", 3))
    border_sensitivity = int(request.form.get("border_sensitivity", 12))

    # yeni sistem: şeffaf arkaplan varsayılan açık
    remove_bg = request.form.get("remove_bg")
    remove_bg = (str(remove_bg).strip().lower() in ("1", "true", "yes", "on")) if remove_bg is not None else True
    bg_threshold = float(request.form.get("bg_threshold", 0.5))

    try:
        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"success": False, "message": "Boş dosya"}), 400

        # --- FastAPI Engine Proxy (Flask ile aynı domain'den çalışsın diye) ---
        engine_url = (os.getenv("BOTLAB_ENGINE_URL") or "").strip().rstrip("/")
        if engine_url:
            # 1) upload
            up = requests.post(
                f"{engine_url}/api/upload",
                files={"image": (getattr(file, "filename", "image"), raw_bytes, getattr(file, "mimetype", "application/octet-stream"))},
                timeout=120,
            )
            if up.status_code >= 400:
                return jsonify({"success": False, "message": f"Engine upload hata: {up.text}"}), 502
            upj = up.json()
            job_id = upj.get("id")
            if not job_id:
                return jsonify({"success": False, "message": "Engine upload id dönmedi"}), 502

            # 2) vector process (EPS + PNG)
            try:
                tw = int(request.form.get("target_width") or 0)
            except Exception:
                tw = 0
            try:
                th = int(request.form.get("target_height") or 0)
            except Exception:
                th = 0
            num_colors = int(color_count)
            # UI'da fotoğraf için 8 renk gibi değerler kullanılabiliyor (daha doğal tonlar).
            # Engine tarafında da destekliyorsak burada 2–12 aralığında tutalım.
            num_colors = max(2, min(num_colors, 12))
            mode = "logo" if method == "outline" else "photo"

            vp = requests.post(
                f"{engine_url}/api/process/vector",
                json={"id": job_id, "num_colors": num_colors, "width": (tw or None), "height": (th or None), "keep_ratio": bool(lock_aspect), "mode": mode, "outline": True, "outline_thickness": int(edge_thickness)},
                timeout=300,
            )
            if vp.status_code >= 400:
                return jsonify({"success": False, "message": f"Engine vector hata: {vp.text}"}), 502
            vpj = vp.json()
            eps_url = vpj.get("eps_url")
            png_url = vpj.get("png_url")
            colors = vpj.get("colors") or []
            if not eps_url or not png_url:
                return jsonify({"success": False, "message": "Engine vector eps/png url dönmedi"}), 502

            eps_r = requests.get(f"{engine_url}{eps_url}", timeout=120)
            png_r = requests.get(f"{engine_url}{png_url}", timeout=120)
            if eps_r.status_code >= 400 or png_r.status_code >= 400:
                return jsonify({"success": False, "message": "Engine static fetch hata"}), 502

            eps_b64 = base64.b64encode(eps_r.content).decode()
            png_b64 = base64.b64encode(png_r.content).decode()

            # 3) embroidery process -> BOT (editable)
            ep = requests.post(
                f"{engine_url}/api/process/embroidery",
                json={"id": job_id, "format": "bot", "num_colors": num_colors, "width": (tw or None), "height": (th or None), "keep_ratio": bool(lock_aspect), "mode": mode, "outline": True, "outline_thickness": int(edge_thickness)},
                timeout=300,
            )
            if ep.status_code >= 400:
                return jsonify({"success": False, "message": f"Engine embroidery hata: {ep.text}"}), 502
            epj = ep.json()
            bot_url = epj.get("bot_url") or epj.get("file_url")
            if not bot_url:
                return jsonify({"success": False, "message": "Engine bot url dönmedi"}), 502

            bot_r = requests.get(f"{engine_url}{bot_url}", timeout=120)
            if bot_r.status_code >= 400:
                return jsonify({"success": False, "message": "Engine bot fetch hata"}), 502

            bot_b64 = base64.b64encode(bot_r.content).decode()

            increase_usage(email, "vector", "default")
            st_after = check_user_status(email, "vector", "default")

            return jsonify({
                "success": True,
                "id": job_id,
                "eps": eps_b64,
                "bot": bot_b64,
                "preview_img": png_b64,
                "colors": colors,
                "status": st_after
            })

        # 2) pre-process: bg removal + enhance (EPS/embroidery için daha temiz giriş)
        original_rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # resize burada (performans + daha deterministik)
        try:
            tw = int(target_width) if target_width is not None else 0
        except Exception:
            tw = 0
        try:
            th = int(target_height) if target_height is not None else 0
        except Exception:
            th = 0
        if tw > 0 or th > 0:
            ow, oh = original_rgb.size
            if lock_aspect:
                if tw > 0 and th <= 0 and oh > 0:
                    th = int(round((tw * oh) / ow)) if ow > 0 else tw
                elif th > 0 and tw <= 0 and ow > 0:
                    tw = int(round((th * ow) / oh)) if oh > 0 else th
                if tw > 0 and th > 0 and ow > 0 and oh > 0:
                    s = min(tw / ow, th / oh)
                    tw = int(round(ow * s))
                    th = int(round(oh * s))
            tw = max(1, min(tw, 3000)) if tw > 0 else ow
            th = max(1, min(th, 3000)) if th > 0 else oh
            if (tw, th) != (ow, oh):
                original_rgb = original_rgb.resize((tw, th), Image.LANCZOS)

        np_bgr = cv2.cvtColor(np.array(original_rgb), cv2.COLOR_RGB2BGR)

        # enhance: denoise + unsharp + clahe
        den = cv2.fastNlMeansDenoisingColored(np_bgr, None, 6, 6, 7, 21)
        blur = cv2.GaussianBlur(den, (0, 0), 1.0)
        sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)
        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if (method != "outline") and (AILoader.gan_session is not None):
            enhanced = AILoader.whitebox_cartoonize(enhanced)

        # bg removal: u2net varsa alpha çıkar; yoksa border renk heuristiği
        alpha = np.full((enhanced.shape[0], enhanced.shape[1]), 255, dtype=np.uint8)
        if remove_bg:
            if AILoader.u2net_session:
                pil_enh = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                small = pil_enh.resize((320, 320))
                inp = np.transpose(np.array(small).astype(np.float32) / 255.0, (2, 0, 1))
                inp = np.expand_dims(inp, 0)
                out = AILoader.u2net_session.run(None, {AILoader.u2net_input_name: inp})[0].squeeze()
                mask = cv2.resize(out, (pil_enh.size[0], pil_enh.size[1]))
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                alpha = ((mask >= float(bg_threshold)) * 255).astype(np.uint8)
            else:
                # basit: kenar rengi = background, uzaklık threshold
                h0, w0 = enhanced.shape[:2]
                border = np.concatenate([enhanced[0, :, :], enhanced[-1, :, :], enhanced[:, 0, :], enhanced[:, -1, :]], axis=0)
                bg = np.median(border.reshape(-1, 3), axis=0).astype(np.uint8)
                diff = np.linalg.norm(enhanced.astype(np.int16) - bg.astype(np.int16), axis=2)
                alpha = (diff > 18).astype(np.uint8) * 255

        rgba = cv2.cvtColor(enhanced, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)).save(buf, "PNG")
        img_stream = io.BytesIO(buf.getvalue())

        engine = AdvancedVectorEngine(
            img_stream,
            target_width=target_width,
            target_height=target_height,
            lock_aspect=lock_aspect,
            max_dim=1000,
        )
        options = {"edge_thickness": edge_thickness, "color_count": color_count, "border_sensitivity": border_sensitivity}
        
        if method == "outline":
            # "outline" UI adı kalsa da; logo/grafik için temiz posterize taban üret
            engine.img = engine.process_logo_style(color_count=color_count)
        else:
            st = (style or "cartoon").strip().lower()
            if st == "flat":
                st = "cartoon"
            if st == "cartoon":
                engine.img = engine.process_hybrid_cartoon(edge_thickness=edge_thickness, color_count=color_count, portrait_mode=True)
            else:
                engine.img = engine.process_artistic_style(style=st, options=options)
        
        svg = engine.generate_artistic_svg(
            num_colors=color_count,
            simplify_factor=simplify_factor,
            min_area=min_area,
            cleanup_kernel=cleanup_kernel,
            ignore_background=True,
        )
        svg_b64 = base64.b64encode(svg.encode()).decode()

        # EPS + BOT (basit v1)
        eps = engine.generate_eps(
            num_colors=color_count,
            simplify_factor=simplify_factor,
            min_area=min_area,
            cleanup_kernel=cleanup_kernel,
            ignore_background=True,
        )
        eps_b64 = base64.b64encode(eps).decode()

        bot_json = engine.generate_bot_json(
            num_colors=color_count,
            simplify_factor=simplify_factor,
            min_area=min_area,
            cleanup_kernel=cleanup_kernel,
            ignore_background=True,
        )
        bot_b64 = base64.b64encode(bot_json.encode("utf-8")).decode()
        
        _, buf = cv2.imencode(".png", engine.img)
        preview_b64 = base64.b64encode(buf).decode()
        
        increase_usage(email, "vector", "default")
        st_after = check_user_status(email, "vector", "default")

        return jsonify({
            "success": True,
            "file": svg_b64,
            "eps": eps_b64,
            "bot": bot_b64,
            "preview_img": preview_b64,
            "info": {"style": style},
            "status": st_after
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


@bp.route("/export_bot_project", methods=["POST"])
def api_export_bot_project():
    """
    BOTLAB Proje dosyası (.bot):
    - Programların (Wilcom/Accurate/Wings) proprietary formatları yerine, bizim editlenebilir proje paketimiz.
    - İçerik: project.json + vector.svg + preview.png + opsiyonel source image
    """
    if "svg" not in request.form:
        return jsonify({"success": False, "message": "SVG verisi eksik"}), 400

    svg_b64 = request.form.get("svg", "")
    preview_b64 = request.form.get("preview_png", "")
    meta_json = request.form.get("meta", "{}")

    try:
        svg_bytes = base64.b64decode(svg_b64.encode())
    except Exception:
        return jsonify({"success": False, "message": "SVG decode edilemedi"}), 400

    preview_bytes = b""
    if preview_b64:
        try:
            preview_bytes = base64.b64decode(preview_b64.encode())
        except Exception:
            preview_bytes = b""

    # meta json doğrulama
    try:
        import json
        meta = json.loads(meta_json) if meta_json else {}
    except Exception:
        meta = {}

    # opsiyonel source image
    source_bytes = b""
    if "image" in request.files:
        try:
            source_bytes = request.files["image"].read() or b""
        except Exception:
            source_bytes = b""

    try:
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("vector.svg", svg_bytes)
            if preview_bytes:
                zf.writestr("preview.png", preview_bytes)
            if source_bytes:
                zf.writestr("source_image", source_bytes)
            zf.writestr("project.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

        return jsonify({
            "success": True,
            "file": base64.b64encode(out.getvalue()).decode(),
            "filename": "design.bot",
            "mime": "application/octet-stream"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@bp.route("/bot_to_embroidery", methods=["POST"])
def api_bot_to_embroidery():
    if bot_json_to_pattern is None or export_pattern is None:
        return jsonify({"success": False, "message": "BOT stitch engine sunucuda yüklü değil (deploy eksik)."}), 503
    if "bot" not in request.form:
        return jsonify({"success": False, "message": "BOT verisi yok"}), 400
    fmt = (request.form.get("format") or "dst").lower()
    bot_b64 = request.form.get("bot", "")
    try:
        bot_text = base64.b64decode(bot_b64.encode()).decode("utf-8", errors="replace")
        bot_json = json.loads(bot_text)
    except Exception:
        return jsonify({"success": False, "message": "BOT decode edilemedi"}), 400

    try:
        pattern = bot_json_to_pattern(bot_json)
        out_bytes = export_pattern(pattern, fmt)
        return jsonify({
            "success": True,
            "file": base64.b64encode(out_bytes).decode(),
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
        pdf_file = request.files["pdf_file"]
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)

        mode = (request.form.get("mode") or "extract").strip().lower()  # extract | split
        range_str = (request.form.get("range") or "").strip()

        def parse_page_range(s: str, total: int):
            """
            Accepts: "1-3", "5", "1,3,5-7" (1-based). Returns sorted unique 0-based indices.
            Empty => first page.
            """
            if not s:
                return [0] if total > 0 else []
            pages = set()
            for token in s.replace(" ", "").split(","):
                if not token:
                    continue
                if "-" in token:
                    a, b = token.split("-", 1)
                    if not a.isdigit() or not b.isdigit():
                        continue
                    start = int(a)
                    end = int(b)
                    if start <= 0 or end <= 0:
                        continue
                    if start > end:
                        start, end = end, start
                    for p in range(start, end + 1):
                        idx = p - 1
                        if 0 <= idx < total:
                            pages.add(idx)
                else:
                    if not token.isdigit():
                        continue
                    p = int(token)
                    idx = p - 1
                    if 0 <= idx < total:
                        pages.add(idx)
            if not pages and total > 0:
                pages.add(0)
            return sorted(pages)

        page_indices = parse_page_range(range_str, total_pages)
        if total_pages == 0:
            return jsonify({"success": False, "message": "PDF boş"}), 400
        if not page_indices:
            return jsonify({"success": False, "message": "Geçerli sayfa aralığı bulunamadı"}), 400

        if mode == "split":
            # Seçilen sayfaları ayrı PDF'ler olarak ZIP içinde döndür
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for idx in page_indices:
                    w = PdfWriter()
                    w.add_page(reader.pages[idx])
                    part = io.BytesIO()
                    w.write(part)
                    part_name = f"page_{idx+1}.pdf"
                    zf.writestr(part_name, part.getvalue())

            increase_usage(email, "pdf", "split")
            return jsonify({
                "success": True,
                "file": base64.b64encode(zip_buf.getvalue()).decode(),
                "mime": "application/zip",
                "filename": "pdf_parcalar.zip",
                "pages": [i + 1 for i in page_indices]
            })

        # default: extract -> tek PDF içinde seçilen sayfaları sırayla birleştir
        writer = PdfWriter()
        for idx in page_indices:
            writer.add_page(reader.pages[idx])

        out = io.BytesIO()
        writer.write(out)

        increase_usage(email, "pdf", "split")
        filename = "pdf_ayiklanan.pdf" if len(page_indices) > 1 else f"page_{page_indices[0]+1}.pdf"
        return jsonify({
            "success": True,
            "file": base64.b64encode(out.getvalue()).decode(),
            "mime": "application/pdf",
            "filename": filename,
            "pages": [i + 1 for i in page_indices]
        })
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
    email = session.get("user_email", "guest")
    if not check_user_status(email, "pdf", "word_to_pdf")["allowed"]:
        return jsonify({"success": False, "reason": "limit"}), 403

    if "word_file" not in request.files:
        return jsonify({"success": False, "message": "Word dosyası yüklenmedi"}), 400

    # Bu özellik sunucu tarafında LibreOffice/unoserver gerektirir. Şimdilik pasif.
    return jsonify({
        "success": False,
        "reason": "not_configured",
        "message": "Bu özellik için sunucu yapılandırması gerekiyor (LibreOffice)."
    }), 501

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


# ---------------- USER DASHBOARD / TICKETS ----------------

@bp.route("/user/tickets", methods=["GET"])
def user_tickets_list():
    email = session.get("user_email")
    if not email:
        return jsonify([]), 401

    tickets = Ticket.query.filter_by(user_email=email).order_by(Ticket.updated_at.desc()).limit(100).all()
    out = []
    for t in tickets:
        unread = TicketMessage.query.filter_by(ticket_id=t.id, sender="admin", is_read_by_user=False).count()
        out.append({
            "id": t.id,
            "subject": t.subject,
            "status": t.status,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            "unread": unread
        })
    return jsonify(out)


@bp.route("/user/tickets", methods=["POST"])
def user_tickets_create():
    email = session.get("user_email")
    if not email:
        return jsonify({"status": "error", "message": "login gerekli"}), 401

    data = request.get_json(silent=True) or {}
    subject = (data.get("subject") or "").strip()
    message = (data.get("message") or "").strip()
    if not subject or not message:
        return jsonify({"status": "error", "message": "subject ve message gerekli"}), 400

    try:
        t = Ticket(user_email=email, subject=subject, status="open")
        db.session.add(t)
        db.session.commit()
        db.session.add(TicketMessage(ticket_id=t.id, sender="user", message=message, is_read_by_admin=False, is_read_by_user=True))
        db.session.commit()
        return jsonify({"status": "ok", "id": t.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/user/tickets/<int:ticket_id>/messages", methods=["GET"])
def user_ticket_messages(ticket_id):
    email = session.get("user_email")
    if not email:
        return jsonify([]), 401

    t = Ticket.query.get(ticket_id)
    if not t or t.user_email != email:
        return jsonify([]), 404

    msgs = TicketMessage.query.filter_by(ticket_id=ticket_id).order_by(TicketMessage.created_at.asc()).all()
    # kullanıcı okudu -> admin mesajlarını read yap
    try:
        for m in msgs:
            if m.sender == "admin":
                m.is_read_by_user = True
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify([{
        "id": m.id,
        "sender": m.sender,
        "message": m.message,
        "created_at": m.created_at.isoformat() if m.created_at else None
    } for m in msgs])


@bp.route("/user/tickets/<int:ticket_id>/reply", methods=["POST"])
def user_ticket_reply(ticket_id):
    email = session.get("user_email")
    if not email:
        return jsonify({"status": "error", "message": "login gerekli"}), 401

    t = Ticket.query.get(ticket_id)
    if not t or t.user_email != email:
        return jsonify({"status": "error", "message": "ticket bulunamadı"}), 404

    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "message gerekli"}), 400

    try:
        db.session.add(TicketMessage(
            ticket_id=ticket_id,
            sender="user",
            message=msg,
            is_read_by_admin=False,
            is_read_by_user=True
        ))
        t.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/user/dashboard", methods=["GET"])
def user_dashboard_data():
    email = session.get("user_email")
    if not email:
        return jsonify({}), 401

    ops = UsageEvent.query.filter_by(user_email=email).order_by(UsageEvent.created_at.desc()).limit(5).all()
    open_ticket_count = Ticket.query.filter_by(user_email=email, status="open").count()
    unread_replies = TicketMessage.query.join(Ticket, Ticket.id == TicketMessage.ticket_id)\
        .filter(Ticket.user_email == email, TicketMessage.sender == "admin", TicketMessage.is_read_by_user == False).count()  # noqa: E712

    return jsonify({
        "recent_ops": [{
            "tool": o.tool,
            "subtool": o.subtool,
            "created_at": o.created_at.isoformat() if o.created_at else None
        } for o in ops],
        "open_tickets": open_ticket_count,
        "unread_replies": unread_replies
    })
