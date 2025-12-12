import os
from flask import Blueprint, render_template, redirect, abort, jsonify # jsonify EKLENDİ
from jinja2 import TemplateNotFound
from app.utils.helpers import load_settings

bp = Blueprint('main', __name__)

# --- YENİ EKLENEN ROTA ---
@bp.route("/get_settings")
def get_settings_route():
    # Admin paneli bu adresten ayarları okur
    return jsonify(load_settings())
# -------------------------

@bp.route("/")
def home():
    settings = load_settings()
    tool_status = settings.get("tool_status", {})
    try:
        return render_template("index.html", tools=tool_status)
    except TemplateNotFound:
        return "HATA: index.html bulunamadı.", 404

@bp.route("/<page>")
def render_page(page):
    page_map = {
        "remove-bg": "background_remove.html",
        "vektor": "vektor.html", 
        "pdf-merge": "pdf_merge.html",
        "pdf-split": "pdf_split.html",
        "pdf-compress": "pdf_compress.html",
        "word-to-pdf": "word_to_pdf.html",
        "image-compress": "image_compress.html",
        "image-convert": "image_convert.html",
        "qr-generator": "qr_generator.html",
        "logo-generator": "logo_generator.html",
        "dashboard": "dashboard.html",
        "admin": "admin.html",
    }

    if page.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".svg")):
        abort(404)

    if page not in page_map:
        return redirect("/")

    template_to_render = page_map[page]

    try:
        settings = load_settings()
        tool_status = settings.get("tool_status", {})
        return render_template(template_to_render, tools=tool_status)
    except TemplateNotFound:
        return redirect("/")
