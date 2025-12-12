import os
from flask import Blueprint, render_template, redirect, abort
from jinja2 import TemplateNotFound
from app.utils.helpers import load_settings  # BU SATIR EKLENDİ

bp = Blueprint('main', __name__)

@bp.route("/")
def home():
    # Ayarları yükle
    settings = load_settings()
    # tool_status verisini al (yoksa boş sözlük ver)
    tool_status = settings.get("tool_status", {})
    
    try:
        # tools değişkenini şablona gönder
        return render_template("index.html", tools=tool_status)
    except TemplateNotFound:
        return "HATA: index.html bulunamadı. Lütfen templates klasörünü kontrol edin.", 404


@bp.route("/<page>")
def render_page(page):
    # URL -> HTML Eşleştirmeleri
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

    # Statik dosyaları iptal et
    if page.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".svg")):
        abort(404)

    # Haritada olmayan sayfalar ana sayfaya dönsün
    if page not in page_map:
        return redirect("/")

    template_to_render = page_map[page]

    try:
        # Alt sayfalara da tools verisini gönderelim ki navbar vb. yerlerde lazım olursa hata vermesin
        settings = load_settings()
        tool_status = settings.get("tool_status", {})
        return render_template(template_to_render, tools=tool_status)
    except TemplateNotFound:
        return redirect("/")
