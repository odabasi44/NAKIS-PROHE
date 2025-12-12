import os
from flask import Blueprint, render_template, redirect, abort
from jinja2 import TemplateNotFound

bp = Blueprint('main', __name__)

@bp.route("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "HATA: index.html bulunamadı. Lütfen templates klasörünü kontrol edin.", 404


@bp.route("/<page>")
def render_page(page):

    # URL → HTML eşleştirmeleri (SENİN DOSYA ADLARINA GÖRE AYARLANDI)
    page_map = {
        "pdf-merge": "pdf_merge.html",
        "remove-bg": "background_remove.html",
        "vectorizer": "vektor.html",
        "image-compress": "image_compress.html",
        "image-convert": "image_convert.html",
        "dashboard": "dashboard.html",
        "admin": "admin.html",

        # Henüz olmayan sayfalar (oluşturunca çalışacak)
        "embroidery": "embroidery.html",
        "pdf-split": "pdf_split.html",
        "word-to-pdf": "word_to_pdf.html",
        "image-to-pdf": "image_to_pdf.html",
    }

    # Statik dosyaları iptal et
    if page.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".svg")):
        abort(404)

    # Haritada olmayan sayfalar ana sayfaya dönsün
    if page not in page_map:
        return redirect("/")

    template_to_render = page_map[page]

    try:
        return render_template(template_to_render)
    except TemplateNotFound:
        # Dosya yoksa ana sayfaya dön
        return redirect("/")
