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


# app/routes/main_routes.py

@bp.route("/<page>")
def render_page(page):
    # URL -> HTML Eşleştirmeleri
    # BURAYI DÜZELTTİK: index.html'deki linklerle uyumlu hale getirdik.
    page_map = {
        # Çalışanlar
        "remove-bg": "background_remove.html",
        
        # Vektör (Düzeltildi: "vectorizer" -> "vektor")
        "vektor": "vektor.html", 
        
        # PDF Araçları (Düzeltildi: Tek kelimeye indirgendi)
        "pdf-merge": "pdf_merge.html",
        "pdf-split": "pdf_split.html",     # Dosyası oluşturulmalı
        "pdf-compress": "pdf_compress.html", # Dosyası oluşturulmalı
        "word-to-pdf": "word_to_pdf.html", # Dosyası oluşturulmalı
        
        # Görsel Araçlar
        "image-compress": "image_compress.html",
        "image-convert": "image_convert.html",
        
        # Diğer
        "qr-generator": "qr_generator.html", # Dosyası oluşturulmalı
        "logo-generator": "logo_generator.html", # Dosyası oluşturulmalı
        "dashboard": "dashboard.html",
        "admin": "admin.html",
    }
    
    # ... kodun geri kalanı aynı ...

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
