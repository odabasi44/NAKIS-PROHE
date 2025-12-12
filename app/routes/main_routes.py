import os
from flask import Blueprint, render_template, redirect, abort
from jinja2 import TemplateNotFound

bp = Blueprint("main", __name__)

@bp.route("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "HATA: index.html bulunamadı. Lütfen templates klasörünü kontrol edin.", 404


@bp.route("/<page>")
def render_page(page):

    # İzin verilen sayfalar (gerektikçe buraya ekleriz Reis)
    allowed_pages = [
        "index",
        "pdf-merge",
        "pdf-split",
        "pdf-compress",
        "word-to-pdf",
        "image-to-pdf",
        "pdf-to-image",
        "vectorizer",
        "embroidery",
        "remove-bg",
        "contact",
        "pricing",
        "login",
        "register",
        "dashboard",
    ]

    # Zararlı veya statik dosya denemelerini engelle
    if page.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".map", ".svg")):
        abort(404)

    # İzin verilmeyen bir sayfaysa ana sayfaya dön
    if page not in allowed_pages:
        return redirect("/")

    try:
        return render_template(f"{page}.html")
    except TemplateNotFound:
        # Dosya yoksa yine ana sayfaya yönlendir
        return redirect("/")
