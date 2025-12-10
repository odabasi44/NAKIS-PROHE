import os
from flask import Blueprint, render_template, redirect, abort
from jinja2 import TemplateNotFound

bp = Blueprint('main', __name__)

@bp.route("/")
def home():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "HATA: Index.html bulunamadı! Lütfen 'templates' klasörünün 'app' klasörü içinde olduğundan emin olun.", 404

@bp.route("/<page>")
def render_page(page):
    try:
        # Statik dosya uzantılarını (resim, css vb.) engelle
        if page.endswith(('.css', '.js', '.png', '.jpg', '.ico', '.map')):
            abort(404)
            
        return render_template(f"{page}.html")
    except TemplateNotFound:
        # Sayfa yoksa ana sayfaya yönlendir
        return redirect("/")
