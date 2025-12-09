import os
from flask import Blueprint, render_template, redirect

bp = Blueprint('main', __name__)

@bp.route("/")
def home():
    return render_template("index.html")
    except TemplateNotFound:
        return "Index.html bulunamadı! Lütfen 'templates' klasörünü 'app' içine taşıdığınızdan emin olun.", 404

@bp.route("/<page>")
def render_page(page):
    try:
        # Uzantı kontrolü yapalım ki statik dosyaları (css, js) şablon sanmasın
        if page.endswith(('.css', '.js', '.png', '.jpg', '.ico')):
            return abort(404)
            
        return render_template(f"{page}.html")
    except TemplateNotFound:
        # Eğer sayfa yoksa ana sayfaya yönlendir
        return redirect("/")
