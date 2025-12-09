import os
from flask import Blueprint, render_template, redirect

bp = Blueprint('main', __name__)

@bp.route("/")
def home():
    return render_template("index.html")

@bp.route("/<page>")
def render_page(page):
    # Basit bir sayfa yönlendiricisi
    # templates klasöründe <page>.html varsa onu açar, yoksa ana sayfaya atar.
    if os.path.exists(f"templates/{page}.html"):
        return render_template(f"{page}.html")
    return redirect("/")
