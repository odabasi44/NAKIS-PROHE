import os
from flask import Blueprint, render_template, redirect, abort, jsonify, session
from jinja2 import TemplateNotFound
from app.utils.helpers import load_settings, get_user_data_by_email
from app.models import UsageEvent
from datetime import datetime

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
    tool_access = settings.get("tool_access", {})  # min tier vb.

    # Kullanıcının tier bilgisi (guest/free/starter/pro/unlimited)
    user_tier = session.get("user_tier", "free" if session.get("user_email") else "free")
    tier_rank = {"free": 0, "starter": 1, "pro": 2, "unlimited": 3}

    def _compute_min_tier_from_tiers(tiers_dict):
        for t in ("free", "starter", "pro", "unlimited"):
            if tiers_dict.get(t) is True:
                return t
        return "unlimited"

    def _suggest_upgrade_tier(access_conf, current_tier):
        """Tool erişimi yoksa, checkbox'lara göre en düşük uygun yükseltmeyi öner."""
        ranks = ("free", "starter", "pro", "unlimited")
        cur_i = ranks.index(current_tier) if current_tier in ranks else 0
        if isinstance(access_conf, dict) and isinstance(access_conf.get("tiers"), dict):
            tiers_dict = access_conf["tiers"]
            for t in ranks[cur_i + 1:]:
                if tiers_dict.get(t) is True:
                    return t
            # hiçbiri uygun değilse, en düşük açık tier
            return _compute_min_tier_from_tiers(tiers_dict)
        # min_tier modelinde min_tier zaten hedef
        return (access_conf or {}).get("min_tier", "starter") if isinstance(access_conf, dict) else "starter"

    def _allowed_for_tier(access_conf, current_tier):
        if not isinstance(access_conf, dict):
            return True, "free"
        if isinstance(access_conf.get("tiers"), dict):
            tiers_dict = access_conf["tiers"]
            return bool(tiers_dict.get(current_tier, False)), _compute_min_tier_from_tiers(tiers_dict)
        min_tier = access_conf.get("min_tier", "free")
        return tier_rank.get(current_tier, 0) >= tier_rank.get(min_tier, 0), min_tier

    # Template'te tek yerden yönetmek için tool objelerini normalize et
    tools_view = {}
    for key, st in tool_status.items():
        active = bool(st.get("active", True)) if isinstance(st, dict) else bool(st)
        maintenance = bool(st.get("maintenance", False)) if isinstance(st, dict) else False
        access_conf = tool_access.get(key, {})
        allowed, min_tier = _allowed_for_tier(access_conf, user_tier)
        suggest_tier = _suggest_upgrade_tier(access_conf, user_tier) if not allowed else None
        tools_view[key] = {
            "active": active and (not maintenance),
            "maintenance": maintenance,
            "min_tier": min_tier,
            "allowed": allowed,
            "suggest_tier": suggest_tier
        }
    try:
        return render_template("index.html", tools=tools_view)
    except TemplateNotFound:
        return "HATA: index.html bulunamadı.", 404

@bp.route("/dashboard")
def dashboard():
    """
    Premium kullanıcı paneli.
    Önceden /<page> üzerinden dashboard.html render edildiği için `user` context'i gelmiyor ve 500 oluşuyordu.
    """
    email = session.get("user_email")
    if not email:
        return redirect("/")

    user_data = get_user_data_by_email(email)
    if not user_data:
        session.clear()
        return redirect("/")

    # Üyelik kontrolü (admin unlimited ise end_date olmayabilir; burada basitçe izin veriyoruz)
    try:
        if user_data.get("end_date"):
            end_date = datetime.strptime(user_data["end_date"], "%Y-%m-%d").date()
            if end_date < datetime.now().date():
                return redirect("/")
    except Exception:
        pass

    # Kullanım toplamları (dashboard kartları için)
    # Not: user.usage_stats_json subtool bazlı tutulduğu için "compress" gibi anahtarlar PDF/IMAGE çakışabiliyor.
    # Bu yüzden doğru sayım için UsageEvent tablosunu baz alıyoruz.
    try:
        usage_total = UsageEvent.query.filter_by(user_email=email).count()
        image_total = UsageEvent.query.filter_by(user_email=email, tool="image").count()
    except Exception:
        usage_total = 0
        image_total = 0
    user_data["usage_total"] = usage_total
    user_data["image_total"] = image_total

    settings = load_settings()
    tool_status = settings.get("tool_status", {})
    return render_template("dashboard.html", user=user_data, tools=tool_status)

@bp.route("/admin")
def admin():
    """Admin paneli: admin login zorunlu."""
    if not session.get("admin_logged"):
        return redirect("/admin_login")
    settings = load_settings()
    tool_status = settings.get("tool_status", {})
    return render_template("admin.html", tools=tool_status)

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
        # dashboard/admin ayrı route ile render ediliyor
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
