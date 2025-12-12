import json
from flask import Blueprint, request, jsonify, render_template, session, redirect
from app.utils.helpers import load_settings, get_user_data_by_email
from app.models import User
from app.extensions import db
from datetime import datetime

bp = Blueprint('auth', __name__)

@bp.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET": 
        return render_template("admin_login.html")
    
    data = request.get_json()
    settings = load_settings()
    
    # Admin ayarlarını güvenli çek
    admin_conf = settings.get("admin", {})
    
    if data.get("email") == admin_conf.get("email") and data.get("password") == admin_conf.get("password"):
        session["admin_logged"] = True
        session["user_email"] = data["email"]
        session["user_tier"] = "unlimited"
        session["is_premium"] = True
        return jsonify({"status":"ok"})
    
    return jsonify({"status":"error"}), 401

@bp.route("/user_login", methods=["POST"])
def user_login():
    data = request.get_json()
    email = data.get("email")
    
    # NOT: Buradaki admin kontrolünü sildik. 
    # Artık admin e-postası yazılsa bile veritabanında "normal üye" olarak yoksa giriş yapamaz.
    # Adminler sadece /admin_login sayfasından girebilir.

    # Veritabanından kullanıcı kontrolü
    user_data = get_user_data_by_email(email)
    
    if not user_data: 
        return jsonify({"status":"not_found"})
    
    try:
        # Tarih verisi string veya date objesi olabilir, kontrol et
        end_date_str = user_data.get("end_date")
        if isinstance(end_date_str, str):
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        else:
            end_date = end_date_str

        if end_date >= datetime.now().date():
            session["user_email"] = email
            session["is_premium"] = True
            session["user_tier"] = user_data.get("tier", "starter")
            return jsonify({"status":"premium","tier":session["user_tier"]})
        else: 
            return jsonify({"status":"expired"})
    except Exception as e:
        print(f"Login Hatası: {e}")
        return jsonify({"status":"error"})

@bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# --- Admin API'leri (ARTIK VERİTABANI KULLANIYOR) ---

@bp.route("/api/admin/users", methods=["GET"])
def get_users_api():
    if not session.get("admin_logged"): return jsonify([]), 403
    
    # Tüm kullanıcıları veritabanından çek
    users = User.query.all()
    users_list = []
    for u in users:
        users_list.append({
            "email": u.email,
            "tier": u.tier,
            "end_date": u.end_date.strftime("%Y-%m-%d") if u.end_date else None,
            "usage_stats": u.get_usage()
        })
    return jsonify(users_list)

@bp.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    
    data = request.get_json()
    email = data.get("email")
    
    # Mevcut kullanıcı mı?
    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"status": "exists", "message": "Kullanıcı zaten var"}), 400

    try:
        new_user = User(
            email=email,
            tier=data.get("tier", "free"),
            end_date=datetime.strptime(data.get("end_date"), "%Y-%m-%d").date()
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"): return jsonify({}), 403
    
    user = User.query.filter_by(email=email).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"status":"ok"})
    
    return jsonify({"status":"error", "message": "Kullanıcı bulunamadı"}), 404

@bp.route("/api/admin/save_general", methods=["POST"])
def save_general_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    
    data = request.get_json()
    new_general = data.get("general", {})
    
    settings = load_settings()
    settings["general"] = new_general
    
    try:
        with open("settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
