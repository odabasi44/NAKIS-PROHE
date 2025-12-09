from flask import Blueprint, request, jsonify, render_template, session, redirect
from app.utils.helpers import load_settings, get_user_data_by_email, load_premium_users, save_premium_users
from datetime import datetime

bp = Blueprint('auth', __name__)

@bp.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET": 
        return render_template("admin_login.html")
    
    data = request.get_json()
    settings = load_settings()
    
    # Güvenlik: Admin ayarları yoksa varsayılanı kullanma veya hata ver
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
    settings = load_settings()
    
    if email == settings.get("admin", {}).get("email"): 
        return jsonify({"status":"admin"})
    
    user = get_user_data_by_email(email)
    if not user: 
        return jsonify({"status":"not_found"})
    
    try:
        if datetime.strptime(user.get("end_date"), "%Y-%m-%d") >= datetime.now():
            session["user_email"] = email
            session["is_premium"] = True
            session["user_tier"] = user.get("tier", "starter")
            return jsonify({"status":"premium","tier":session["user_tier"]})
        else: 
            return jsonify({"status":"expired"})
    except: 
        return jsonify({"status":"error"})

@bp.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# --- Admin API'leri ---
@bp.route("/api/admin/users", methods=["GET"])
def get_users_api():
    if not session.get("admin_logged"): return jsonify([]), 403
    return jsonify(load_premium_users())

@bp.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    data = request.get_json()
    users = load_premium_users()
    users.append({"email": data["email"], "end_date": data["end_date"], "tier": data["tier"], "usage_stats": {}})
    save_premium_users(users)
    return jsonify({"status":"ok"})

@bp.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"): return jsonify({}), 403
    users = [u for u in load_premium_users() if u["email"] != email]
    save_premium_users(users)
    return jsonify({"status":"ok"})
