import json
from flask import Blueprint, request, jsonify, render_template, session, redirect
from app.utils.helpers import load_settings, save_settings, get_user_data_by_email
from app.models import User
from app.extensions import db
from datetime import datetime
from sqlalchemy import func
from app.models import Ticket, TicketMessage, UsageEvent

bp = Blueprint('auth', __name__)

@bp.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET": 
        return render_template("admin_login.html")
    
    data = request.get_json(silent=True) or {}
    settings = load_settings()
    
    # Admin ayarlarını güvenli çek
    admin_conf = settings.get("admin", {})
    
    req_email = (data.get("email") or "").strip().lower()
    admin_email = (admin_conf.get("email") or "").strip().lower()
    if req_email == admin_email and data.get("password") == admin_conf.get("password"):
        session["admin_logged"] = True
        session["user_email"] = req_email
        session["user_tier"] = "unlimited"
        session["is_premium"] = True
        return jsonify({"status":"ok"})
    
    return jsonify({"status":"error"}), 401

@bp.route("/user_login", methods=["POST"])
def user_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"status": "error", "message": "email gerekli"}), 400
    
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
        if not end_date_str:
            return jsonify({"status": "expired"})

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
        return jsonify({"status":"error"}), 500

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
        usage = u.get_usage()
        usage_total = 0
        try:
            usage_total = sum(int(v) for v in usage.values() if isinstance(v, (int, float, str)) and str(v).isdigit())
        except Exception:
            usage_total = 0
        users_list.append({
            "email": u.email,
            "tier": u.tier,
            "end_date": u.end_date.strftime("%Y-%m-%d") if u.end_date else None,
            "usage_stats": usage,
            "usage_total": usage_total
        })
    return jsonify(users_list)

@bp.route("/api/admin/add_user", methods=["POST"])
def add_user_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"status": "error", "message": "email gerekli"}), 400
    
    # Mevcut kullanıcı mı?
    existing = User.query.filter(func.lower(User.email) == email).first()
    if existing:
        return jsonify({"status": "exists", "message": "Kullanıcı zaten var"}), 400

    try:
        end_date_raw = data.get("end_date")
        if not end_date_raw:
            return jsonify({"status": "error", "message": "end_date gerekli (YYYY-MM-DD)"}), 400

        new_user = User(
            email=email,
            tier=data.get("tier", "free"),
            end_date=datetime.strptime(end_date_raw, "%Y-%m-%d").date()
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/admin/delete_user/<email>", methods=["DELETE"])
def del_user_api(email):
    if not session.get("admin_logged"): return jsonify({}), 403
    
    email_norm = (email or "").strip().lower()
    user = User.query.filter(func.lower(User.email) == email_norm).first()
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"status":"ok"})
    
    return jsonify({"status":"error", "message": "Kullanıcı bulunamadı"}), 404

@bp.route("/api/admin/save_general", methods=["POST"])
def save_general_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    
    data = request.get_json(silent=True) or {}
    new_general = data.get("general", {})
    
    settings = load_settings()
    settings["general"] = new_general
    
    try:
        save_settings(settings)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.route("/api/admin/save_tool_status", methods=["POST"])
def save_tool_status_api():
    if not session.get("admin_logged"): return jsonify({}), 403
    
    data = request.get_json(silent=True) or {}
    new_statuses = data.get("tool_status", {})
    new_access = data.get("tool_access", None)
    
    settings = load_settings()
    
    # Mevcut ayarları koruyarak güncelle
    if "tool_status" not in settings: settings["tool_status"] = {}
    if "tool_access" not in settings: settings["tool_access"] = {}
    
    # Gelen veriyi settings'e işle
    for key, val in new_statuses.items():
        # Eski maintenance ayarı varsa koru, yoksa false yap
        old_maint = settings["tool_status"].get(key, {}).get("maintenance", False)
        settings["tool_status"][key] = {
            "active": val.get("active", True),
            "maintenance": old_maint # Bakım modu ayarı ayrı, onu bozmuyoruz
        }

    # Tool tier erişimleri (opsiyonel)
    if isinstance(new_access, dict):
        for key, val in new_access.items():
            if not isinstance(val, dict):
                continue
            min_tier = val.get("min_tier", "free")
            settings["tool_access"][key] = {"min_tier": min_tier}
    
    try:
        save_settings(settings)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/admin/save_limits", methods=["POST"])
def save_limits_api():
    if not session.get("admin_logged"):
        return jsonify({}), 403
    data = request.get_json(silent=True) or {}
    new_limits = data.get("limits", {})
    settings = load_settings()
    settings["limits"] = new_limits
    try:
        save_settings(settings)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/admin/save_packages", methods=["POST"])
def save_packages_api():
    if not session.get("admin_logged"):
        return jsonify({}), 403
    data = request.get_json(silent=True) or {}
    new_packages = data.get("packages", {})
    settings = load_settings()
    settings["packages"] = new_packages
    try:
        save_settings(settings)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@bp.route("/api/admin/stats", methods=["GET"])
def admin_stats_api():
    if not session.get("admin_logged"):
        return jsonify({}), 403

    today = datetime.now().date()
    total_users = User.query.count()
    active_premium = User.query.filter(User.end_date != None).filter(User.end_date >= today).count()  # noqa: E711
    total_ops = UsageEvent.query.count()
    open_tickets = Ticket.query.filter_by(status="open").count()

    return jsonify({
        "total_users": total_users,
        "active_premium": active_premium,
        "total_ops": total_ops,
        "open_tickets": open_tickets
    })


@bp.route("/api/admin/tickets", methods=["GET"])
def admin_tickets_list_api():
    if not session.get("admin_logged"):
        return jsonify([]), 403

    tickets = Ticket.query.order_by(Ticket.updated_at.desc()).limit(200).all()
    out = []
    for t in tickets:
        last_msg = TicketMessage.query.filter_by(ticket_id=t.id).order_by(TicketMessage.created_at.desc()).first()
        out.append({
            "id": t.id,
            "user_email": t.user_email,
            "subject": t.subject,
            "status": t.status,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            "last_message": last_msg.message[:200] if last_msg else ""
        })
    return jsonify(out)


@bp.route("/api/admin/tickets/<int:ticket_id>/messages", methods=["GET"])
def admin_ticket_messages_api(ticket_id):
    if not session.get("admin_logged"):
        return jsonify([]), 403
    msgs = TicketMessage.query.filter_by(ticket_id=ticket_id).order_by(TicketMessage.created_at.asc()).all()
    # admin okuduğu için işaretle
    try:
        for m in msgs:
            if m.sender == "user":
                m.is_read_by_admin = True
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify([{
        "id": m.id,
        "sender": m.sender,
        "message": m.message,
        "created_at": m.created_at.isoformat() if m.created_at else None
    } for m in msgs])


@bp.route("/api/admin/tickets/<int:ticket_id>/reply", methods=["POST"])
def admin_ticket_reply_api(ticket_id):
    if not session.get("admin_logged"):
        return jsonify({}), 403
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"status": "error", "message": "message gerekli"}), 400

    t = Ticket.query.get(ticket_id)
    if not t:
        return jsonify({"status": "error", "message": "ticket yok"}), 404

    try:
        db.session.add(TicketMessage(
            ticket_id=ticket_id,
            sender="admin",
            message=msg,
            is_read_by_user=False,
            is_read_by_admin=True
        ))
        t.updated_at = datetime.utcnow()
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500
