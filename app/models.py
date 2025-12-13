from app.extensions import db
from datetime import datetime
import json

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    tier = db.Column(db.String(20), default='free')  # free, starter, pro, unlimited
    end_date = db.Column(db.Date, nullable=True)     # Üyelik bitiş tarihi
    usage_stats_json = db.Column(db.Text, default="{}") # Kullanım istatistikleri (JSON olarak metin saklayacağız)

    def get_usage(self):
        """Kullanım verisini sözlük (dict) olarak döndürür."""
        try:
            return json.loads(self.usage_stats_json)
        except:
            return {}

    def set_usage(self, usage_dict):
        """Kullanım verisini kaydeder."""
        self.usage_stats_json = json.dumps(usage_dict)

    def increase_usage(self, subtool):
        """Belirli bir aracın kullanımını 1 artırır."""
        usage = self.get_usage()
        current = usage.get(subtool, 0)
        usage[subtool] = current + 1
        self.set_usage(usage)


class AppSetting(db.Model):
    """Deploy sonrası sıfırlanmayan ayarlar için DB-backed settings."""
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    value_json = db.Column(db.Text, default="{}")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_value(self):
        try:
            return json.loads(self.value_json or "{}")
        except Exception:
            return {}

    def set_value(self, value_dict):
        self.value_json = json.dumps(value_dict or {}, ensure_ascii=False)


class UsageEvent(db.Model):
    """Admin raporları ve dashboard 'son işlemler' için işlem logu."""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=True, index=True)
    tool = db.Column(db.String(64), nullable=False, index=True)
    subtool = db.Column(db.String(64), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


class Ticket(db.Model):
    """Destek talebi."""
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False, index=True)
    subject = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default="open", index=True)  # open/closed
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)


class TicketMessage(db.Model):
    """Ticket içindeki mesajlar (user/admin)."""
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey("ticket.id"), nullable=False, index=True)
    sender = db.Column(db.String(10), nullable=False)  # user/admin
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    is_read_by_user = db.Column(db.Boolean, default=True)
    is_read_by_admin = db.Column(db.Boolean, default=False)

    ticket = db.relationship("Ticket", backref=db.backref("messages", lazy=True, cascade="all, delete-orphan"))
