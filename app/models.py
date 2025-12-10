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
