import json
from app.extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    tier = db.Column(db.String(50), default="free")
    end_date = db.Column(db.Date, nullable=True)
    
    # Kullanım istatistiklerini JSON formatında tutuyoruz
    # Örn: {"remove_bg": 5, "vector": 2}
    usage_stats = db.Column(db.Text, default="{}")

    def get_usage(self):
        try:
            return json.loads(self.usage_stats or "{}")
        except:
            return {}

    def increase_usage(self, tool_key):
        """Bir aracın kullanım sayısını artırır."""
        stats = self.get_usage()
        # tool_key örn: 'remove_bg' veya 'default' (vektör için 'default' geliyor helpers'dan)
        current_val = stats.get(tool_key, 0)
        stats[tool_key] = current_val + 1
        self.usage_stats = json.dumps(stats)
