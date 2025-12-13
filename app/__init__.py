from flask import Flask
from flask_cors import CORS
from config import Config
from app.extensions import db
import os
from sqlalchemy import text, inspect

def create_app(config_class=Config):

    # TEMPLATE YOLUNU AÇIKÇA BELİRTİYORUZ (KRİTİK)
    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

    app = Flask(__name__, template_folder=template_dir)

    app.config.from_object(config_class)
    CORS(app)

    db.init_app(app)
    config_class.init_app(app)

    # MAIN ROUTES
    from app.routes.main_routes import bp as main_bp
    app.register_blueprint(main_bp)

    # API ROUTES
    from app.routes.api_routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # AUTH ROUTES
    from app.routes.auth_routes import bp as auth_bp
    app.register_blueprint(auth_bp)

    with app.app_context():
        from app.models import User
        db.create_all()

        # ---- MINI MIGRATION (Postgres) ----
        # Not: db.create_all mevcut tablolara kolon eklemez. Eski şemadan gelen DB'lerde
        # `user.usage_stats_json` kolonu eksik kalabiliyor ve admin API'leri 500 veriyor.
        try:
            insp = inspect(db.engine)
            if "user" in insp.get_table_names():
                cols = {c["name"] for c in insp.get_columns("user")}
                # Eski isimden taşı (varsa)
                if ("usage_stats" in cols) and ("usage_stats_json" not in cols):
                    db.session.execute(text('ALTER TABLE "user" RENAME COLUMN usage_stats TO usage_stats_json'))
                    db.session.commit()
                    cols = {c["name"] for c in insp.get_columns("user")}

                if "usage_stats_json" not in cols:
                    db.session.execute(text('ALTER TABLE "user" ADD COLUMN usage_stats_json TEXT DEFAULT \'{}\''))
                    db.session.execute(text('UPDATE "user" SET usage_stats_json = \'{}\' WHERE usage_stats_json IS NULL'))
                    db.session.commit()
        except Exception:
            db.session.rollback()

        from app.services.ai_loader import AILoader
        AILoader.load_models()

    return app
