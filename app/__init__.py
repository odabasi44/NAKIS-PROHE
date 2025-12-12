from flask import Flask
from flask_cors import CORS
from config import Config
from app.extensions import db
import os

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

        from app.services.ai_loader import AILoader
        AILoader.load_models()

    return app
