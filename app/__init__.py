from flask import Flask
from flask_cors import CORS
from config import Config
from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)
    
    # Eklentileri başlat
    db.init_app(app)
    config_class.init_app(app)

    # --- Blueprint'leri Kaydetme (BURASI ÇOK ÖNEMLİ) ---
    from app.routes.main_routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.routes.api_routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    from app.routes.auth_routes import bp as auth_bp
    app.register_blueprint(auth_bp)

    with app.app_context():
        from app.models import User
        db.create_all()
        from app.services.ai_loader import AILoader
        AILoader.load_models()

    return app
