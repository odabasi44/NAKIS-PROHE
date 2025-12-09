from flask import Flask
from flask_cors import CORS
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Eklentileri başlat
    CORS(app)
    
    # Konfigürasyona özel işlemler
    config_class.init_app(app)

    # --- Blueprint'leri Kaydetme ---
    from app.routes.main_routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.routes.api_routes import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    from app.routes.auth_routes import bp as auth_bp
    app.register_blueprint(auth_bp)

    # --- AI Modellerini Yükleme ---
    # Uygulama bağlamı içinde modelleri yüklüyoruz
    with app.app_context():
        from app.services.ai_loader import AILoader
        print("Modeller Yükleniyor...")
        AILoader.load_models()

    return app
