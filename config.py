import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class Config:
    # Not: SECRET_KEY production'da mutlaka env ile verilmelidir.
    # Dev ortamında çalışmayı kolaylaştırmak için fallback bırakıyoruz.
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.environ.get('FLASK_SECRET_KEY') or 'dev-unsafe-secret-key'
    DEBUG = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    ENV = os.environ.get("FLASK_ENV", os.environ.get("ENV", "development"))
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 200 * 1024 * 1024))
    
    # Dosya Yükleme Ayarları (cwd yerine proje kökü baz alınır)
    PORT = int(os.environ.get("PORT", 5001))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf', 'docx'}
    
    # Model Yolları
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # YENİ: Veritabanı Ayarı
    # app.db adında bir dosya oluşacak
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(os.getcwd(), 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        # Production'da varsayılan dev anahtarla çalışmayı engelle (session güvenliği)
        if (app.config.get("ENV") == "production") and (app.config.get("SECRET_KEY") in (None, "", "dev-unsafe-secret-key", "varsayilan-guvensiz-anahtar")):
            raise RuntimeError("Production ortamında SECRET_KEY .env/env ile set edilmelidir.")

        # Yükleme klasörü yoksa oluştur
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
