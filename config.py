import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'varsayilan-guvensiz-anahtar'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 200 * 1024 * 1024))
    
    # Dosya Yükleme Ayarları
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf', 'docx'}
    
    # Model Yolları
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    @staticmethod
    def init_app(app):
        # Yükleme klasörü yoksa oluştur
        if not os.path.exists(Config.UPLOAD_FOLDER):
            os.makedirs(Config.UPLOAD_FOLDER)
