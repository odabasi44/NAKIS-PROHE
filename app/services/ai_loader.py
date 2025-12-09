import os
import onnxruntime as ort
from flask import current_app

class AILoader:
    gan_session = None
    u2net_session = None
    u2net_input_name = "input"

    @classmethod
    def load_models(cls):
        # Base dir'i config'den veya os'den alabiliriz
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        # Whitebox Model
        wb_path = os.path.join(base_dir, "models", "whitebox_cartoon.onnx")
        if os.path.exists(wb_path):
            try:
                cls.gan_session = ort.InferenceSession(wb_path, providers=["CPUExecutionProvider"])
                print(f"🚀 Whitebox Model Yüklendi: {wb_path}")
            except Exception as e:
                print(f"⚠️ Whitebox Model Hatası: {e}")

        # U2Net Model
        u2_path = os.path.join(base_dir, "models", "u2net.onnx")
        if os.path.exists(u2_path):
            try:
                cls.u2net_session = ort.InferenceSession(u2_path, providers=["CPUExecutionProvider"])
                cls.u2net_input_name = cls.u2net_session.get_inputs()[0].name
                print(f"✅ U2Net Model Yüklendi: {u2_path}")
            except Exception as e:
                print(f"⚠️ U2Net Model Hatası: {e}")

# Uygulama başlarken modelleri yüklemek için init'te çağıracağız.
