import os
import onnxruntime as ort
import cv2
import numpy as np
from flask import current_app

class AILoader:
    gan_session = None
    u2net_session = None
    u2net_input_name = "input"

    @classmethod
    def model_status(cls):
        def _sess_info(sess):
            if sess is None:
                return None
            try:
                inp0 = sess.get_inputs()[0] if sess.get_inputs() else None
                return {
                    "providers": list(getattr(sess, "get_providers", lambda: [])()),
                    "input_name": getattr(inp0, "name", None),
                    "input_shape": getattr(inp0, "shape", None),
                }
            except Exception:
                return {"providers": None, "input_name": None, "input_shape": None}

        return {
            "onnxruntime_version": getattr(ort, "__version__", None),
            "whitebox_cartoon": {"loaded": cls.gan_session is not None, "session": _sess_info(cls.gan_session)},
            "u2net": {"loaded": cls.u2net_session is not None, "input_name": cls.u2net_input_name, "session": _sess_info(cls.u2net_session)},
        }

    @staticmethod
    def whitebox_cartoonize(bgr_img: "np.ndarray") -> "np.ndarray":
        if AILoader.gan_session is None:
            return bgr_img
        if bgr_img is None or not hasattr(bgr_img, "shape"):
            return bgr_img
        try:
            h0, w0 = bgr_img.shape[:2]
            inp = AILoader.gan_session.get_inputs()[0]
            shape = getattr(inp, "shape", None) or []
            ih = 256
            iw = 256
            if len(shape) >= 4:
                if isinstance(shape[2], int) and shape[2] > 0:
                    ih = int(shape[2])
                if isinstance(shape[3], int) and shape[3] > 0:
                    iw = int(shape[3])
            ih = max(64, min(int(ih), 1024))
            iw = max(64, min(int(iw), 1024))

            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_AREA)
            x = rgb_r.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 0)

            out = AILoader.gan_session.run(None, {inp.name: x})
            if not out:
                return bgr_img
            y = out[0]
            y = np.array(y)

            if y.ndim == 4:
                if y.shape[1] == 3:
                    y = np.transpose(y[0], (1, 2, 0))
                elif y.shape[-1] == 3:
                    y = y[0]
                else:
                    return bgr_img
            elif y.ndim == 3:
                if y.shape[0] == 3:
                    y = np.transpose(y, (1, 2, 0))
            else:
                return bgr_img

            y = y.astype(np.float32)
            if y.min() < 0.0:
                y = (y + 1.0) * 0.5
            y = np.clip(y, 0.0, 1.0)
            y = (y * 255.0).astype(np.uint8)

            y = cv2.resize(y, (w0, h0), interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        except Exception:
            return bgr_img

    @classmethod
    def load_models(cls):
        # Base dir'i config'den veya os'den alabiliriz
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        try:
            logger = current_app.logger
        except Exception:
            logger = None

        def _log(level: str, msg: str):
            if logger is not None:
                try:
                    getattr(logger, level)(msg)
                    return
                except Exception:
                    pass
            print(msg)
        
        # Whitebox Model
        wb_path = os.path.join(base_dir, "models", "whitebox_cartoon.onnx")
        if os.path.exists(wb_path):
            try:
                cls.gan_session = ort.InferenceSession(wb_path, providers=["CPUExecutionProvider"])
                _log("info", f"Whitebox Model Yüklendi: {wb_path} | providers={cls.gan_session.get_providers()} | ort={getattr(ort, '__version__', None)}")
            except Exception as e:
                _log("warning", f"Whitebox Model Hatası: {e}")
        else:
            _log("warning", f"Whitebox Model Bulunamadı: {wb_path}")

        # U2Net Model
        u2_path = os.path.join(base_dir, "models", "u2net.onnx")
        if os.path.exists(u2_path):
            try:
                cls.u2net_session = ort.InferenceSession(u2_path, providers=["CPUExecutionProvider"])
                cls.u2net_input_name = cls.u2net_session.get_inputs()[0].name
                _log("info", f"U2Net Model Yüklendi: {u2_path} | providers={cls.u2net_session.get_providers()} | input={cls.u2net_input_name} | ort={getattr(ort, '__version__', None)}")
            except Exception as e:
                _log("warning", f"U2Net Model Hatası: {e}")
        else:
            _log("warning", f"U2Net Model Bulunamadı: {u2_path}")

# Uygulama başlarken modelleri yüklemek için init'te çağıracağız.
