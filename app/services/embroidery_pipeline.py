import io
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from app.services.vector_engine import AdvancedVectorEngine
from app.services.bot_stitch_engine import bot_json_to_pattern, export_pattern


@dataclass
class AutoDigitizeDecision:
    """Stores auto-digitize decisions for ML training"""
    area_id: str
    fill_type: str  # 'satin' | 'tatami'
    reason: str
    confidence: float
    width_mm: float
    area_size_px: int


@dataclass
class BOTFormat:
    """Single source of truth for embroidery data"""
    # Versioning
    engine_version: str = "4.0"
    ruleset_version: str = "1.0"
    generated_at: str = ""
    
    # Vector geometry
    metadata: Dict[str, Any] = None
    objects: List[Dict[str, Any]] = None
    
    # Area measurements
    width_mm: float = 0.0
    height_mm: float = 0.0
    aspect_ratio: float = 1.0
    
    # Auto-digitize decisions
    decisions: List[Dict[str, Any]] = None
    
    # Stitch parameters
    stitch_density: float = 4.0
    stitch_angle: float = 45.0
    underlay_type: str = "default"
    underlay_density: float = 2.0
    
    # Processing info
    processing_time_ms: int = 0
    source_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generated_at == "":
            self.generated_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        if self.objects is None:
            self.objects = []
        if self.decisions is None:
            self.decisions = []
        if self.source_params is None:
            self.source_params = {}


class EmbroideryPipeline:
    """
    Enforces mandatory flow: image → vector → BOT → embroidery_format
    BOT format is the single source of truth
    """
    
    def __init__(self, image_stream, params: Optional[Dict[str, Any]] = None):
        self.image_stream = image_stream
        self.params = params or {}
        self.vector_engine = None
        self.bot_format = None
        
    def process_vector(self) -> Dict[str, Any]:
        """Step 1: Convert image to vector"""
        self.vector_engine = AdvancedVectorEngine(self.image_stream)
        
        # Get parameters
        method = self.params.get("method", "cartoon")
        edge_thickness = int(self.params.get("edge_thickness", 2))
        color_count = int(self.params.get("color_count", 12))
        simplify_factor = float(self.params.get("simplify_factor", 0.003))
        min_area = int(self.params.get("min_area", 20))
        cleanup_kernel = int(self.params.get("cleanup_kernel", 3))
        border_sensitivity = int(self.params.get("border_sensitivity", 12))
        portrait_mode = self.params.get("portrait_mode", False)
        
        # Process vector
        if method == "outline":
            result = self.vector_engine.process_flat_cartoon(
                edge_thickness=edge_thickness,
                color_count=color_count,
                simplify_factor=simplify_factor,
                min_area=min_area,
                cleanup_kernel=cleanup_kernel,
                border_sensitivity=border_sensitivity
            )
        else:
            result = self.vector_engine.process_hybrid_cartoon(
                edge_thickness=edge_thickness,
                color_count=color_count,
                portrait_mode=portrait_mode
            )
        
        return result
    
    def generate_bot(self, vector_result: Dict[str, Any]) -> BOTFormat:
        """Step 2: Convert vector to BOT format with decisions"""
        start_time = datetime.now()
        
        # Create BOT format
        bot = BOTFormat()
        
        # Store source parameters
        bot.source_params = self.params.copy()
        
        # Get image dimensions
        img = self.vector_engine.img
        h, w = img.shape[:2]
        
        # Calculate physical dimensions (assuming 96 DPI as default)
        dpi = self.params.get("dpi", 96)
        bot.width_mm = (w / dpi) * 25.4
        bot.height_mm = (h / dpi) * 25.4
        bot.aspect_ratio = w / h
        
        # Store metadata
        bot.metadata = {
            "width": w,
            "height": h,
            "dpi": dpi,
            "format": "bot",
            "created_by": "Auto-Digitize v4"
        }
        
        # Convert vector paths to BOT objects
        objects = []
        decisions = []
        
        # Extract color areas from vector result
        if "paths" in vector_result:
            for i, path_group in enumerate(vector_result["paths"]):
                color = path_group.get("color", [0, 0, 0])
                paths = path_group.get("paths", [])
                
                for j, path in enumerate(paths):
                    if len(path) < 3:
                        continue
                    
                    # Calculate area size
                    area_points = np.array(path, dtype=np.int32)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [area_points], 255)
                    area_size = cv2.countNonZero(mask)
                    
                    # Calculate width in mm
                    bbox = cv2.boundingRect(area_points)
                    width_px = bbox[2]
                    width_mm = (width_px / dpi) * 25.4
                    
                    # Auto-digitize decision
                    decision = self._make_fill_decision(width_mm, area_size, i, j)
                    decisions.append(asdict(decision))
                    
                    # Create BOT object
                    obj = {
                        "type": decision.fill_type,
                        "id": f"area_{i}_{j}",
                        "points": path,
                        "thread": {
                            "rgb": color,
                            "index": len(objects)
                        },
                        "density": bot.stitch_density,
                        "angle": bot.stitch_angle,
                        "underlay": {
                            "type": bot.underlay_type,
                            "density": bot.underlay_density
                        }
                    }
                    objects.append(obj)
        
        bot.objects = objects
        bot.decisions = decisions
        
        # Calculate processing time
        end_time = datetime.now()
        bot.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        self.bot_format = bot
        return bot
    
    def _make_fill_decision(self, width_mm: float, area_size: int, group_idx: int, path_idx: int) -> AutoDigitizeDecision:
        """Make deterministic fill type decision"""
        area_id = f"area_{group_idx}_{path_idx}"
        
        # Decision logic (deterministic)
        if width_mm < 10:
            fill_type = "satin"
            reason = "width < 10mm"
            confidence = 0.95
        elif area_size < 500:
            fill_type = "satin"
            reason = "small area"
            confidence = 0.85
        else:
            fill_type = "tatami"
            reason = "large area"
            confidence = 0.90
        
        return AutoDigitizeDecision(
            area_id=area_id,
            fill_type=fill_type,
            reason=reason,
            confidence=confidence,
            width_mm=width_mm,
            area_size_px=area_size
        )
    
    def export_bot(self) -> str:
        """Export BOT format as JSON"""
        if not self.bot_format:
            raise ValueError("BOT format not generated. Call generate_bot() first.")
        
        return json.dumps(asdict(self.bot_format), indent=2)
    
    def convert_to_format(self, format_type: str, preset_params: Optional[Dict[str, Any]] = None) -> bytes:
        """Step 3: Convert BOT to machine format"""
        if not self.bot_format:
            raise ValueError("BOT format not generated. Call generate_bot() first.")
        
        # Apply preset parameters (only affects stitch parameters, not decisions)
        bot_copy = BOTFormat(**asdict(self.bot_format))
        
        if preset_params:
            # Apply multipliers to stitch parameters
            if "density_multiplier" in preset_params:
                bot_copy.stitch_density *= preset_params["density_multiplier"]
            if "underlay_multiplier" in preset_params:
                bot_copy.underlay_density *= preset_params["underlay_multiplier"]
            if "stitch_length" in preset_params:
                # Update stitch length in objects
                for obj in bot_copy.objects:
                    obj["stitch_length"] = preset_params["stitch_length"]
        
        # Convert to pyembroidery pattern
        bot_dict = asdict(bot_copy)
        pattern = bot_json_to_pattern(bot_dict)
        
        # Export to requested format
        return export_pattern(pattern, format_type.lower())
    
    def get_thread_colors(self) -> List[Dict[str, Any]]:
        """Get thread colors from BOT format"""
        if not self.bot_format:
            return []
        
        threads = []
        for obj in self.bot_format.objects:
            thread = obj.get("thread", {})
            if "rgb" in thread:
                threads.append({
                    "rgb": thread["rgb"],
                    "hex": "#%02x%02x%02x" % tuple(thread["rgb"]),
                    "index": thread.get("index", 0)
                })
        
        return threads
    
    def get_decisions(self) -> List[Dict[str, Any]]:
        """Get auto-digitize decisions"""
        if not self.bot_format:
            return []
        return self.bot_format.decisions


def export_pattern(pattern, format_type: str) -> bytes:
    """Export pyembroidery pattern to bytes"""
    import pyembroidery
    
    out_stream = io.BytesIO()
    fmt = format_type.lower()
    if fmt == "emb": 
        fmt = "dst"
    
    # Write format
    if fmt == "dst": 
        pyembroidery.write_dst(pattern, out_stream)
    elif fmt == "pes": 
        pyembroidery.write_pes(pattern, out_stream)
    elif fmt == "exp": 
        pyembroidery.write_exp(pattern, out_stream)
    elif fmt == "jef": 
        pyembroidery.write_jef(pattern, out_stream)
    elif fmt == "vp3": 
        pyembroidery.write_vp3(pattern, out_stream)
    elif fmt == "xxx": 
        pyembroidery.write_xxx(pattern, out_stream)
    elif fmt == "pcs": 
        pyembroidery.write_pcs(pattern, out_stream)
    else: 
        pyembroidery.write_dst(pattern, out_stream)
    
    return out_stream.getvalue()
