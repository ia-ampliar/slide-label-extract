"""
Pacote de módulos para processamento de lâminas de microscopia.

Módulos:
- slide_extract: Extração e correção de perspectiva de lâminas
- info_label_extract: Extração de informações de etiquetas usando OCR
"""

from .slide_extract import extract_slide, load_config, order_points
from .info_label_extract import extract_label_info_ocr, extract_pattern

__all__ = [
    "extract_slide",
    "extract_label_info_ocr",
    "extract_pattern",
    "load_config",
    "order_points"
]

__version__ = "1.0.0"
