"""
Pacote principal para processamento e análise de lâminas de microscopia.
"""

from .modules import (
    extract_slide,
    extract_label_info_ocr,
    extract_pattern,
    load_config,
    order_points
)

__all__ = [
    "extract_slide",
    "extract_label_info_ocr", 
    "extract_pattern",
    "load_config",
    "order_points"
]
