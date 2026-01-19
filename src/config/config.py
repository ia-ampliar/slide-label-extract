from pathlib import Path


class Config:
    # HSV thresholds para detecção de regiões claras e pouco saturadas
    LOW_H: int = 0
    LOW_S: int = 0
    LOW_V: int = 120

    HIGH_H: int = 179
    HIGH_S: int = 80
    HIGH_V: int = 255

    # Tamanho do kernel para operações morfológicas
    KERNEL_SIZE: int = 25  