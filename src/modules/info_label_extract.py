"""
Módulo para extração de informações de etiquetas em imagens usando OCR.

Este módulo contém funções para:
- Extrair padrões de texto usando expressões regulares
- Processar imagens e aplicar OCR (Tesseract)
- Extrair informações de casos/IDs das etiquetas de lâminas
"""

import cv2
import pytesseract
import re
import numpy as np
from slide_extract import load_config, extract_slide
import json
import os
from typing import List, Optional, Dict, Any

from rapidocr_onnxruntime import RapidOCR


def load_config(config_path: str = "data/config.json") -> Dict[str, Any]:
    """
    Carrega as configurações de um arquivo JSON.
    
    Args:
        config_path (str): Caminho para o arquivo config.json (padrão: "data/config.json")
        
    Returns:
        Dict[str, Any]: Dicionário com as configurações
        
    Raises:
        FileNotFoundError: Se o arquivo de configuração não for encontrado
        json.JSONDecodeError: Se o arquivo não for um JSON válido
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def extract_pattern(
    text_input: str,
    patterns: Optional[List[str]] = None,
    config_path: str = "data/config.json"
) -> Optional[List[str]]:
    """
    Extrai padrões de texto de uma string usando expressões regulares.
    
    Se nenhum padrão for fornecido, utiliza os padrões definidos em 'case_id_list'
    do arquivo config.json. Procura por todos os padrões fornecidos e retorna
    o primeiro conjunto de correspondências encontrado.
    
    Args:
        text_input (str): String de entrada para buscar padrões
        patterns (Optional[List[str]]): Lista de expressões regulares a serem procuradas.
                                       Se None, usa padrões do config.json (padrão: None)
        config_path (str): Caminho para arquivo config.json (padrão: "data/config.json")
        
    Returns:
        Optional[List[str]]: Lista com os padrões encontrados, ou None se nenhum encontrado
        
    Raises:
        ValueError: Se text_input estiver vazio
    """
    if not text_input or not isinstance(text_input, str):
        raise ValueError("text_input deve ser uma string não-vazia")
    
    # Usar padrões do config se não fornecidos
    if patterns is None:
        config = load_config(config_path)
        patterns = config.get("case_id_list", [r"[A-Z][0-9]{2}-[0-9]{6}"])
    
    # Procurar por cada padrão
    for pattern in patterns:
        try:
            # matches = re.findall(pattern, text_input)

            matches = re.search(pattern, text_input, re.DOTALL)
            if matches:
                resultado = f"{matches.group(1)}{matches.group(2)}-{matches.group(3)}"
                return [resultado]

            # if matches:
            #     return matches
        except re.error as e:
            print(f"Erro ao compilar expressão regular '{pattern}': {e}")
            continue
    
    return None


def _preprocess_roi(
    roi_bgr: cv2.Mat,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: tuple = (8, 8),
    upscale_factor: float = 2.0,
    blur_kernel: tuple = (3, 3)
) -> cv2.Mat:
    """
    Pré-processa uma região de interesse (ROI) para melhorar OCR.
    
    Aplica as seguintes operações em sequência:
    1. Conversão para escala de cinza
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Upscale da imagem
    4. Desfoque gaussiano para redução de ruído
    5. Binarização automática (OTSU)
    6. Operação morfológica de abertura para limpeza
    
    Args:
        roi_bgr (cv2.Mat): Imagem ROI em formato BGR
        clahe_clip_limit (float): Limite de clipping para CLAHE (padrão: 2.0)
        clahe_tile_grid (tuple): Tamanho da grade de tiles para CLAHE (padrão: (8, 8))
        upscale_factor (float): Fator de upscale da imagem (padrão: 2.0)
        blur_kernel (tuple): Tamanho do kernel de desfoque gaussiano (padrão: (3, 3))
        
    Returns:
        cv2.Mat: Imagem binarizada e processada
    """
    # Conversão para escala de cinza
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Contraste adaptativo local (melhora muito a qualidade da etiqueta)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    gray = clahe.apply(gray)
    
    # Upscale: melhora OCR em texto pequeno
    gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, 
                      interpolation=cv2.INTER_CUBIC)
    
    # Redução de ruído leve com desfoque gaussiano
    gray = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Binarização automática usando Otsu
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpeza de grãos: remove pequenas componentes de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return bw


def extract_label_info_ocr(
    slide_image: np.ndarray,
    roi_width_ratio: float = 0.25,
    rotate_90: bool = True,
    tesseract_config: str = "--oem 3 --psm 6",
    tesseract_lang: str = "por",
    config_path: str = "data/config.json",
    patterns: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extrai informações de ID de etiquetas em imagens usando OCR (Tesseract).
    
    Processo:
    1. Carrega a imagem e extrai ROI (região de interesse) da etiqueta
    2. Rotaciona ROI se necessário
    3. Pré-processa a imagem para melhorar qualidade do OCR
    4. Aplica Tesseract OCR
    5. Extrai IDs usando padrões configuráveis
    
    Args:
        slide_image (np.ndarray): Imagem da placa de slide
        roi_width_ratio (float): Proporção da largura a usar como ROI
                                (padrão: 0.25 = 25% da largura)
        rotate_90 (bool): Se True, rotaciona ROI 90° para a direita (padrão: True)
        tesseract_config (str): Configuração do Tesseract PSM/OEM (padrão: "--oem 3 --psm 6")
        tesseract_lang (str): Idioma para OCR (padrão: "por" para português)
        config_path (str): Caminho para arquivo config.json (padrão: "data/config.json")
        patterns (Optional[List[str]]): Padrões regex customizados. Se None, usa config.json
        
    Returns:
        Optional[str]: ID da etiqueta encontrado, ou None se não encontrado
        
    Raises:
        FileNotFoundError: Se a imagem não for encontrada
        RuntimeError: Se a imagem não puder ser carregada
        ValueError: Se ROI não puder ser extraída
    """
    # Validar e carregar imagem
    if slide_image is None:
        raise ValueError("Imagem da placa de slide é inválida")

    h, w = slide_image.shape[:2]

    # ===== ETAPA 1: Extrair ROI da etiqueta =====
    # Extrai apenas a região esquerda da imagem (onde fica a etiqueta)
    roi_width = int(w * roi_width_ratio)
    roi_bgr = slide_image[:, :roi_width]
    
    if roi_bgr.size == 0:
        raise ValueError(f"ROI inválida. Dimensões: {roi_bgr.shape}")
    
    # ===== ETAPA 2: Rotação (opcional) =====
    if rotate_90:
        roi_bgr = cv2.rotate(roi_bgr, cv2.ROTATE_90_CLOCKWISE)
    
    # ===== ETAPA 3: Pré-processamento da imagem =====
    bw = _preprocess_roi(roi_bgr)
    
    # ===== ETAPA 4: OCR com RapidOCR =====
    try:
        engine = RapidOCR()
        result, _ = engine(bw)
        # concatena todo o texto
        text = "\n".join([linha[1] for linha in result])
        print(f"Texto extraído pelo OCR: \n{text}")

    except Exception as e:
        print(f"Erro durante OCR: {e}")
        return None
    
    # ===== ETAPA 5: Extração de padrões =====
    found_ids = extract_pattern(text, patterns=patterns, config_path=config_path)
    print(f"Padrões encontrados: {found_ids}")
    
    if found_ids:
        return found_ids[0]  # Retorna o primeiro ID encontrado
    
    return None


def main():
    """
    Função principal para teste do módulo de extração de informações de etiquetas.
    
    Esta função é usada para testar a extração de IDs de etiquetas em uma imagem de exemplo.
    Ela carrega uma imagem, aplica OCR e extrai o ID usando padrões do config.json.
    
    Ajuste os caminhos das imagens e padrões no arquivo config.json conforme necessário.
    """
    config = load_config()
    test_image_path = config.get("slide_img_path", "data/slide_img.png")
    patterns = config.get("case_id_list", [r"([A-Z]).*?([0-9]{2}).*?-.*?([0-9]{6})"])

    slide_image = extract_slide(image_path=test_image_path)
    
    try:
        extracted_id = extract_label_info_ocr(
            slide_image=slide_image,
            patterns=patterns,
            config_path="data/config.json"
        )
        
        if extracted_id:
            print(f"✓ ID da etiqueta extraído com sucesso: {extracted_id}")
        else:
            print("✗ Nenhum ID de etiqueta foi encontrado na imagem")
            
    except Exception as e:
        print(f"✗ Erro durante a extração: {e}")


if __name__ == "__main__":
    main()