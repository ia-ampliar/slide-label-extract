"""
Módulo para extração de lâminas de imagens de microscopia com detecção e correção de perspectiva.

Este módulo contém funções para:
- Detectar contornos de lâminas em imagens
- Ordenar pontos de contorno
- Corrigir perspectiva usando transformação de perspectiva
- Remover bordas pretas e aplicar margens
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
import json


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


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena os pontos de um contorno em ordem específica.
    
    Organiza os 4 pontos de um retângulo na seguinte ordem:
    top-left, top-right, bottom-right, bottom-left
    
    Usa a soma das coordenadas para identificar diagonais opostas:
    - Soma mínima = top-left (menor x + y)
    - Soma máxima = bottom-right (maior x + y)
    - Diferença mínima = top-right (maior x - y)
    - Diferença máxima = bottom-left (menor x - y)
    
    Args:
        pts (np.ndarray): Array com 4 pontos de formato (4, 2)
        
    Returns:
        np.ndarray: Pontos ordenados no formato (4, 2) com float32
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _draw_box_on_image(
    image: np.ndarray,
    box: np.ndarray,
    output_path: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 20
) -> None:
    """
    Desenha um retângulo em uma imagem e salva o resultado.
    
    Args:
        image (np.ndarray): Imagem BGR original
        box (np.ndarray): Array com 4 pontos do retângulo
        output_path (str): Caminho para salvar a imagem com o retângulo desenhado
        color (Tuple[int, int, int]): Cor do retângulo em BGR (padrão: verde)
        thickness (int): Espessura da linha do retângulo em pixels (padrão: 20)
        
    Returns:
        None
    """
    img_with_box = image.copy()
    box_draw = np.int32(box)
    cv2.drawContours(img_with_box, [box_draw], 0, color, thickness)
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_with_box)


def extract_slide(
    image_path: str,
    draw_box: bool = False,
    output_boximg_path: Optional[str] = None,
    hsv_lower: Optional[Tuple[int, int, int]] = None,
    hsv_upper: Optional[Tuple[int, int, int]] = None,
    kernel_size: Optional[Tuple[int, int]] = None,
    morph_close_iterations: Optional[int] = None,
    morph_open_iterations: Optional[int] = None,
    binary_threshold: Optional[int] = None,
    padding: Optional[int] = None,
    draw_box_color: Optional[Tuple[int, int, int]] = None,
    draw_box_thickness: Optional[int] = None,
    config_path: str = "data/config.json"
) -> np.ndarray:
    """
    Detecta e extrai a lâmina de uma imagem microscópica com correção de perspectiva.
    
    Processo:
    1. Carrega a imagem e converte para HSV
    2. Cria máscara baseada em faixa de valores HSV (V alto e S baixo para regiões claras)
    3. Aplica operações morfológicas (close e open) para consolidar o contorno
    4. Detecta o contorno de maior área (a lâmina)
    5. Encontra o retângulo rotacionado envolvente
    6. Ordena os pontos e calcula dimensões
    7. Aplica transformação de perspectiva para corrigir inclinação
    8. Remove bordas pretas e aplica margem ao resultado final
    
    Args:
        image_path (str): Caminho para a imagem de entrada
        draw_box (bool): Se True, salva uma imagem com o retângulo detectado desenhado.
                        Requer output_boximg_path definido. (padrão: False)
        output_boximg_path (Optional[str]): Caminho para salvar a imagem com o retângulo
                                           desenhado. Usado apenas se draw_box=True.
                                           (padrão: None)
        hsv_lower (Optional[Tuple[int, int, int]]): Limite inferior HSV para máscara.
                                                    Padrão do config.json se não fornecido.
        hsv_upper (Optional[Tuple[int, int, int]]): Limite superior HSV para máscara.
                                                    Padrão do config.json se não fornecido.
        kernel_size (Optional[Tuple[int, int]]): Tamanho do kernel para operações morfológicas.
                                                Padrão do config.json se não fornecido.
        morph_close_iterations (Optional[int]): Iterações para MORPH_CLOSE.
                                               Padrão do config.json se não fornecido.
        morph_open_iterations (Optional[int]): Iterações para MORPH_OPEN.
                                              Padrão do config.json se não fornecido.
        binary_threshold (Optional[int]): Valor de threshold para separar bordas pretas.
                                         Padrão do config.json se não fornecido.
        padding (Optional[int]): Margem em pixels ao redor do conteúdo útil.
                               Padrão do config.json se não fornecido.
        draw_box_color (Optional[Tuple[int, int, int]]): Cor do retângulo em BGR.
                                                        Padrão do config.json se não fornecido.
        draw_box_thickness (Optional[int]): Espessura da linha do retângulo em pixels.
                                           Padrão do config.json se não fornecido.
        config_path (str): Caminho para arquivo config.json (padrão: "data/config.json")
    
    Returns:
        np.ndarray: Imagem da lâmina extraída e corrigida
        
    Raises:
        FileNotFoundError: Se a imagem ou arquivo de config não podem ser carregados
        RuntimeError: Se nenhum contorno for encontrado na imagem
        ValueError: Se draw_box=True mas output_boximg_path não foi fornecido
    """
    # Carregar configurações do arquivo
    config = load_config(config_path)
    mask_params = config.get("mask_params", {})
    
    # Usar parâmetros fornecidos ou valores padrão do config
    hsv_lower = tuple(hsv_lower) if hsv_lower else tuple(mask_params.get("hsv_lower", [0, 0, 120]))
    hsv_upper = tuple(hsv_upper) if hsv_upper else tuple(mask_params.get("hsv_upper", [179, 80, 255]))
    kernel_size = tuple(kernel_size) if kernel_size else tuple(mask_params.get("kernel_size", [25, 25]))
    morph_close_iterations = morph_close_iterations or mask_params.get("morph_close_iterations", 2)
    morph_open_iterations = morph_open_iterations or mask_params.get("morph_open_iterations", 1)
    binary_threshold = binary_threshold or mask_params.get("binary_threshold", 10)
    padding = padding or mask_params.get("padding", 5)
    draw_box_color = tuple(draw_box_color) if draw_box_color else tuple(mask_params.get("draw_box_color", [0, 255, 0]))
    draw_box_thickness = draw_box_thickness or mask_params.get("draw_box_thickness", 20)
    
    # Verificar se arquivo existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    # Carregar imagem
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Não foi possível carregar a imagem: {image_path}")
    
    h, w = img_bgr.shape[:2]
    
    # ===== ETAPA 1: Criar máscara HSV =====
    # Conversão para HSV: melhor para separar cores de fundo escuro
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Máscara: regiões claras (V alto) e pouco saturadas (S baixo)
    # Estes valores são carregados do arquivo de configuração e ajustáveis
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # ===== ETAPA 2: Operações morfológicas =====
    # Kernel retangular para operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # MORPH_CLOSE: fecha pequenos buracos dentro da lâmina
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)
    
    # MORPH_OPEN: remove pequenos ruídos fora da lâmina
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iterations)
    
    # ===== ETAPA 3: Detectar contornos =====
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError(
            "Nenhum contorno encontrado. Ajuste os thresholds HSV ou verifique a imagem."
        )
    
    # Selecionar contorno de maior área (deve ser a lâmina)
    c = max(cnts, key=cv2.contourArea)
    
    # ===== ETAPA 4: Encontrar retângulo rotacionado =====
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    
    # ===== ETAPA 5: Desenhar retângulo se solicitado =====
    if draw_box:
        if output_boximg_path is None:
            raise ValueError(
                "draw_box=True requer output_boximg_path para salvar a imagem"
            )
        _draw_box_on_image(img_bgr, box, output_boximg_path, draw_box_color, draw_box_thickness)
    
    # Converter para float32 e ordenar pontos
    box = np.array(box, dtype="float32")
    box = order_points(box)
    
    # ===== ETAPA 6: Calcular dimensões do retângulo =====
    (tl, tr, br, bl) = box
    
    # Largura: máximo entre os dois lados
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    
    # Altura: máximo entre os dois lados
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    
    # ===== ETAPA 7: Transformação de perspectiva =====
    # Pontos de destino (retângulo sem rotação)
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")
    
    # Calcular matriz de transformação perspectiva
    M = cv2.getPerspectiveTransform(box, dst)
    
    # Aplicar transformação
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    
    # ===== ETAPA 8: Remover bordas pretas e aplicar margem =====
    # Converter para escala de cinza para detectar regiões não-pretas
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Threshold: separa pixels pretos do resto usando valor configurável
    _, bin2 = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Encontrar contorno do conteúdo útil (não-preto)
    cnts2, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2 = max(cnts2, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c2)
    
    # Aplicar margem configurável ao redor do conteúdo
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(warped.shape[1], x + ww + padding)
    y1 = min(warped.shape[0], y + hh + padding)
    
    # Extrair região final
    final = warped[y0:y1, x0:x1]
    
    return final


def main():
    """
    Função principal para teste do módulo de extração de lâminas.
    
    Esta função é usada para testar a extração de lâminas em uma imagem de exemplo.
    Ela carrega uma imagem, chama a função extract_slide e salva o resultado.
    
    Ajuste os caminhos das imagens no arquivo config.json e parâmetros conforme necessário para seus testes.
    """
    config = load_config()
    test_image_path = config.get("slide_img_path", "data/test_slide.jpg")
    output_image_path = config.get("output_image_path", "data/output/extracted_slide.jpg")
    output_boximg_path = config.get("output_boximg_path", "data/output/box_detected.jpg")
    
    try:
        extracted_slide = extract_slide(
            image_path=test_image_path,
            draw_box=True,
            output_boximg_path=output_boximg_path,
            config_path="data/config.json"
        )
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, extracted_slide)
        print(f"Lâmina extraída salva em: {output_image_path}")
    except Exception as e:
        print(f"Erro durante a extração: {e}")


if __name__ == "__main__":
    main()