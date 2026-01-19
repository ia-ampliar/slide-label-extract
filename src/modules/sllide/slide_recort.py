import cv2
import numpy as np
from src.config.config import Config

def order_points(pts):
    # ordena: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def crop_slide_with_label(img_bgr):
    h, w = img_bgr.shape[:2]

    # 1) HSV ajuda a separar fundo escuro
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # 2) Máscara: regiões claras (V alto) e pouco saturadas (S baixo)
    # Ajuste fino pode ser necessário dependendo da iluminação
    # cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    mask = cv2.inRange(
        hsv, 
        (Config.LOW_H, Config.LOW_S, Config.LOW_V), 
        (Config.HIGH_H, Config.HIGH_S, Config.HIGH_V)
    )

    # 3) Morfologia para consolidar o "bloco" da lâmina
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (Config.KERNEL_SIZE, Config.KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) Contornos
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Nenhum contorno encontrado. Ajuste thresholds da máscara.")

    c = max(cnts, key=cv2.contourArea)

    # 5) Retângulo rotacionado
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")
    box = order_points(box)

    # dimensões do retângulo
    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))

    # 6) Opcional: remover bordas pretas e dar uma margem
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bin2 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # remove preto
    cnts2, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2 = max(cnts2, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c2)

    pad = 5
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(warped.shape[1], x + ww + pad)
    y1 = min(warped.shape[0], y + hh + pad)
    final = warped[y0:y1, x0:x1]

    return final, mask

def extract_slide():
    img_name = "macro_0"
    img = cv2.imread(f"data/{img_name}.jpg")
    slide, mask = crop_slide_with_label(img)
    cv2.imwrite(f"output/{img_name}_crop.png", slide)

if __name__ == "__main__":
    
    extract_slide()