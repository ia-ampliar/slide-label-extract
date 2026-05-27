# Slide Label Extract - Documentação

Sistema completo para extração de lâminas de microscopia e informações de etiquetas usando processamento de imagem e OCR.

## 📋 Visão Geral

Este projeto implementa um pipeline de processamento de lâminas microscópicas que:

1. **Extrai a lâmina** da imagem com correção automática de perspectiva
2. **Extrai informações** da etiqueta usando OCR (Tesseract)
3. **Identifica IDs de casos** usando expressões regulares configuráveis

## 🗂️ Estrutura do Projeto

```
slide-label-extract/
├── src/
│   ├── __init__.py
│   └── modules/
│       ├── __init__.py
│       ├── slide_extract.py        # Extração de lâminas
│       └── info_label_extract.py   # Extração de etiquetas via OCR
├── data/
│   └── config.json                 # Configurações centralizadas
├── main.py                         # Script principal de orquestração
├── LICENSE
└── README.md                       # Este arquivo
```

## ⚙️ Configuração

Todas as configurações estão centralizadas em `data/config.json`:

```json
{
    "slide_img_path": "data/slide_img.png",
    "output_image_path": "data/output/extracted_slide.jpg",
    "case_id_list": ["[A-Z][0-9]{2}-[0-9]{6}"],
    "mask_params": {
        "hsv_lower": [0, 0, 120],
        "hsv_upper": [179, 80, 255],
        "kernel_size": [25, 25],
        "morph_close_iterations": 2,
        "morph_open_iterations": 1,
        "binary_threshold": 10,
        "padding": 5,
        "draw_box_color": [0, 255, 0],
        "draw_box_thickness": 20
    },
    "ocr_params": {
        "roi_width_ratio": 0.25,
        "rotate_90": true,
        "tesseract_config": "--oem 3 --psm 6",
        "tesseract_lang": "por",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": [8, 8],
        "upscale_factor": 2.0,
        "blur_kernel": [3, 3]
    }
}
```

### Parâmetros Principais

#### `case_id_list`
- **Tipo**: Array de strings (regex patterns)
- **Padrão**: `["[A-Z][0-9]{2}-[0-9]{6}"]`
- **Descrição**: Padrões de expressão regular para extrair IDs de casos
- **Exemplo**: Extrai "AB-123456"

#### `mask_params`
Parâmetros para detecção da lâmina:

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `hsv_lower` | Array | [0, 0, 120] | Limite inferior HSV |
| `hsv_upper` | Array | [179, 80, 255] | Limite superior HSV |
| `kernel_size` | Array | [25, 25] | Tamanho do kernel morfológico |
| `morph_close_iterations` | Int | 2 | Iterações de fechamento |
| `morph_open_iterations` | Int | 1 | Iterações de abertura |
| `binary_threshold` | Int | 10 | Threshold para remover bordas pretas |
| `padding` | Int | 5 | Margem ao redor do conteúdo |
| `draw_box_color` | Array | [0, 255, 0] | Cor BGR do retângulo desenhado |
| `draw_box_thickness` | Int | 20 | Espessura do retângulo |

#### `ocr_params`
Parâmetros para OCR de etiquetas:

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `roi_width_ratio` | Float | 0.25 | Proporção da largura para ROI |
| `rotate_90` | Bool | true | Rotaciona ROI 90° |
| `tesseract_config` | String | "--oem 3 --psm 6" | Configuração Tesseract |
| `tesseract_lang` | String | "por" | Idioma para OCR |
| `clahe_clip_limit` | Float | 2.0 | Limite de clipping CLAHE |
| `clahe_tile_grid` | Array | [8, 8] | Grade de tiles CLAHE |
| `upscale_factor` | Float | 2.0 | Fator de upscale |
| `blur_kernel` | Array | [3, 3] | Kernel de desfoque |

## 🚀 Uso

### Via Script Principal

```bash
# Usar configurações padrão do config.json
python main.py

# Especificar imagem
python main.py --image data/minha_lamina.jpg

# Com opções customizadas
python main.py --image data/minha_lamina.jpg --no-label --no-save

# Modo verbose
python main.py --image data/minha_lamina.jpg --verbose
```

### Via Python (Uso Programático)

```python
from src.modules import extract_slide, extract_label_info_ocr

# Extrair lâmina
slide = extract_slide(
    image_path="data/slide.jpg",
    config_path="data/config.json"
)

# Extrair informações da etiqueta
case_id = extract_label_info_ocr(
    image_path="data/slide.jpg",
    config_path="data/config.json"
)

print(f"ID do caso: {case_id}")
```

### Via main.py (Integrado)

```python
from main import process_slide

results = process_slide(
    image_path="data/slide.jpg",
    extract_label=True,
    save_output=True,
    verbose=True
)

print(f"Sucesso: {results['success']}")
print(f"ID extraído: {results['label_id']}")
```

## 📚 Módulos

### `slide_extract.py`

Extração da lâmina com correção de perspectiva.

**Funções principais:**
- `load_config(config_path)` - Carrega configurações JSON
- `extract_slide(image_path, ...)` - Extrai e corrige lâmina
- `order_points(pts)` - Ordena pontos de contorno
- `main()` - Teste do módulo

**Processo:**
1. Detecção em HSV
2. Operações morfológicas
3. Detecção de contornos
4. Transformação de perspectiva
5. Remoção de bordas pretas

### `info_label_extract.py`

Extração de informações de etiquetas via OCR.

**Funções principais:**
- `extract_label_info_ocr(image_path, ...)` - OCR completo
- `extract_pattern(text, patterns)` - Extração de regex
- `_preprocess_roi(roi_bgr)` - Pré-processamento
- `main()` - Teste do módulo

**Processo:**
1. Extração de ROI
2. Rotação (opcional)
3. Pré-processamento
4. OCR com Tesseract
5. Extração de padrões

## 🔧 Instalação de Dependências

```bash
# Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Instalar dependências Python
pip install -r requirements.txt

# Instalar Tesseract (para OCR)
# Windows: Download em https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

## 📊 Exemplo de Saída

```
============================================================
Processando: slide.jpg
============================================================

[1/3] Extraindo lâmina com correção de perspectiva...
   ✓ Lâmina extraída com sucesso
   - Dimensões: 1024x768 pixels
   - Box detectado em: data/output/box_detected.jpg
   - Salvo em: data/output/extracted_slide.jpg

[2/3] Extraindo informações de etiqueta via OCR...
   ✓ ID da etiqueta extraído: AB-123456

[3/3] Resumo do processamento:
   - Lâmina extraída: ✓ Sim
   - ID da etiqueta: AB-123456
============================================================
```

## 🐛 Troubleshooting

### Nenhuma lâmina encontrada
- Ajuste `hsv_lower` e `hsv_upper` em `mask_params`
- Verifique a iluminação da imagem
- Ajuste `kernel_size`

### ID não encontrado
- Verifique se o padrão regex em `case_id_list` está correto
- Ajuste `tesseract_config` ou `tesseract_lang`
- Aumente `upscale_factor` em `ocr_params`

### Tesseract não encontrado
- Instale Tesseract (ver seção Instalação)
- Configure variável de ambiente `TESSDATA_PREFIX`

## 📝 Licença

Veja arquivo [LICENSE](../LICENSE)

## 🤝 Contribuições

Para contribuições, favor seguir:
1. Manter coesão com estilo de código existente
2. Adicionar docstrings em português
3. Manter testes atualizados
4. Seguir convenções de nomenclatura
