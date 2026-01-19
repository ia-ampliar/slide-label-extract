# README.md - Extrair Foto Macro

# Extração automática da lâmina (com etiqueta) a partir de foto do slide

Este módulo implementa um pipeline em **Python + OpenCV** para **isolar e recortar automaticamente** a região correspondente à **lâmina histológica inteira com etiqueta**, a partir de uma foto capturada em bancada (fundo escuro/texturizado, lâmina clara).

A saída final é um recorte **retificado** (corrigido de rotação/perspectiva), pronto para:

- padronização de dataset (controle de qualidade / ingestão em pipeline),
- pré-processamento antes de OCR/QR,
- preparo para digitalização ou indexação.

---

## Sumário

- [Visão geral da abordagem](#visão-geral-da-abordagem)
- [Premissas e limitações](#premissas-e-limitações)
- [Dependências](#dependências)
- [Como usar](#como-usar)
- [Descrição do pipeline (passo a passo)](#descrição-do-pipeline-passo-a-passo)
- [Detalhamento das funções](#detalhamento-das-funções)
  - [order_points](#order_points)
  - [crop_slide_with_label](#crop_slide_with_label)
- [Parâmetros ajustáveis](#parâmetros-ajustáveis)
- [Modo de falha e reforços recomendados](#modo-de-falha-e-reforços-recomendados)
- [Boas práticas](#boas-práticas)

---

## Visão geral da abordagem

O algoritmo explora um contraste comum em fotos de bancada:

- **Fundo**: escuro (baixo valor/brightness), texturizado
- **Lâmina + etiqueta**: área clara (alto valor/brightness), pouco saturada

A estratégia é:

1. Converter para **HSV** e segmentar **regiões claras** usando threshold em `(S, V)`.
2. Consolidar essa máscara com **operações morfológicas** para formar um “bloco” único.
3. Encontrar o **maior contorno externo** (assumindo que é a lâmina completa).
4. Obter o **retângulo mínimo rotacionado** (`minAreaRect`) e os 4 vértices.
5. Aplicar **transformação de perspectiva** (`warpPerspective`) para retificar.
6. Fazer um **crop final** removendo bordas pretas e adicionando uma margem.

---

## Premissas e limitações

✅ Funciona bem quando:

- fundo é predominantemente escuro,
- a lâmina/etiqueta aparecem mais claras que o fundo,
- a lâmina é o maior objeto claro na cena.

⚠️ Pode falhar quando:

- fundo não é escuro (mesa clara),
- reflexos especulares intensos no vidro “estouram” áreas e mudam o histograma,
- a lâmina está muito subexposta (não fica “clara” o bastante),
- existem outros objetos claros grandes (papel, luva, etc.) competindo com a lâmina.

---

## Dependências

- Python 3.8+
- OpenCV:
    - `opencv-python`
- NumPy:
    - `numpy`

Instalação:

```bash
pip install opencv-python numpy
```

---

## Como usar

### Uso básico (recortar e salvar)

```python
import cv2

img = cv2.imread("sua_imagem.jpg")
slide_crop, mask = crop_slide_with_label(img)

cv2.imwrite("slide_crop.png", slide_crop)
cv2.imwrite("mask_debug.png", mask)

```

Saídas:

- `slide_crop.png`: imagem final recortada e retificada (lâmina + etiqueta).
- `mask_debug.png`: máscara binária usada para segmentação (útil para ajuste fino).

---

## Descrição do pipeline (passo a passo)

### 1) Conversão de cor: BGR → HSV

A segmentação é feita em HSV porque:

- **V (Value)** captura brilho/intensidade,
- **S (Saturation)** ajuda a separar áreas muito saturadas do tecido vs. vidro/etiqueta.

### 2) Máscara por threshold (regiões claras e pouco saturadas)

Usa-se:

- `V` alto para regiões claras,
- `S` baixo/moderado para evitar “cores fortes”.

```python
mask = cv2.inRange(hsv, (0,0,120), (179,80,255))
```

### 3) Morfologia para consolidar a máscara

Objetivos:

- **CLOSE**: unir regiões próximas, preencher buracos (tornar a máscara da lâmina mais contínua)
- **OPEN**: remover pequenos ruídos (pontos e grãos)

Kernel retangular grande:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
```

### 4) Contornos e seleção do maior

Depois de limpar a máscara, encontra-se o maior contorno externo.

A hipótese aqui é: **a lâmina é o maior objeto detectado**.

### 5) Retângulo rotacionado e retificação (warpPerspective)

- `minAreaRect` fornece um retângulo mínimo mesmo com rotação.
- `boxPoints` dá os 4 vértices.
- `order_points` ordena os pontos para a transformação ficar consistente.
- `getPerspectiveTransform + warpPerspective` retificam a lâmina.

### 6) Crop final e remoção de bordas pretas

Após o warp, surgem bordas pretas.

O algoritmo:

- binariza (removendo preto),
- acha o maior contorno de “conteúdo real”,
- calcula `boundingRect`,
- aplica um `pad` (margem) e recorta.

---

## Detalhamento das funções

### `order_points`

**Objetivo:** ordenar os 4 vértices do retângulo em uma ordem fixa para o warp:

- `top-left (tl)`
- `top-right (tr)`
- `bottom-right (br)`
- `bottom-left (bl)`

**Por que isso importa?**

A transformação de perspectiva depende de mapear corretamente cada vértice de origem para o destino. Se a ordem estiver trocada, o resultado pode sair invertido, rotacionado ou “dobrado”.

**Estratégia usada:**

- soma `x+y` para achar o canto superior esquerdo (menor soma) e inferior direito (maior soma),
- diferença `x-y` para superior direito (menor diff) e inferior esquerdo (maior diff).

---

### `crop_slide_with_label`

**Assinatura:**

```python
final, mask = crop_slide_with_label(img_bgr)
```

**Entrada:**

- `img_bgr`: imagem em formato BGR (como lida por `cv2.imread`).

**Saídas:**

- `final`: recorte retificado (lâmina + etiqueta).
- `mask`: máscara binária (útil para debug e ajuste).

**Etapas internas:**

1. HSV + threshold
2. close/open
3. contornos (maior)
4. minAreaRect + warpPerspective
5. crop final removendo preto

**Erros possíveis:**

- Se nenhum contorno for detectado, a função levanta:

```python
RuntimeError("Nenhum contorno encontrado. Ajuste thresholds da máscara.")
```

---

## Parâmetros ajustáveis

### Threshold HSV (segmentação)

```python
mask = cv2.inRange(hsv, (0,0,120), (179,80,255))
```

- `V_min = 120`
    
    Aumente se o fundo está “vazando” para a máscara.
    
    Diminua se a lâmina estiver escura (subexposta).
    
- `S_max = 80`
    
    Aumente se a máscara está pegando só etiqueta (perdendo tecido/vidro).
    
    Diminua se o tecido muito colorido estiver sendo interpretado como “objeto”.
    

### Kernel morfológico

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
```

- Aumente o kernel se:
    - a máscara vem “quebrada” em várias partes,
    - há buracos grandes dentro da região da lâmina.
- Diminua o kernel se:
    - a máscara está “engolindo” áreas do fundo próximas.

### Padding final

```python
pad =5
```

- Aumente para garantir borda de segurança.
- Reduza se quiser crop mais justo.

---

## Modo de falha e reforços recomendados

### Caso 1: fundo claro ou variação de iluminação

**Sintoma:** a máscara pega muita coisa além da lâmina.

**Soluções:**

- usar **Lab** e threshold no canal `L`,
- aplicar **CLAHE** no canal de intensidade antes do threshold.

### Caso 2: reflexos fortes

**Sintoma:** a máscara quebra ou inclui regiões reflexivas fora da lâmina.

**Soluções:**

- adicionar etapa de detecção de retângulos por **Canny + Hough Lines**,
- usar `grabCut` inicializado com o bounding box do contorno principal.

### Caso 3: outros objetos claros grandes

**Sintoma:** maior contorno não é a lâmina.

**Soluções:**

- filtrar por **aspect ratio** esperado (lâmina tem proporção típica),
- filtrar por **retangularidade** (área do contorno / área do minAreaRect).

### Caso 4: lâmina subexposta

**Sintoma:** nenhum contorno encontrado.

**Soluções:**

- diminuir `V_min`,
- equalizar brilho (CLAHE) antes de threshold.

---

## Boas práticas

- Salvar `mask` em lote para auditoria e ajuste rápido de parâmetros.
- Fixar condições de captura:
    - distância/câmera,
    - iluminação constante,
    - fundo escuro uniforme.
- Considerar validações:
    - se a saída final tiver dimensões muito pequenas, disparar alerta,
    - se aspect ratio estiver fora do esperado, classificar como “falha de extração”.
