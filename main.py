"""
Script principal para processamento completo de lâminas de microscopia.

Este script integra os dois módulos principais:
1. Extração da lâmina da imagem
2. Extração de informações (ID) da etiqueta

Uso:
    python main.py --image <path/to/image> [options]
    python main.py  # usa config.json padrão
"""

import sys
import os
import argparse
from pathlib import Path

# Adicionar src ao path para importações
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from src.modules.slide_extract import extract_slide, load_config as load_slide_config
from src.modules.info_label_extract import extract_label_info_ocr, extract_pattern


def process_slide(
    image_path: str,
    config_path: str = "data/config.json",
    extract_label: bool = True,
    save_output: bool = True,
    verbose: bool = True
) -> dict:
    """
    Processa uma lâmina de microscopia de forma completa.
    
    Realiza:
    1. Extração da lâmina com correção de perspectiva
    2. Extração da etiqueta (opcional)
    3. Extração de ID usando OCR (opcional)
    
    Args:
        image_path (str): Caminho para a imagem de entrada
        config_path (str): Caminho para arquivo config.json
        extract_label (bool): Se True, extrai informações da etiqueta
        save_output (bool): Se True, salva imagens de saída
        verbose (bool): Se True, imprime progresso
        
    Returns:
        dict: Dicionário com resultados contendo:
            - "success": bool indicando sucesso geral
            - "slide_extracted": bool se lâmina foi extraída
            - "slide_image": np.ndarray imagem da lâmina extraída
            - "label_id": str ID extraído da etiqueta (se extract_label=True)
            - "errors": lista de mensagens de erro
    """
    results = {
        "success": False,
        "slide_extracted": False,
        "slide_image": None,
        "label_id": None,
        "errors": []
    }
    
    try:
        # Carregar configurações
        config = load_slide_config(config_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processando: {os.path.basename(image_path)}")
            print(f"{'='*60}")
        
        # ===== ETAPA 1: Extração da lâmina =====
        if verbose:
            print("\n[1/3] Extraindo lâmina com correção de perspectiva...")
        
        try:
            output_boximg_path = config.get("output_boximg_path", "data/output/box_detected.jpg")
            
            slide_image = extract_slide(
                image_path=image_path,
                draw_box=save_output,
                output_boximg_path=output_boximg_path if save_output else None,
                config_path=config_path
            )
            
            results["slide_extracted"] = True
            results["slide_image"] = slide_image
            
            if verbose:
                print(f"   ✓ Lâmina extraída com sucesso")
                print(f"   - Dimensões: {slide_image.shape[1]}x{slide_image.shape[0]} pixels")
            
            # Salvar imagem da lâmina extraída
            if save_output:
                output_image_path = config.get("output_image_path", "data/output/extracted_slide.jpg")
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                
                import cv2
                cv2.imwrite(output_image_path, slide_image)
                
                if verbose:
                    print(f"   - Salvo em: {output_image_path}")
                    if "output_boximg_path" in config:
                        print(f"   - Box detectado em: {output_boximg_path}")
        
        except Exception as e:
            results["errors"].append(f"Erro na extração da lâmina: {str(e)}")
            if verbose:
                print(f"   ✗ Erro: {str(e)}")
            return results
        
        # ===== ETAPA 2: Extração de informações da etiqueta (OCR) =====
        if extract_label:
            if verbose:
                print("\n[2/3] Extraindo informações de etiqueta via OCR...")
            
            try:
                patterns = config.get("case_id_list", [r"[A-Z][0-9]{2}-[0-9]{6}"])
                
                label_id = extract_label_info_ocr(
                    image_path=image_path,
                    patterns=patterns,
                    config_path=config_path
                )
                
                results["label_id"] = label_id
                
                if label_id:
                    if verbose:
                        print(f"   ✓ ID da etiqueta extraído: {label_id}")
                else:
                    if verbose:
                        print(f"   ⚠ Nenhum ID encontrado na etiqueta")
                        
            except Exception as e:
                results["errors"].append(f"Erro na extração OCR: {str(e)}")
                if verbose:
                    print(f"   ⚠ Aviso: {str(e)}")
        
        # ===== RESUMO FINAL =====
        results["success"] = results["slide_extracted"]
        
        if verbose:
            print(f"\n[3/3] Resumo do processamento:")
            print(f"   - Lâmina extraída: {'✓ Sim' if results['slide_extracted'] else '✗ Não'}")
            if extract_label:
                print(f"   - ID da etiqueta: {results['label_id'] or '(não encontrado)'}")
            print(f"{'='*60}\n")
        
        return results
    
    except Exception as e:
        results["errors"].append(f"Erro geral: {str(e)}")
        if verbose:
            print(f"\n✗ Erro: {str(e)}\n")
        return results


def main():
    """
    Função principal com argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Processamento de lâminas de microscopia com extração de etiquetas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --image data/slide.png
  python main.py --image data/slide.png --config config.json --no-label
  python main.py --image data/slide.png --no-save --verbose
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Caminho para a imagem de entrada. Se não fornecido, usa config.json"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="data/config.json",
        help="Caminho para arquivo config.json (padrão: data/config.json)"
    )
    
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Se fornecido, não extrai informações da etiqueta (OCR)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Se fornecido, não salva imagens de saída"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Modo verbose com mais detalhes de processamento"
    )
    
    args = parser.parse_args()
    
    # Determinar caminho da imagem
    image_path = args.image
    if not image_path:
        try:
            config = load_slide_config(args.config)
            image_path = config.get("slide_img_path")
            if not image_path:
                print("✗ Erro: Caminho da imagem não fornecido e não encontrado em config.json")
                return 1
        except Exception as e:
            print(f"✗ Erro ao carregar config.json: {e}")
            return 1
    
    # Verificar se arquivo existe
    if not os.path.exists(image_path):
        print(f"✗ Erro: Arquivo não encontrado: {image_path}")
        return 1
    
    # Processar lâmina
    results = process_slide(
        image_path=image_path,
        config_path=args.config,
        extract_label=not args.no_label,
        save_output=not args.no_save,
        verbose=args.verbose or True  # Always verbose por padrão
    )
    
    # Retornar código de sucesso/falha
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
