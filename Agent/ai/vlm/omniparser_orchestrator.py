from typing import Any, Dict, List, Optional, Tuple
from robot.api import logger
from PIL import Image

from Agent.ai.vlm.omniparser_client import OmniParserClient
from Agent.ai.vlm.omniparser_parser import OmniParserResultProcessor
from Agent.ai.vlm.omniparser_selector import OmniParserElementSelector


class OmniParserOrchestrator:
    """
    Orchestrateur principal pour la sélection d'éléments GUI via OmniParser + LLM.
    
    Cette classe coordonne:
    1. OmniParserClient - Analyse l'image via Hugging Face
    2. OmniParserResultProcessor - Parse et filtre les éléments
    3. OmniParserElementSelector - Sélectionne l'élément via LLM
    """

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        omniparser_space_id: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        """
        Initialise l'orchestrateur avec tous les composants nécessaires.
        
        Args:
            llm_provider: Fournisseur LLM (openai, anthropic, etc.)
            llm_model: Modèle à utiliser
            omniparser_space_id: ID de l'espace Hugging Face OmniParser (optionnel)
            hf_token: Token Hugging Face (optionnel)
        """
        self.client = OmniParserClient(
            space_id=omniparser_space_id,
            hf_token=hf_token
        )
        self.selector = OmniParserElementSelector(
            provider=llm_provider,
            model=llm_model
        )
        logger.info("OmniParserOrchestrator initialisé avec succès")

    def find_element(
        self,
        element_description: str,
        *,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_name: Optional[str] = None,
        element_type: str = "interactive",
        box_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        use_paddleocr: Optional[bool] = None,
        imgsz: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Trouve l'élément GUI qui correspond à la description.
        
        Workflow complet:
        1. Envoie l'image à OmniParser pour détecter les éléments
        2. Parse et filtre les éléments par type
        3. Utilise le LLM pour sélectionner l'élément correspondant
        
        Args:
            element_description: Description de l'élément recherché (ex: "YouTube app")
            image_path: Chemin local de l'image
            image_url: URL de l'image
            image_base64: Image encodée en base64
            image_name: Nom de l'image (pour inférer l'extension si base64)
            element_type: Type d'éléments à filtrer:
                - "interactive" (défaut): éléments interactifs uniquement
                - "icon": icônes uniquement
                - "text": textes uniquement
                - "all": tous les éléments
            box_threshold: Seuil de détection OmniParser
            iou_threshold: Seuil IOU OmniParser
            use_paddleocr: Utiliser PaddleOCR
            imgsz: Taille de l'image pour OmniParser
            temperature: Température pour le LLM
            
        Returns:
            Dictionnaire avec:
            - element_key: Clé de l'élément (ex: 'icon3')
            - element_data: Données complètes de l'élément (bbox, content, etc.)
            - confidence: Niveau de confiance du LLM
            - reason: Raison du choix
            - image_temp_path: Chemin de l'image temporaire annotée (optionnel)
            
            Retourne None si aucun élément n'est trouvé.
        """
        logger.info(f"🔍 Recherche de l'élément: '{element_description}'")
        
        # Étape 1: Analyse de l'image avec OmniParser
        logger.info("📸 Étape 1/3: Analyse de l'image avec OmniParser...")
        image_temp_path, parsed_text = self.client.parse_image(
            image_path=image_path,
            image_url=image_url,
            image_base64=image_base64,
            image_name=image_name,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        
        if not parsed_text:
            logger.error("❌ OmniParser n'a détecté aucun élément")
            return None
        
        # Étape 2: Parse et filtre les éléments par type
        logger.info(f"🔧 Étape 2/3: Parsing et filtrage des éléments (type={element_type})...")
        processor = OmniParserResultProcessor(
            response_text=parsed_text,
            image_temp_path=image_temp_path,
        )
        elements_data = processor.get_parsed_ui_elements(element_type=element_type)
        
        if not elements_data:
            logger.error(f"❌ Aucun élément de type '{element_type}' trouvé")
            return None
         
        logger.info(f"✓ {len(elements_data)} éléments filtrés")
        
        # Étape 3: Sélection de l'élément via LLM
        logger.info("🤖 Étape 3/3: Sélection de l'élément via LLM...")
        result = self.selector.select_element(
            elements_data=elements_data,
            element_description=element_description,
            temperature=temperature,
        )
        
        if not result:
            logger.error("❌ Le LLM n'a trouvé aucun élément correspondant")
            return None
        
        # Ajouter l'image temporaire au résultat
        result["image_temp_path"] = image_temp_path
        
        logger.info(
            f"✅ Élément trouvé: {result['element_key']} "
            f"(confidence={result.get('confidence', 'unknown')})"
        )
        
        return result

    @staticmethod
    def bbox_to_pixels(
        bbox_normalized: List[float],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        """
        Convertit les coordonnées bbox normalisées en coordonnées pixels réelles.
        
        Args:
            bbox_normalized: Liste [x1, y1, x2, y2] avec valeurs entre 0 et 1
            image_width: Largeur de l'image en pixels
            image_height: Hauteur de l'image en pixels
            
        Returns:
            Tuple (x1, y1, x2, y2) en coordonnées pixels entières
            
        Example:
            >>> bbox = [0.419, 0.170, 0.574, 0.266]
            >>> pixels = OmniParserOrchestrator.bbox_to_pixels(bbox, 1080, 1920)
            >>> print(pixels)  # (452, 326, 620, 510)
        """
        if len(bbox_normalized) != 4:
            raise ValueError(f"bbox doit contenir 4 valeurs, reçu {len(bbox_normalized)}")
        
        x1_norm, y1_norm, x2_norm, y2_norm = bbox_normalized
        
        # Convertir en pixels
        x1 = int(x1_norm * image_width)
        y1 = int(y1_norm * image_height)
        x2 = int(x2_norm * image_width)
        y2 = int(y2_norm * image_height)
        
        logger.debug(
            f"Conversion bbox: [{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}] "
            f"-> [{x1}, {y1}, {x2}, {y2}] (image: {image_width}x{image_height})"
        )
        
        return (x1, y1, x2, y2)

    @staticmethod
    def bbox_to_pixels_from_image(
        bbox_normalized: List[float],
        image_path: str,
    ) -> Tuple[int, int, int, int]:
        """
        Convertit les bbox normalisées en pixels en lisant automatiquement les dimensions.
        
        Args:
            bbox_normalized: Liste [x1, y1, x2, y2] avec valeurs entre 0 et 1
            image_path: Chemin vers l'image pour obtenir les dimensions
            
        Returns:
            Tuple (x1, y1, x2, y2) en coordonnées pixels entières
            
        Example:
            >>> bbox = [0.419, 0.170, 0.574, 0.266]
            >>> pixels = OmniParserOrchestrator.bbox_to_pixels_from_image(
            ...     bbox, "screenshot.png"
            ... )
            >>> print(pixels)  # (452, 326, 620, 510)
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            logger.debug(f"Dimensions de l'image '{image_path}': {width}x{height}")
        except Exception as e:
            logger.error(f"Impossible d'ouvrir l'image '{image_path}': {e}")
            raise
        
        return OmniParserOrchestrator.bbox_to_pixels(
            bbox_normalized=bbox_normalized,
            image_width=width,
            image_height=height,
        )


# Test rapide
if __name__ == "__main__":
    from PIL import ImageDraw
    
    orchestrator = OmniParserOrchestrator()
    
    image_path = "tests/_data/images/screenshots/screenshot-Google Pixel 5-11.0.png"
    result = orchestrator.find_element(
        element_description="YouTube icon",
        image_path=image_path,
        element_type="interactive",
    )
    
    print("=" * 80)
    if result:
        print(f"Element trouvé: {result['element_key']}")
        print(f"Content: {result['element_data']['content']}")
        print(f"Bbox normalisée: {result['element_data']['bbox']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reason: {result['reason']}")
        
        # Conversion en coordonnées pixels
        print("\n" + "=" * 80)
        print("Conversion en coordonnées pixels:")
        
        bbox_normalized = result['element_data']['bbox']
        bbox_pixels = orchestrator.bbox_to_pixels_from_image(bbox_normalized, image_path)
        print(f"Bbox pixels: {bbox_pixels}")
        
        # Calculer le centre
        x1, y1, x2, y2 = bbox_pixels
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        print(f"Centre: ({center_x}, {center_y})")
        
        # Dessiner un point sur l'image
        print("\n" + "=" * 80)
        print("Création de l'image avec marqueur...")
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Dessiner un rectangle autour de l'élément
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Dessiner un point au centre
        point_radius = 10
        draw.ellipse(
            [center_x - point_radius, center_y - point_radius,
             center_x + point_radius, center_y + point_radius],
            fill="red",
            outline="white",
            width=2
        )
        
        # Sauvegarder l'image
        output_path = "test_output_with_marker.png"
        img.save(output_path)
        print(f"✅ Image sauvegardée: {output_path}")
        print(f"   - Rectangle rouge autour de l'élément")
        print(f"   - Point rouge au centre ({center_x}, {center_y})")
        
    else:
        print("Aucun élément trouvé")

