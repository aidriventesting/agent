from typing import Any, Dict, Optional
from robot.api import logger

from src.AiHelper.agent.vlm.omniparser_client import OmniParserClient
from src.AiHelper.agent.vlm.omniparser_parser import OmniParserResultProcessor
from src.AiHelper.agent.vlm.omniparser_selector import OmniParserElementSelector


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


# Test rapide
if __name__ == "__main__":
    orchestrator = OmniParserOrchestrator()
    
    result = orchestrator.find_element(
        element_description="YouTube icon",
        image_path="tests/_data/images/screenshots/screenshot-Google Pixel 5-11.0.png",
        element_type="interactive",
    )
    
    print("=" * 80)
    if result:
        print(f"Element trouvé: {result['element_key']}")
        print(f"Content: {result['element_data']['content']}")
        print(f"Bbox: {result['element_data']['bbox']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reason: {result['reason']}")
    else:
        print("Aucun élément trouvé")

