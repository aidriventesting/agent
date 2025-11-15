from typing import Any, Dict, Optional
from robot.api import logger

from Agent.ai.llm.facade import UnifiedLLMFacade


class OmniParserElementSelector:
    """
    Sélectionne un élément GUI en utilisant ChatGPT.
    
    Prend un dictionnaire d'éléments et une description, 
    puis demande à l'IA de trouver l'élément correspondant.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini") -> None:
        """
        Initialise le sélecteur avec le modèle d'IA.
        
        Args:
            provider: Le fournisseur d'IA (openai, anthropic, etc.)
            model: Le modèle à utiliser
        """
        self.llm = UnifiedLLMFacade(provider=provider, model=model)
        logger.info(f"OmniParserElementSelector initialisé avec {provider}/{model}")

    def select_element(
        self,
        elements_data: Dict[str, Dict[str, Any]],
        element_description: str,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Sélectionne l'élément GUI qui correspond à la description.
        
        Args:
            elements_data: Dictionnaire des éléments (ex: {'icon3': {'type': 'icon', ...}})
            element_description: Description de l'élément à trouver (ex: "YouTube app")
            temperature: Température pour la génération (0.0 = déterministe)
            
        Returns:
            Un dictionnaire avec:
            - element_key: La clé de l'élément trouvé (ex: 'icon3')
            - element_data: Les données de l'élément
            - confidence: Niveau de confiance (optionnel)
            - reason: Raison du choix (optionnel)
            
            Retourne None si aucun élément n'est trouvé.
        """
        logger.info(f"Recherche de l'élément: '{element_description}'")
        logger.debug(f"Nombre d'éléments à analyser: {len(elements_data)}")

        # Construire le prompt
        messages = self._build_prompt(elements_data, element_description)
        
        # Envoyer à l'IA
        try:
            response = self.llm.send_ai_request_and_return_response(
                messages=messages,
                temperature=temperature
            )
            
            # Parser la réponse
            result = self._parse_response(response, elements_data)
            
            if result:
                logger.info(f"✅ Élément trouvé: {result.get('element_key')}")
            else:
                logger.warn("❌ Aucun élément correspondant trouvé")
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la sélection: {str(e)}")
            return None

    def _build_prompt(
        self,
        elements_data: Dict[str, Dict[str, Any]],
        element_description: str,
    ) -> list:
        """
        Construit le prompt pour l'IA.
        
        Args:
            elements_data: Les éléments UI disponibles
            element_description: La description de l'élément recherché
            
        Returns:
            Liste de messages pour l'IA
        """
        # Formater les éléments de manière lisible
        elements_text = self._format_elements(elements_data)
        
        system_prompt = """Tu es un assistant spécialisé dans la sélection d'éléments GUI.
Ta tâche est de trouver l'élément qui correspond le mieux à la description donnée.

Analyse les éléments disponibles et retourne celui qui correspond le mieux.
Si aucun élément ne correspond, indique 'element_key': null.

Réponds UNIQUEMENT en JSON avec cette structure:
{
    "element_key": "la clé de l'élément (ex: icon3) ou null",
    "confidence": "high, medium ou low",
    "reason": "explication brève de ton choix"
}"""

        user_prompt = f"""Éléments disponibles:
{elements_text}

Description de l'élément recherché: "{element_description}"

Trouve l'élément qui correspond le mieux à cette description."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_elements(self, elements_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Formate les éléments pour le prompt.
        
        Args:
            elements_data: Les éléments à formater
            
        Returns:
            String formaté des éléments
        """
        lines = []
        for key, data in elements_data.items():
            content = data.get("content", "")
            element_type = data.get("type", "unknown")
            interactive = data.get("interactivity", False)
            
            lines.append(
                f"- {key}: type={element_type}, content='{content}', "
                f"interactive={interactive}"
            )
        
        return "\n".join(lines)

    def _parse_response(
        self,
        response: Dict[str, Any],
        elements_data: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Parse la réponse de l'IA.
        
        Args:
            response: La réponse JSON de l'IA
            elements_data: Les éléments originaux
            
        Returns:
            Dictionnaire avec l'élément trouvé ou None
        """
        element_key = response.get("element_key")
        
        # Si aucun élément trouvé
        if not element_key or element_key == "null":
            return None
        
        # Vérifier que l'élément existe
        if element_key not in elements_data:
            logger.warn(f"L'élément '{element_key}' retourné n'existe pas dans les données")
            return None
        
        # Construire le résultat
        return {
            "element_key": element_key,
            "element_data": elements_data[element_key],
            "confidence": response.get("confidence", "unknown"),
            "reason": response.get("reason", ""),
        }


# Test rapide
if __name__ == "__main__":
    # Exemple de données
    test_data = {
        'icon3': {
            'type': 'icon',
            'bbox': [0.41938668489456177, 0.17028668522834778, 0.5745916366577148, 0.2660691440105438],
            'interactivity': True,
            'content': 'YouTube '
        },
        'icon9': {
            'type': 'icon',
            'bbox': [0.23282678425312042, 0.17132169008255005, 0.38811373710632324, 0.26554766297340393],
            'interactivity': True,
            'content': 'Gmail '
        },
        'icon22': {
            'type': 'icon',
            'bbox': [0.05158957466483116, 0.639639139175415, 0.2134605348110199, 0.7337194681167603],
            'interactivity': True,
            'content': 'Chrome '
        }
    }
    
    selector = OmniParserElementSelector()
    result = selector.select_element(test_data, "YouTube")
    
    print("=" * 80)
    print(f"Résultat: {result}")

