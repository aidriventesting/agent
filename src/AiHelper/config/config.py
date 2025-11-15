import os
from dotenv import load_dotenv
from .model_config import ModelConfig

load_dotenv()


class Config:
    """
    Configuration class for AI Helper.
    Loads API keys from environment variables (.env file).
    Default models are loaded from llm_models.json (single source of truth).
    """
    
    # LLM Provider Settings
    DEFAULT_LLM_CLIENT = os.getenv("DEFAULT_LLM_CLIENT", "openai")
    
    # API Keys for LLM Providers
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

    # OmniParser (Hugging Face space)
    OMNIPARSER_SPACE = os.getenv("OMNIPARSER_SPACE", "AI-DrivenTesting/OmniParser-v2")
    OMNIPARSER_API_NAME = os.getenv("OMNIPARSER_API_NAME", "/process")
    OMNIPARSER_USE_PADDLE_OCR = os.getenv("OMNIPARSER_USE_PADDLE_OCR", "true").lower() == "true"
    OMNIPARSER_DEFAULT_BOX_THRESHOLD = float(os.getenv("OMNIPARSER_BOX_THRESHOLD", "0.25"))
    OMNIPARSER_DEFAULT_IOU_THRESHOLD = float(os.getenv("OMNIPARSER_IOU_THRESHOLD", "0.1"))
    OMNIPARSER_DEFAULT_IMAGE_SIZE = int(float(os.getenv("OMNIPARSER_IMAGE_SIZE", "640")))
    
    # Default Models per Provider
    _model_config = ModelConfig()
    DEFAULT_OPENAI_MODEL = os.getenv("DEFAULT_OPENAI_MODEL") or _model_config.get_provider_default_model("openai")
    DEFAULT_ANTHROPIC_MODEL = os.getenv("DEFAULT_ANTHROPIC_MODEL") or _model_config.get_provider_default_model("anthropic")
    DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL") or _model_config.get_provider_default_model("gemini")
    DEFAULT_DEEPSEEK_MODEL = os.getenv("DEFAULT_DEEPSEEK_MODEL") or _model_config.get_provider_default_model("deepseek")
    DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL") or _model_config.get_provider_default_model("ollama")
    
    # Ollama Configuration (local server)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    
    # Image Upload Provider API Keys
    IMGBB_API_KEY = os.getenv("IMGBB_API_KEY", "")
    FREEIMAGEHOST_API_KEY = os.getenv("FREEIMAGEHOST_API_KEY", "")

    # DEFAULT_MAX_TOKENS = 1400
    # DEFAULT_TEMPERATURE = 1.0
    # DEFAULT_TOP_P = 1.0

    @classmethod
    def get_huggingface_token(cls) -> str:
        return cls.HUGGINGFACE_API_KEY

    @classmethod
    def get_omniparser_params(
        cls,
        box_threshold: float | None = None,
        iou_threshold: float | None = None,
        use_paddleocr: bool | None = None,
        imgsz: int | None = None,
    ) -> dict[str, float | bool | int]:
        params: dict[str, float | bool | int] = {
            "box_threshold": box_threshold if box_threshold is not None else cls.OMNIPARSER_DEFAULT_BOX_THRESHOLD,
            "iou_threshold": iou_threshold if iou_threshold is not None else cls.OMNIPARSER_DEFAULT_IOU_THRESHOLD,
            "use_paddleocr": use_paddleocr if use_paddleocr is not None else cls.OMNIPARSER_USE_PADDLE_OCR,
            "imgsz": imgsz if imgsz is not None else cls.OMNIPARSER_DEFAULT_IMAGE_SIZE,
        }
        return params