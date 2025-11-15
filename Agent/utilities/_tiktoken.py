import tiktoken
import os
import json
import fcntl
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import warnings
from Agent.utilities._logger import RobotCustomLogger
from Agent.config.model_config import ModelConfig

@dataclass
class TokenStats:
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float

class TokenHelper:
    
    # File-based storage for true cross-process persistence
    _COST_FILE = "/tmp/ai_cost_tracker.json"
    
    # Class-level variables for singleton behavior
    _instance = None
    _initialized = False
    
    # Load configuration from JSON file
    _model_config = ModelConfig()
    
    # Pricing per 1000 tokens (input/output) in USD
    # Loaded from llm_models.json configuration file
    PRICING = _model_config.get_pricing_dict()

    # Maximum context tokens per model
    # Loaded from llm_models.json configuration file
    MAX_CONTEXT_TOKENS = _model_config.get_max_context_dict()
    
    def __new__(cls, model_name: str = "gpt-4o-mini"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not TokenHelper._initialized:
            self.model_name = model_name
            self.encoding = self._get_encoding_for_model()
            self.logger = RobotCustomLogger()
            TokenHelper._initialized = True
            current_cost = self.get_cumulated_cost()
            current_tokens = self.get_cumulated_tokens()
            self.logger.info(f"TokenHelper singleton initialized with model: {model_name} - Current accumulated: cost={current_cost}, tokens={current_tokens}", False)
        else:
            current_cost = self.get_cumulated_cost()
            current_tokens = self.get_cumulated_tokens()
            self.logger.info(f"TokenHelper singleton reused - Current accumulated: cost={current_cost}, tokens={current_tokens}", False)

    def _get_encoding_for_model(self) -> tiktoken.Encoding:
        try:
            if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
                return tiktoken.get_encoding("cl100k_base")
            return tiktoken.get_encoding("p50k_base")
        except KeyError:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def _count_batch_tokens(self, texts: List[str]) -> List[int]:
        return [self.count_tokens(text) for text in texts]
    
    
    #################
    def estimate_tokens_and_cost(
        self,
        prompt: str,
        completion: str,
        model: str = None
    ) -> TokenStats:
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(completion)
        total_tokens = prompt_tokens + completion_tokens
        cost_dict = self.calculate_cost(prompt_tokens, completion_tokens, model)
        
        return TokenStats(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=cost_dict["total_cost"]
        )
    
    def _split_text(
        self,
        text: str,
        chunk_size: int = 2048,
        overlap: int = 100
    ) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(self.encoding.decode(chunk))
            
        return chunks
    #################

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
        from_end: bool = False
    ) -> str:
        tokens = self.encoding.encode(text)
        truncated = tokens[-max_tokens:] if from_end else tokens[:max_tokens]
        return self.encoding.decode(truncated)
    
    def _get_max_context_tokens(
        self,
        model: str = None,
        max_tokens: int = None
    ) -> int:
        model = model or self.model_name
        max_context = self.MAX_CONTEXT_TOKENS[model]  # y aura erreur si le model n'est pas dans le dictionnaire -> ajouter un default ? 
        self.logger.info(f"Max context tokens for {model}: {max_context}")
        if max_tokens is not None:
            return min(max_tokens, max_context)
        return max_context


####
    def ensure_token_limit(
        self,
        text: str,
        model: str = None,
        max_tokens: int = None
    ) -> str:
        """ tronque le texte si il dépasse le max_tokens du modele """
        effective_max = self._get_max_context_tokens(model, max_tokens)
        token_count = self._count_tokens(text)
        
        if token_count <= effective_max:
            self.logger.info(f"Text is within token limit: {token_count} <= {effective_max}", True)
            return text
        else:
            self.logger.warning(f"Text is NOT within token limit: {token_count} <= {effective_max}", True)
            return self._truncate_text(
                text, 
                max_tokens=effective_max,
                from_end=True
            )

###
    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = None
    ) -> Dict[str, float]:
        model = model or self.model_name
        if model not in self.PRICING:
            self.logger.warning(f"Pricing not available for {model}, using GPT-4o default")
            self.logger.info(f"Existing models: {self.PRICING.keys()}")
            model = "gpt-4o-mini"
            
        input_cost = round((prompt_tokens / 1000) * self.PRICING[model]["input"], 5)
        output_cost = round((completion_tokens / 1000) * self.PRICING[model]["output"], 5)
        total_cost = round(input_cost + output_cost, 5)

        # Debug logging - use file storage for true cross-process persistence
        old_data = self._load_costs()
        old_cumulated_cost = old_data['cost']
        old_cumulated_tokens = old_data['tokens']
        
        new_cumulated_cost = old_cumulated_cost + total_cost
        new_cumulated_tokens = old_cumulated_tokens + prompt_tokens + completion_tokens
        
        # Save to file with lock
        self._save_costs(new_cumulated_cost, new_cumulated_tokens)

        self.logger.info(f"Cost calculation: {old_cumulated_cost} + {total_cost} = {new_cumulated_cost}", False)
        self.logger.info(f"Token calculation: {old_cumulated_tokens} + {prompt_tokens + completion_tokens} = {new_cumulated_tokens}", False)

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def _load_costs(self) -> Dict[str, float]:
        """Load costs from file with lock"""
        try:
            with open(self._COST_FILE, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return {"cost": data.get("cost", 0.0), "tokens": data.get("tokens", 0)}
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return {"cost": 0.0, "tokens": 0}
    
    def _save_costs(self, cost: float, tokens: int):
        """Save costs to file with lock"""
        try:
            with open(self._COST_FILE, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                json.dump({"cost": cost, "tokens": tokens}, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            self.logger.warning(f"Failed to save costs to file: {e}", False)
    
    def get_cumulated_cost(self) -> float:
        """Get cumulated cost from file"""
        return self._load_costs()['cost']
    
    def get_cumulated_tokens(self) -> int:
        """Get cumulated tokens from file"""
        return self._load_costs()['tokens']
    
    def reset_accumulation(self):
        """Reset cumulated cost and tokens to zero"""
        old_data = self._load_costs()
        old_cost = old_data['cost']
        old_tokens = old_data['tokens']
        self._save_costs(0.0, 0)
        self.logger.info(f"Reset accumulation: cost {old_cost} → 0, tokens {old_tokens} → 0", False)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary"""
        data = self._load_costs()
        return {
            "cumulated_cost": data['cost'],
            "cumulated_tokens": data['tokens'],
            "model_name": getattr(self, 'model_name', 'unknown'),
            "instance_id": id(self),
            "class_instance_id": id(TokenHelper._instance),
            "storage_file": self._COST_FILE,
            "file_exists": os.path.exists(self._COST_FILE)
        }