"""
Production AI Inference Engine
Provides llama.cpp and BERT model integration for autonomous agents
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class AIConfig:
    """Configuration for AI inference engines"""
    # LLaMA configuration
    llama_model_path: str = "./models/llama-2-7b-chat.q4_0.gguf"
    llama_context_length: int = 2048
    llama_threads: int = 4
    llama_temperature: float = 0.7
    llama_top_p: float = 0.95
    
    # BERT configuration  
    bert_model_name: str = "sentence-transformers/allenai-specter"
    bert_device: str = "cpu"
    bert_cache_dir: Optional[str] = None
    bert_max_length: int = 512
    
    # General configuration
    fallback_enabled: bool = True
    fallback_model_path: Optional[str] = None
    force_production_models: bool = False
    model_cache_dir: str = "/opt/ai-models/cache"

class ProductionAIEngine:
    """
    Production-grade AI inference engine supporting multiple backends
    Prioritizes real inference over mock implementations
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.llama_engine = None
        self.bert_engine = None
        self.is_initialized = False
        
        # Override config from environment variables
        self._load_env_config()
        
        # Production quality check
        if os.getenv('ENVIRONMENT', '').lower() == 'production':
            if not self._validate_production_requirements():
                raise ValueError(
                    "PRODUCTION VIOLATION: AI models not properly configured. "
                    "NEVER SACRIFICE QUALITY!! Configure production models."
                )
    
    def _load_env_config(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'AI_MODEL_PATH': 'llama_model_path',
            'BERT_MODEL_NAME': 'bert_model_name', 
            'AI_CONTEXT_LENGTH': 'llama_context_length',
            'AI_DEVICE': 'bert_device',
            'AI_THREADS': 'llama_threads',
            'MODEL_CACHE_DIR': 'model_cache_dir',
            'AI_FORCE_PRODUCTION': 'force_production_models'
        }
        
        for env_var, config_attr in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_attr in ['llama_context_length', 'llama_threads']:
                    value = int(value)
                elif config_attr in ['llama_temperature', 'llama_top_p']:
                    value = float(value)
                elif config_attr == 'force_production_models':
                    value = value.lower() in ('true', '1', 'yes')
                
                setattr(self.config, config_attr, value)
                logger.info(f"AI Config loaded from env: {config_attr} = {value}")
    
    def _validate_production_requirements(self) -> bool:
        """Validate that production AI requirements are met"""
        requirements = [
            (os.path.exists(self.config.llama_model_path), f"LLaMA model not found: {self.config.llama_model_path}"),
            (self.config.bert_model_name, "BERT model name not configured"),
            (self.config.llama_context_length > 0, "Context length must be positive"),
            (self.config.llama_threads > 0, "Thread count must be positive")
        ]
        
        for requirement, error_msg in requirements:
            if not requirement:
                logger.error(f"Production validation failed: {error_msg}")
                return False
        
        return True
    
    async def initialize(self):
        """Initialize AI inference engines"""
        if self.is_initialized:
            return
        
        logger.info("Initializing production AI engines...")
        
        try:
            await self._initialize_llama_engine()
            await self._initialize_bert_engine()
            self.is_initialized = True
            logger.info("✅ Production AI engines initialized successfully")
            
        except Exception as e:
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: AI engine initialization failed: {e}")
            
            if self.config.fallback_enabled:
                logger.warning(f"AI engine initialization failed, attempting fallback: {e}")
                await self._initialize_fallback_engines()
                self.is_initialized = True
            else:
                raise
    
    async def _initialize_llama_engine(self):
        """Initialize LLaMA inference engine"""
        try:
            from llama_cpp import Llama
            
            # Validate model file exists
            if not os.path.exists(self.config.llama_model_path):
                if self.config.force_production_models:
                    raise FileNotFoundError(f"LLaMA model required but not found: {self.config.llama_model_path}")
                
                logger.warning(f"LLaMA model not found: {self.config.llama_model_path}")
                return
            
            logger.info(f"Loading LLaMA model: {self.config.llama_model_path}")
            
            self.llama_engine = Llama(
                model_path=self.config.llama_model_path,
                n_ctx=self.config.llama_context_length,
                n_threads=self.config.llama_threads,
                verbose=False
            )
            
            # Test inference
            test_output = self.llama_engine(
                "Test prompt",
                max_tokens=10,
                temperature=0.1,
                echo=False
            )
            
            if test_output and 'choices' in test_output:
                logger.info("✅ LLaMA engine test inference successful")
            else:
                raise ValueError("LLaMA engine test inference failed")
                
        except ImportError:
            if self.config.force_production_models:
                raise ValueError("PRODUCTION VIOLATION: llama-cpp-python not installed but required")
            logger.warning("llama-cpp-python not available, skipping LLaMA initialization")
        except Exception as e:
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: LLaMA initialization failed: {e}")
            logger.error(f"LLaMA initialization failed: {e}")
    
    async def _initialize_bert_engine(self):
        """Initialize BERT inference engine"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading BERT model: {self.config.bert_model_name}")
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                self.config.bert_model_name,
                cache_dir=self.config.bert_cache_dir
            )
            
            self.bert_model = AutoModel.from_pretrained(
                self.config.bert_model_name,
                cache_dir=self.config.bert_cache_dir
            ).to(self.config.bert_device)
            
            # Test inference
            test_inputs = self.bert_tokenizer(
                "Test text", 
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.config.bert_device)
            
            with torch.no_grad():
                outputs = self.bert_model(**test_inputs)
                if outputs.last_hidden_state is not None:
                    logger.info("✅ BERT engine test inference successful")
                else:
                    raise ValueError("BERT engine test inference failed")
                    
        except ImportError as e:
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: Required AI libraries not installed: {e}")
            logger.warning(f"BERT dependencies not available: {e}")
        except Exception as e:
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: BERT initialization failed: {e}")
            logger.error(f"BERT initialization failed: {e}")
    
    async def _initialize_fallback_engines(self):
        """Initialize minimal fallback AI engines"""
        logger.warning("Initializing fallback AI engines - performance will be degraded")
        
        # Even fallbacks must be real AI, not mocks
        if self.config.fallback_model_path and os.path.exists(self.config.fallback_model_path):
            try:
                from llama_cpp import Llama
                self.llama_engine = Llama(
                    model_path=self.config.fallback_model_path,
                    n_ctx=1024,  # Smaller context for fallback
                    n_threads=2,
                    verbose=False
                )
                logger.info("✅ Fallback LLaMA engine initialized")
            except Exception as e:
                logger.error(f"Fallback LLaMA initialization failed: {e}")
        
        # For BERT, try a smaller model
        try:
            from transformers import AutoTokenizer, AutoModel
            fallback_bert = "distilbert-base-uncased"
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained(fallback_bert)
            self.bert_model = AutoModel.from_pretrained(fallback_bert).to('cpu')
            logger.info("✅ Fallback BERT engine initialized")
            
        except Exception as e:
            logger.error(f"Fallback BERT initialization failed: {e}")
    
    async def generate_text(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using LLaMA inference engine"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.llama_engine is None:
            raise ValueError("PRODUCTION VIOLATION: No LLaMA engine available for text generation")
        
        try:
            output = self.llama_engine(
                prompt,
                max_tokens=max_tokens,
                temperature=self.config.llama_temperature,
                top_p=self.config.llama_top_p,
                echo=False
            )
            
            if output and 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['text'].strip()
            else:
                raise ValueError("LLaMA inference returned empty result")
                
        except Exception as e:
            logger.error(f"LLaMA text generation failed: {e}")
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: Text generation failed: {e}")
            raise
    
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using BERT inference engine"""
        if not self.is_initialized:
            await self.initialize()
        
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("PRODUCTION VIOLATION: No BERT engine available for text classification")
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.bert_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.bert_max_length,
                return_tensors='pt'
            ).to(self.config.bert_device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                text_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Calculate similarities with categories
            category_scores = {}
            for category in categories:
                # This is a simplified approach - in production, you'd use trained classifiers
                category_inputs = self.bert_tokenizer(
                    category,
                    truncation=True,
                    padding=True,
                    max_length=self.config.bert_max_length,
                    return_tensors='pt'
                ).to(self.config.bert_device)
                
                with torch.no_grad():
                    category_outputs = self.bert_model(**category_inputs)
                    category_embedding = category_outputs.last_hidden_state.mean(dim=1)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(text_embedding, category_embedding, dim=1)
                category_scores[category] = float(similarity.item())
            
            # Normalize scores
            total_score = sum(category_scores.values())
            if total_score > 0:
                category_scores = {k: v / total_score for k, v in category_scores.items()}
            
            return category_scores
            
        except Exception as e:
            logger.error(f"BERT text classification failed: {e}")
            if self.config.force_production_models:
                raise ValueError(f"PRODUCTION VIOLATION: Text classification failed: {e}")
            raise
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get status of AI engines for monitoring"""
        return {
            'initialized': self.is_initialized,
            'llama_available': self.llama_engine is not None,
            'bert_available': self.bert_model is not None,
            'config': {
                'llama_model_path': self.config.llama_model_path,
                'bert_model_name': self.config.bert_model_name,
                'context_length': self.config.llama_context_length,
                'device': self.config.bert_device,
                'force_production': self.config.force_production_models
            }
        }

# Global AI engine instance
_ai_engine_instance = None

async def get_production_ai_engine() -> ProductionAIEngine:
    """Get global production AI engine instance"""
    global _ai_engine_instance
    
    if _ai_engine_instance is None:
        _ai_engine_instance = ProductionAIEngine()
        await _ai_engine_instance.initialize()
    
    return _ai_engine_instance

async def get_fallback_ai_engine() -> ProductionAIEngine:
    """Get fallback AI engine with minimal configuration"""
    config = AIConfig(
        llama_context_length=1024,
        llama_threads=2,
        bert_device='cpu',
        fallback_enabled=True,
        force_production_models=False
    )
    
    engine = ProductionAIEngine(config)
    await engine.initialize()
    return engine