"""Model loading and optimization utilities."""

import os
import platform
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification
from transformers.models.modernbert import ModernBertForSequenceClassification  
from transformers.models.t5 import T5ForSequenceClassification
import psutil

logger = logging.getLogger(__name__)

class ModelLoader:
    """Efficient model loader with optimizations for different architectures."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "encoder_server")
        self.device = self._get_optimal_device()
        self.dtype = self._get_optimal_dtype()
        
    def _get_optimal_device(self) -> torch.device:
        """Determine the best device for inference."""
        if torch.cuda.is_available():
            # Use the first GPU with most memory
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_memory.append((i, props.total_memory))
            best_gpu = max(gpu_memory, key=lambda x: x[1])[0]
            return torch.device(f"cuda:{best_gpu}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine the best dtype for inference."""
        if self.device.type == "cuda":
            # Use bfloat16 if supported, otherwise float16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        elif self.device.type == "mps":
            # MPS supports float16
            return torch.float16
        else:
            # CPU - use float32 for stability
            return torch.float32
    
    def load_model(
        self, 
        model_name: str,
        use_flash_attention: bool = True,
        compile_model: bool = True,
        use_better_transformer: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and optimize a model for inference.
        
        Args:
            model_name: HuggingFace model name or local path
            use_flash_attention: Whether to use flash attention (if available)
            compile_model: Whether to compile the model with torch.compile
            use_better_transformer: Whether to use BetterTransformer optimization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model {model_name} on {self.device} with dtype {self.dtype}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Determine model task type from config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
            is_token_classification = hasattr(config, 'id2label') and getattr(config, 'problem_type', None) == 'single_label_classification'
            
            # Check if it's actually token classification from the pipeline_tag in model card
            model_info = None
            try:
                from huggingface_hub import model_info as hf_model_info
                model_info = hf_model_info(model_name)
                if hasattr(model_info, 'pipeline_tag') and model_info.pipeline_tag == 'token-classification':
                    is_token_classification = True
            except:
                pass
                
        except:
            is_token_classification = False
        
        # Load model with appropriate class
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": self.dtype,
            "trust_remote_code": True,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        
        # Add flash attention config if available and requested
        if use_flash_attention and self._supports_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            if is_token_classification:
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name, **model_kwargs
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **model_kwargs
                )
        except Exception as e:
            logger.warning(f"Failed to load with flash attention: {e}")
            # Fallback without flash attention
            model_kwargs.pop("attn_implementation", None)
            if is_token_classification:
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name, **model_kwargs
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **model_kwargs
                )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            model = model.to(self.device)
        
        # Set to eval mode
        model.eval()
        
        # Apply optimizations
        if use_better_transformer and hasattr(model, 'to_bettertransformer'):
            try:
                model = model.to_bettertransformer()
                logger.info("Applied BetterTransformer optimization")
            except Exception as e:
                logger.warning(f"Failed to apply BetterTransformer: {e}")
        
        # Compile model if requested and supported
        if compile_model and hasattr(torch, 'compile'):
            try:
                # Use different compile modes based on device
                if self.device.type == "cuda":
                    model = torch.compile(model, mode="max-autotune")
                else:
                    model = torch.compile(model, mode="default")
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Memory optimization
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
        
        logger.info(f"Model loaded successfully. Memory usage: {self._get_memory_usage()}")
        
        return model, tokenizer
    
    def _supports_flash_attention(self) -> bool:
        """Check if flash attention is available."""
        if self.device.type != "cuda":
            return False
        
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _get_memory_usage(self) -> str:
        """Get current memory usage."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
        else:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            return f"RAM: {memory_mb:.2f}MB"
    
    def optimize_for_inference(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply additional inference optimizations."""
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Fuse operations where possible
        if hasattr(model, 'fuse'):
            try:
                model.fuse()
            except:
                pass
                
        return model