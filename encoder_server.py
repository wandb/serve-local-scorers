#!/usr/bin/env python3
"""
High-performance encoder model serving script for HuggingFace text classification models.
Supports batching, cross-platform deployment (macOS/Linux), and performance optimizations.
"""

import asyncio
import json
import logging
import os
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse
import platform
import sys
from concurrent.futures import ThreadPoolExecutor

# Import our Weave scorer manager
from download_models import WeaveScorerManager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    Pipeline,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for the encoder server."""
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    batch_timeout: float = 0.1  # seconds
    max_length: int = 512
    device: str = "auto"
    num_workers: int = 4
    enable_optimization: bool = True
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    # Weave scorer selection
    weave_scorer: Optional[str] = None  # e.g., "toxicity", "bias", etc.
    models_dir: str = "models"


class ClassificationRequest(BaseModel):
    """Request model for classification."""
    text: Union[str, List[str]]
    return_all_scores: bool = False


class TokenCountRequest(BaseModel):
    """Request model for token counting."""
    text: str


class TokenCountResponse(BaseModel):
    """Response model for token counting."""
    text_preview: str
    token_count: int
    max_length: int
    is_valid: bool
    character_count: int


class ClassificationResponse(BaseModel):
    """Response model for classification."""
    predictions: List[List[Dict[str, Any]]]
    processing_time: float
    batch_size: int


class BatchedDataset(Dataset):
    """Dataset wrapper for batched inference."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class EncoderModelServer:
    """High-performance encoder model server with batching support."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.batch_queue = asyncio.Queue()
        self.response_futures = {}
        self.batch_processor_task = None
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        # Initialize Weave scorer manager
        self.weave_manager = WeaveScorerManager(config.models_dir)
        
        # Resolve model path if using Weave scorer
        if config.weave_scorer:
            logger.info(f"Using Weave scorer: {config.weave_scorer}")
            try:
                self.config.model_path = self.weave_manager.get_model_path(config.weave_scorer)
                logger.info(f"Resolved Weave scorer path: {self.config.model_path}")
            except FileNotFoundError as e:
                logger.error(f"Weave scorer not found: {e}")
                logger.info("Available downloaded scorers:")
                downloaded = self.weave_manager.list_downloaded_scorers()
                if downloaded:
                    for scorer in downloaded:
                        info = self.weave_manager.get_scorer_info(scorer)
                        logger.info(f"  - {scorer}: {info.description}")
                else:
                    logger.info("  No scorers downloaded. Use 'python download_models.py download <scorer>' to download.")
                raise
        
        logger.info(f"Initializing server on device: {self.device}")
        
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device != "auto":
            return self.config.device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def load_model(self):
        """Load the model and tokenizer with optimizations."""
        logger.info(f"Loading model from: {self.config.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Apply optimizations
            if self.config.enable_optimization:
                self._apply_optimizations()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                return_all_scores=True,
                truncation=False,  # No silent truncation - raise errors instead
                max_length=self.config.max_length,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply various optimizations to the model."""
        try:
            # Compile model for PyTorch 2.0+
            if self.config.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model)
            
            # Enable optimized attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("Using optimized attention")
                
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    async def start_batch_processor(self):
        """Start the batch processing task."""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process batches of requests."""
        while True:
            try:
                batch_items = []
                
                # Collect items for batch
                try:
                    # Wait for first item
                    first_item = await asyncio.wait_for(
                        self.batch_queue.get(), 
                        timeout=self.config.batch_timeout
                    )
                    batch_items.append(first_item)
                    
                    # Collect additional items up to max_batch_size
                    while len(batch_items) < self.config.max_batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.batch_queue.get(), 
                                timeout=0.01  # Very short timeout for additional items
                            )
                            batch_items.append(item)
                        except asyncio.TimeoutError:
                            break
                            
                except asyncio.TimeoutError:
                    continue  # No items in queue, continue waiting
                
                # Process batch
                if batch_items:
                    await self._process_batch(batch_items)
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Handle any pending futures
                for request_id, texts, future in batch_items:
                    if not future.done():
                        future.set_exception(e)
    
    async def _process_batch(self, batch_items: List[tuple]):
        """Process a batch of requests."""
        try:
            # Extract texts and futures
            all_texts = []
            futures_map = {}
            
            for request_id, texts, future in batch_items:
                start_idx = len(all_texts)
                all_texts.extend(texts)
                end_idx = len(all_texts)
                futures_map[request_id] = (future, start_idx, end_idx, len(texts))
            
            # Run inference
            start_time = time.time()
            loop = asyncio.get_event_loop()
            
            # Run inference in thread pool to avoid blocking
            results = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                all_texts
            )
            
            processing_time = time.time() - start_time
            
            # Distribute results to futures
            for request_id, (future, start_idx, end_idx, batch_size) in futures_map.items():
                if not future.done():
                    request_results = results[start_idx:end_idx]
                    response = ClassificationResponse(
                        predictions=request_results,
                        processing_time=processing_time,
                        batch_size=batch_size
                    )
                    future.set_result(response)
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all futures
            for request_id, texts, future in batch_items:
                if not future.done():
                    future.set_exception(e)
    
    def _run_inference(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run inference on a batch of texts."""
        try:
            # Check input lengths BEFORE processing to provide clear errors
            oversized_inputs = []
            for i, text in enumerate(texts):
                # Tokenize without truncation to get actual length
                tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
                if len(tokens) > self.config.max_length:
                    oversized_inputs.append({
                        'index': i,
                        'token_count': len(tokens),
                        'text_preview': text[:100] + "..." if len(text) > 100 else text
                    })
            
            # Raise detailed error if any inputs are too long
            if oversized_inputs:
                error_details = []
                for item in oversized_inputs:
                    error_details.append(
                        f"  Input {item['index']}: {item['token_count']} tokens "
                        f"(exceeds limit of {self.config.max_length})\n"
                        f"    Preview: '{item['text_preview']}'"
                    )
                
                raise ValueError(
                    f"Input validation failed: {len(oversized_inputs)} text(s) exceed maximum length.\n"
                    f"Maximum allowed: {self.config.max_length} tokens\n"
                    f"Oversized inputs:\n" + "\n".join(error_details) + "\n\n"
                    f"Solutions:\n"
                    f"  1. Truncate your inputs to {self.config.max_length} tokens\n"
                    f"  2. Increase server max_length with --max-length parameter\n"
                    f"  3. Split long texts into smaller chunks"
                )
            
            # Use pipeline for inference (truncation=False, so it will error on long inputs)
            results = self.pipeline(texts)
            
            # Convert to consistent format
            formatted_results = []
            for result in results:
                if isinstance(result, list):
                    # Multiple scores returned
                    formatted_results.append(result)
                else:
                    # Single score returned
                    formatted_results.append([result])
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
    
    async def predict(self, texts: Union[str, List[str]]) -> ClassificationResponse:
        """Predict on text(s) using batched processing."""
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        # Create future for response
        future = asyncio.Future()
        request_id = id(future)
        
        # Add to batch queue
        await self.batch_queue.put((request_id, texts, future))
        
        # Wait for response
        return await future
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        try:
            tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            config = AutoConfig.from_pretrained(self.config.model_path)
            
            # Base model info
            model_info = {
                "model_path": self.config.model_path,
                "architecture": config.architectures[0] if config.architectures else "unknown",
                "num_labels": config.num_labels,
                "max_length": self.config.max_length,
                "model_max_length": getattr(config, 'max_position_embeddings', 'unknown'),
                "device": self.device,
                "batch_size": self.config.max_batch_size,
                "truncation_enabled": False,  # We now reject oversized inputs
                "tokenizer_info": {
                    "type": type(self.tokenizer).__name__,
                    "vocab_size": self.tokenizer.vocab_size,
                    "model_max_length": self.tokenizer.model_max_length
                }
            }
            
            # Add Weave scorer information if applicable
            if self.config.weave_scorer:
                scorer_info = self.weave_manager.get_scorer_info(self.config.weave_scorer)
                if scorer_info:
                    model_info["weave_scorer"] = {
                        "scorer_type": self.config.weave_scorer,
                        "name": scorer_info.name,
                        "description": scorer_info.description,
                        "task_type": scorer_info.task_type,
                        "size": scorer_info.size_params,
                        "hf_model_id": scorer_info.hf_model_id
                    }
            
            # Add available scorers info
            downloaded_scorers = self.weave_manager.list_downloaded_scorers()
            available_scorers = list(self.weave_manager.list_available_scorers().keys())
            
            model_info["weave_scorers"] = {
                "available": available_scorers,
                "downloaded": downloaded_scorers,
                "active": self.config.weave_scorer
            }
            
            return model_info
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}


# FastAPI app
app = FastAPI(title="Encoder Model Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
server: Optional[EncoderModelServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    global server
    if server is not None:
        await server.start_batch_processor()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global server
    if server is not None and server.batch_processor_task:
        server.batch_processor_task.cancel()
        try:
            await server.batch_processor_task
        except asyncio.CancelledError:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/info")
async def model_info():
    """Get model information."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    return server.get_model_info()


@app.get("/scorers")
async def list_scorers():
    """List available Weave scorers."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    summary = server.weave_manager.get_model_summary()
    return {
        "available_scorers": summary,
        "downloaded_count": len(server.weave_manager.list_downloaded_scorers()),
        "total_count": len(server.weave_manager.list_available_scorers()),
        "current_scorer": server.config.weave_scorer
    }


@app.post("/count-tokens", response_model=TokenCountResponse)
async def count_tokens(request: TokenCountRequest):
    """Count tokens in a text string."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        token_count = server.count_tokens(request.text)
        text_preview = request.text[:100] + "..." if len(request.text) > 100 else request.text
        
        return TokenCountResponse(
            text_preview=text_preview,
            token_count=token_count,
            max_length=server.config.max_length,
            is_valid=token_count <= server.config.max_length,
            character_count=len(request.text)
        )
        
    except Exception as e:
        logger.error(f"Token counting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    """Classify text(s)."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        response = await server.predict(request.text)
        
        # Filter results if not returning all scores
        if not request.return_all_scores:
            filtered_predictions = []
            for pred_list in response.predictions:
                if pred_list:
                    # Get the highest scoring prediction
                    best_pred = max(pred_list, key=lambda x: x['score'])
                    filtered_predictions.append([best_pred])
                else:
                    filtered_predictions.append([])
            response.predictions = filtered_predictions
        
        return response
        
    except ValueError as e:
        # Input validation errors (like oversized inputs) should return 400
        logger.warning(f"Input validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Other errors are internal server errors
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_config_from_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config.yaml in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "encoder_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.info(f"No config file found at {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="High-performance encoder model server")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory or HuggingFace model name"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for inference"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.1,
        help="Batch timeout in seconds"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--disable-optimization",
        action="store_true",
        help="Disable model optimizations"
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Use torch.compile (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--weave-scorer",
        type=str,
        help="Use a Weave local scorer (e.g., toxicity, bias, fluency, etc.)"
    )
    parser.add_argument(
        "--models-dir", 
        type=str,
        default="models",
        help="Directory containing models"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Load YAML configuration
    yaml_config = load_config_from_yaml(args.config)
    
    # Merge configurations: YAML defaults, then command line overrides
    def get_config_value(key: str, cli_value: Any, default: Any = None):
        """Get configuration value with precedence: CLI > YAML > default."""
        if cli_value is not None:
            return cli_value
        return yaml_config.get(key, default)
    
    # Handle model path vs weave scorer selection
    model_path = get_config_value('model_path', args.model_path)
    weave_scorer = get_config_value('weave_scorer', args.weave_scorer)
    
    # Either model_path or weave_scorer must be provided
    if not model_path and not weave_scorer:
        parser.error("Either --model-path or --weave-scorer must be provided (can be set via CLI or config file)")
    
    if model_path and weave_scorer:
        parser.error("Cannot specify both --model-path and --weave-scorer. Choose one.")
    
    # Create server config with merged values
    config = ServerConfig(
        model_path=model_path or "",  # Will be resolved from weave_scorer if needed
        host=get_config_value('host', args.host, "0.0.0.0"),
        port=get_config_value('port', args.port, 8000),
        max_batch_size=get_config_value('max_batch_size', args.max_batch_size, 32),
        batch_timeout=get_config_value('batch_timeout', args.batch_timeout, 0.1),
        max_length=get_config_value('max_length', args.max_length, 512),
        device=get_config_value('device', args.device, "auto"),
        num_workers=get_config_value('num_workers', args.num_workers, 4),
        enable_optimization=not get_config_value('disable_optimization', args.disable_optimization, False),
        compile_model=get_config_value('compile_model', args.compile_model, False),
        weave_scorer=weave_scorer,
        models_dir=get_config_value('models_dir', args.models_dir, "models"),
    )
    
    # Initialize global server
    global server
    server = EncoderModelServer(config)
    
    # Load model
    logger.info("Loading model...")
    server.load_model()
    logger.info("Model loaded successfully")
    
    # Print system info
    logger.info(f"System: {platform.system()} {platform.machine()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Device: {server.device}")
    logger.info(f"Model: {config.model_path}")
    
    # Start server
    logger.info(f"Starting server on {config.host}:{config.port}")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()