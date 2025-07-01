#!/usr/bin/env python3
"""
High-performance Weave scorer serving script for local evaluation models.
Supports all Weave local scorers with batching, caching, and beautiful CLI.
"""

import asyncio
import json
import logging
import os
import time
import yaml
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import argparse
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import weave
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import device utilities
from utils.device_utils import get_device_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScorerType(str, Enum):
    """Available Weave scorer types."""
    BIAS = "bias"
    TOXICITY = "toxicity"
    HALLUCINATION = "hallucination"
    CONTEXT_RELEVANCE = "context_relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    TRUST = "trust"
    PII = "pii"

@dataclass
class WeaveServerConfig:
    """Configuration for the Weave scorer server."""
    host: str = "127.0.0.1"
    port: int = 8001
    max_batch_size: int = 32
    batch_timeout: float = 0.1
    num_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000
    log_level: str = "INFO"
    device: str = "auto"
    enable_optimization: bool = True
    compile_model: bool = False
    mixed_precision: bool = False
    available_scorers: List[ScorerType] = field(default_factory=lambda: list(ScorerType))

class ScorerRequest(BaseModel):
    """Base request model for scoring."""
    text: Union[str, List[str]] = Field(..., description="Text(s) to score")
    scorer_type: ScorerType = Field(..., description="Type of scorer to use")
    
class BiasRequest(ScorerRequest):
    """Request model for bias scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.BIAS)

class ToxicityRequest(ScorerRequest):
    """Request model for toxicity scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.TOXICITY)

class HallucinationRequest(ScorerRequest):
    """Request model for hallucination scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.HALLUCINATION)
    context: Optional[str] = Field(None, description="Context for hallucination detection")

class ContextRelevanceRequest(ScorerRequest):
    """Request model for context relevance scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.CONTEXT_RELEVANCE)
    query: str = Field(..., description="Query to check context relevance against")
    context: str = Field(..., description="Context to evaluate")

class CoherenceRequest(ScorerRequest):
    """Request model for coherence scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.COHERENCE)

class FluencyRequest(ScorerRequest):
    """Request model for fluency scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.FLUENCY)

class TrustRequest(ScorerRequest):
    """Request model for trust scoring."""
    scorer_type: ScorerType = Field(default=ScorerType.TRUST)
    query: str = Field(..., description="Query for trust evaluation")
    context: str = Field(..., description="Context for trust evaluation")

class PIIRequest(ScorerRequest):
    """Request model for PII detection."""
    scorer_type: ScorerType = Field(default=ScorerType.PII)
    entities: Optional[List[str]] = Field(None, description="Specific PII entities to detect")
    language: str = Field(default="en", description="Language for PII detection")

class ScorerResponse(BaseModel):
    """Response model for scoring results."""
    scores: List[Dict[str, Any]] = Field(..., description="Scoring results")
    processing_time: float = Field(..., description="Processing time in seconds")
    batch_size: int = Field(..., description="Number of texts processed")
    scorer_type: ScorerType = Field(..., description="Type of scorer used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ScorerInfo(BaseModel):
    """Information about a scorer."""
    name: str
    description: str
    type: ScorerType
    available: bool
    model_info: Optional[Dict[str, Any]] = None

class WeaveScorerServer:
    """High-performance Weave scorer server with batching and caching."""
    
    def __init__(self, config: WeaveServerConfig):
        self.config = config
        self.scorers: Dict[ScorerType, Any] = {}
        self.batch_queue = asyncio.Queue()
        self.batch_processor_task = None
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.cache = {} if config.enable_caching else None
        
        # Initialize device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device(config.device)
        
        # Apply device-specific optimizations to config
        device_config = self.device_manager.get_optimal_config(self.device)
        if config.device == "auto":
            # Update config with device-optimized settings
            for key, value in device_config.items():
                if hasattr(config, key) and getattr(config, key) == getattr(WeaveServerConfig(), key):
                    setattr(config, key, value)
        
        # Initialize Weave
        weave.init("weave-scorer-server")
        
        # Log device information
        self.device_manager.log_device_info()
        logger.info(f"Using device: {self.device.value.upper()}")
        logger.info(f"Initializing Weave scorer server")
        
    async def initialize_scorers(self):
        """Initialize available Weave scorers."""
        logger.info("Initializing Weave scorers...")
        
        # Import and initialize scorers
        scorer_classes = {}
        
        try:
            # Import all scorer classes
            from weave.scorers import (
                WeaveBiasScorerV1,
                WeaveToxicityScorerV1,
                WeaveHallucinationScorerV1,
                WeaveContextRelevanceScorerV1,
                WeaveCoherenceScorerV1,
                WeaveFluencyScorerV1,
                WeaveTrustScorerV1,
                PresidioScorer
            )
            
            scorer_classes = {
                ScorerType.BIAS: WeaveBiasScorerV1,
                ScorerType.TOXICITY: WeaveToxicityScorerV1,
                ScorerType.HALLUCINATION: WeaveHallucinationScorerV1,
                ScorerType.CONTEXT_RELEVANCE: WeaveContextRelevanceScorerV1,
                ScorerType.COHERENCE: WeaveCoherenceScorerV1,
                ScorerType.FLUENCY: WeaveFluencyScorerV1,
                ScorerType.TRUST: WeaveTrustScorerV1,
                ScorerType.PII: PresidioScorer,
            }
            
        except ImportError as e:
            logger.error(f"Failed to import Weave scorers: {e}")
            logger.info("Make sure to install with: pip install weave[scorers]")
            raise
        
        # Initialize each scorer with device parameter
        for scorer_type in self.config.available_scorers:
            try:
                if scorer_type in scorer_classes:
                    logger.info(f"Initializing {scorer_type.value} scorer on {self.device.value.upper()}...")
                    
                    # Initialize scorer with device parameter if supported
                    try:
                        # Try to pass device parameter
                        self.scorers[scorer_type] = scorer_classes[scorer_type](device=self.device.value)
                        logger.info(f"✓ {scorer_type.value} scorer initialized with device={self.device.value}")
                    except TypeError:
                        # Fallback: initialize without device parameter
                        self.scorers[scorer_type] = scorer_classes[scorer_type]()
                        logger.info(f"✓ {scorer_type.value} scorer initialized (device parameter not supported)")
                    
                else:
                    logger.warning(f"Unknown scorer type: {scorer_type}")
            except Exception as e:
                logger.error(f"Failed to initialize {scorer_type.value} scorer: {e}")
                continue
        
        logger.info(f"Initialized {len(self.scorers)} scorers successfully")
    
    async def start_batch_processor(self):
        """Start the batch processing task."""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Process batches of scoring requests."""
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
                                timeout=0.01
                            )
                            batch_items.append(item)
                        except asyncio.TimeoutError:
                            break
                            
                except asyncio.TimeoutError:
                    continue
                
                # Process batch
                if batch_items:
                    await self._process_batch(batch_items)
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Handle any pending futures
                for request_id, scorer_type, texts, kwargs, future in batch_items:
                    if not future.done():
                        future.set_exception(e)
    
    async def _process_batch(self, batch_items: List[tuple]):
        """Process a batch of scoring requests."""
        try:
            # Group by scorer type
            scorer_batches = {}
            for request_id, scorer_type, texts, kwargs, future in batch_items:
                if scorer_type not in scorer_batches:
                    scorer_batches[scorer_type] = []
                scorer_batches[scorer_type].append((request_id, texts, kwargs, future))
            
            # Process each scorer type batch
            for scorer_type, items in scorer_batches.items():
                if scorer_type not in self.scorers:
                    # Set error for unsupported scorer
                    for request_id, texts, kwargs, future in items:
                        if not future.done():
                            future.set_exception(HTTPException(
                                status_code=400, 
                                detail=f"Scorer {scorer_type.value} not available"
                            ))
                    continue
                
                # Run scoring
                start_time = time.time()
                loop = asyncio.get_event_loop()
                
                for request_id, texts, kwargs, future in items:
                    try:
                        result = await loop.run_in_executor(
                            self.executor,
                            self._run_scorer,
                            scorer_type,
                            texts,
                            kwargs
                        )
                        
                        processing_time = time.time() - start_time
                        
                        response = ScorerResponse(
                            scores=result,
                            processing_time=processing_time,
                            batch_size=len(texts) if isinstance(texts, list) else 1,
                            scorer_type=scorer_type
                        )
                        
                        if not future.done():
                            future.set_result(response)
                            
                    except Exception as e:
                        if not future.done():
                            future.set_exception(e)
                        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all futures
            for request_id, scorer_type, texts, kwargs, future in batch_items:
                if not future.done():
                    future.set_exception(e)
    
    def _run_scorer(self, scorer_type: ScorerType, texts: Union[str, List[str]], kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run scorer on texts."""
        try:
            scorer = self.scorers[scorer_type]
            
            # Normalize input
            if isinstance(texts, str):
                texts = [texts]
            
            results = []
            
            # Score each text
            for text in texts:
                # Check cache first
                cache_key = None
                if self.cache is not None:
                    cache_key = f"{scorer_type.value}:{hash(text)}:{hash(str(kwargs))}"
                    if cache_key in self.cache:
                        results.append(self.cache[cache_key])
                        continue
                
                # Run scorer
                if scorer_type == ScorerType.CONTEXT_RELEVANCE:
                    result = scorer.score(context=kwargs['context'], query=kwargs['query'])
                elif scorer_type == ScorerType.HALLUCINATION:
                    result = scorer.score(output=text, context=kwargs.get('context', ''))
                elif scorer_type == ScorerType.TRUST:
                    result = scorer.score(
                        output=text, 
                        context=kwargs['context'], 
                        query=kwargs['query']
                    )
                elif scorer_type == ScorerType.PII:
                    pii_kwargs = {'output': text}
                    if 'entities' in kwargs and kwargs['entities']:
                        pii_kwargs['entities'] = kwargs['entities']
                    if 'language' in kwargs and kwargs['language']:
                        pii_kwargs['language'] = kwargs['language']
                    result = scorer.score(**pii_kwargs)
                elif scorer_type == ScorerType.COHERENCE:
                    # Coherence scorer needs query and output
                    result = scorer.score(query="", output=text)
                else:
                    # Standard scorers (bias, toxicity, fluency) use 'output' parameter
                    result = scorer.score(output=text)
                
                # Convert WeaveScorerResult to dictionary if needed
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump()
                elif hasattr(result, 'dict'):
                    result_dict = result.dict()
                elif hasattr(result, '__dict__'):
                    result_dict = result.__dict__
                else:
                    result_dict = result
                
                # Store in cache
                if self.cache is not None and cache_key:
                    if len(self.cache) >= self.config.cache_size:
                        # Simple FIFO cache eviction
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    self.cache[cache_key] = result_dict
                
                results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Scoring error for {scorer_type.value}: {e}")
            raise
    
    async def score(self, scorer_type: ScorerType, texts: Union[str, List[str]], **kwargs) -> ScorerResponse:
        """Score text(s) using specified scorer."""
        # Create future for response
        future = asyncio.Future()
        request_id = id(future)
        
        # Add to batch queue
        await self.batch_queue.put((request_id, scorer_type, texts, kwargs, future))
        
        # Wait for response
        return await future
    
    def get_scorer_info(self) -> Dict[ScorerType, ScorerInfo]:
        """Get information about available scorers."""
        scorer_descriptions = {
            ScorerType.BIAS: "Detects bias in gender and race/origin using deberta-small-long-nli model",
            ScorerType.TOXICITY: "Assesses toxicity across five dimensions using Celadon model",
            ScorerType.HALLUCINATION: "Checks for hallucinations using HHEM 2.1 model",
            ScorerType.CONTEXT_RELEVANCE: "Evaluates relevance of context to query using deberta-small-long-nli",
            ScorerType.COHERENCE: "Checks text coherence using deberta-small-long-nli model",
            ScorerType.FLUENCY: "Evaluates text fluency using ModernBERT-base model",
            ScorerType.TRUST: "Composite scorer combining multiple scorers for RAG systems",
            ScorerType.PII: "Detects Personally Identifiable Information using Presidio"
        }
        
        info = {}
        for scorer_type in ScorerType:
            info[scorer_type] = ScorerInfo(
                name=f"Weave {scorer_type.value.replace('_', ' ').title()} Scorer",
                description=scorer_descriptions.get(scorer_type, "Weave scorer"),
                type=scorer_type,
                available=scorer_type in self.scorers
            )
        
        return info

# Global server instance
scorer_server: Optional[WeaveScorerServer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global scorer_server
    if scorer_server is not None:
        await scorer_server.initialize_scorers()
        await scorer_server.start_batch_processor()
    
    yield
    
    # Shutdown
    if scorer_server is not None and scorer_server.batch_processor_task:
        scorer_server.batch_processor_task.cancel()
        try:
            await scorer_server.batch_processor_task
        except asyncio.CancelledError:
            pass

# FastAPI app
app = FastAPI(
    title="Weave Scorer Server",
    description="High-performance server for Weave local scorers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    device_info = {}
    memory_info = {}
    
    if scorer_server:
        device_info = {
            "device": scorer_server.device.value,
            "available_devices": [d.value for d in scorer_server.device_manager.device_info["available_devices"]],
            "cuda_available": scorer_server.device_manager.device_info["cuda_available"],
            "mps_available": scorer_server.device_manager.device_info["mps_available"]
        }
        memory_info = scorer_server.device_manager.get_device_memory_info()
    
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "scorers_loaded": len(scorer_server.scorers) if scorer_server else 0,
        "device_info": device_info,
        "memory_info": memory_info
    }

@app.get("/scorers")
async def list_scorers():
    """List available Weave scorers."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    scorer_info = scorer_server.get_scorer_info()
    
    return {
        "available_scorers": {k.value: v.dict() for k, v in scorer_info.items()},
        "loaded_count": len(scorer_server.scorers),
        "total_count": len(ScorerType)
    }

@app.post("/score/bias", response_model=ScorerResponse)
async def score_bias(request: BiasRequest):
    """Score text for bias."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(ScorerType.BIAS, request.text)
    except Exception as e:
        logger.error(f"Bias scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/toxicity", response_model=ScorerResponse)
async def score_toxicity(request: ToxicityRequest):
    """Score text for toxicity."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(ScorerType.TOXICITY, request.text)
    except Exception as e:
        logger.error(f"Toxicity scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/hallucination", response_model=ScorerResponse)
async def score_hallucination(request: HallucinationRequest):
    """Score text for hallucination."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        kwargs = {}
        if request.context:
            kwargs['context'] = request.context
        return await scorer_server.score(ScorerType.HALLUCINATION, request.text, **kwargs)
    except Exception as e:
        logger.error(f"Hallucination scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/context-relevance", response_model=ScorerResponse)
async def score_context_relevance(request: ContextRelevanceRequest):
    """Score context relevance to query."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(
            ScorerType.CONTEXT_RELEVANCE, 
            request.text, 
            context=request.context,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Context relevance scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/coherence", response_model=ScorerResponse)
async def score_coherence(request: CoherenceRequest):
    """Score text for coherence."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(ScorerType.COHERENCE, request.text)
    except Exception as e:
        logger.error(f"Coherence scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/fluency", response_model=ScorerResponse)
async def score_fluency(request: FluencyRequest):
    """Score text for fluency."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(ScorerType.FLUENCY, request.text)
    except Exception as e:
        logger.error(f"Fluency scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/trust", response_model=ScorerResponse)
async def score_trust(request: TrustRequest):
    """Score text for trust in RAG context."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(
            ScorerType.TRUST, 
            request.text,
            context=request.context,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Trust scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/pii", response_model=ScorerResponse)
async def score_pii(request: PIIRequest):
    """Detect PII in text."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        kwargs = {}
        if request.entities:
            kwargs['entities'] = request.entities
        if request.language:
            kwargs['language'] = request.language
        return await scorer_server.score(ScorerType.PII, request.text, **kwargs)
    except Exception as e:
        logger.error(f"PII detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score", response_model=ScorerResponse)
async def score_generic(request: ScorerRequest):
    """Generic scoring endpoint that routes to specific scorer."""
    if scorer_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    try:
        return await scorer_server.score(request.scorer_type, request.text)
    except Exception as e:
        logger.error(f"Generic scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def load_config_from_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        script_dir = Path(__file__).parent
        config_file = script_dir / "weave_config.yaml"
    else:
        config_file = Path(config_path)
    
    if not config_file.exists():
        logger.info(f"No config file found at {config_file}, using defaults")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config from {config_file}: {e}")
        return {}

def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Weave Local Scorers Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Scorers:
  bias            - Detect bias in gender and race/origin
  toxicity        - Assess toxicity across multiple dimensions  
  hallucination   - Check for hallucinations in AI outputs
  context-relevance - Evaluate context relevance to query
  coherence       - Check text coherence
  fluency         - Evaluate text fluency and readability
  trust           - Composite scorer for RAG systems
  pii             - Detect Personally Identifiable Information

Examples:
  # Start server with all scorers
  python weave_scorer_server.py --scorers bias toxicity fluency
  
  # Start with specific scorers on custom port
  python weave_scorer_server.py --port 8002 --scorers bias toxicity
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind to (default: 8001)"
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=[s.value for s in ScorerType],
        default=[s.value for s in ScorerType],
        help="Scorers to enable (default: all)"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.1,
        help="Batch timeout in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)"
    )
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable result caching"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum cache size (default: 1000)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use: auto (detect best), cuda, mps, cpu (default: auto)"
    )
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable device-specific optimizations"
    )
    parser.add_argument(
        "--compile-model",
        action="store_true", 
        help="Enable model compilation (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (CUDA only)"
    )
    
    return parser

def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load YAML configuration
    yaml_config = load_config_from_yaml(args.config)
    
    # Merge configurations
    def get_config_value(key: str, cli_value: Any, default: Any = None):
        if cli_value is not None:
            return cli_value
        return yaml_config.get(key, default)
    
    # Convert scorer strings to enum values
    scorer_types = [ScorerType(s) for s in args.scorers]
    
    # Create server config
    config = WeaveServerConfig(
        host=get_config_value('host', args.host, "0.0.0.0"),
        port=get_config_value('port', args.port, 8001),
        max_batch_size=get_config_value('max_batch_size', args.max_batch_size, 32),
        batch_timeout=get_config_value('batch_timeout', args.batch_timeout, 0.1),
        num_workers=get_config_value('num_workers', args.num_workers, 4),
        enable_caching=not get_config_value('disable_caching', args.disable_caching, False),
        cache_size=get_config_value('cache_size', args.cache_size, 1000),
        log_level=get_config_value('log_level', args.log_level, "INFO"),
        device=get_config_value('device', args.device, "auto"),
        enable_optimization=get_config_value('enable_optimization', args.enable_optimization, True),
        compile_model=get_config_value('compile_model', args.compile_model, False),
        mixed_precision=get_config_value('mixed_precision', args.mixed_precision, False),
        available_scorers=scorer_types
    )
    
    # Initialize global server
    global scorer_server
    scorer_server = WeaveScorerServer(config)
    
    # Print startup info
    logger.info("=" * 60)
    logger.info("🎯 Weave Local Scorers Server")
    logger.info("=" * 60)
    logger.info(f"System: {platform.system()} {platform.machine()}")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Device: {config.device} → {scorer_server.device.value.upper()}")
    logger.info(f"Enabled Scorers: {', '.join([s.value for s in config.available_scorers])}")
    logger.info(f"Batching: {config.max_batch_size} (timeout: {config.batch_timeout}s)")
    logger.info(f"Caching: {'enabled' if config.enable_caching else 'disabled'}")
    logger.info(f"Optimizations: {'enabled' if config.enable_optimization else 'disabled'}")
    logger.info("=" * 60)
    
    # Start server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True,
    )

if __name__ == "__main__":
    main()