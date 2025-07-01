"""High-performance encoder model server."""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Union, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .model_loader import ModelLoader
from .batch_processor import BatchProcessor, BatchConfig, InferenceRequest

logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """API request model."""
    texts: Union[str, List[str]] = Field(..., description="Text(s) to classify")
    model_name: Optional[str] = Field(None, description="Model to use (if multiple loaded)")
    max_length: Optional[int] = Field(512, description="Maximum sequence length")
    batch_size: Optional[int] = Field(None, description="Preferred batch size")
    
class PredictionResponse(BaseModel):
    """API response model."""
    request_id: str = Field(..., description="Unique request ID")
    results: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="Prediction results")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model that processed the request")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    stats: Dict[str, Any] = Field(..., description="Processing statistics")

class EncoderModelServer:
    """High-performance encoder model server with dynamic batching."""
    
    def __init__(
        self,
        models: Union[str, List[str]],
        batch_config: Optional[BatchConfig] = None,
        model_loader_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the server.
        
        Args:
            models: Model name(s) to load
            batch_config: Batch processing configuration
            model_loader_config: Model loading configuration
        """
        self.model_names = models if isinstance(models, list) else [models]
        self.batch_config = batch_config or BatchConfig()
        self.model_loader_config = model_loader_config or {}
        
        self.model_loader = ModelLoader()
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.processors: Dict[str, BatchProcessor] = {}
        
        self.app = self._create_app()
        
    async def load_models(self):
        """Load all specified models."""
        logger.info(f"Loading {len(self.model_names)} models...")
        
        for model_name in self.model_names:
            try:
                model, tokenizer = self.model_loader.load_model(
                    model_name, **self.model_loader_config
                )
                
                # Optimize model
                model = self.model_loader.optimize_for_inference(model)
                
                # Create batch processor
                processor = BatchProcessor(
                    model=model,
                    tokenizer=tokenizer,
                    config=self.batch_config,
                    device=self.model_loader.device
                )
                
                # Store everything
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.processors[model_name] = processor
                
                # Start the processor
                await processor.start()
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                raise
        
        logger.info("All models loaded successfully")
    
    async def unload_models(self):
        """Unload all models and stop processors."""
        for name, processor in self.processors.items():
            await processor.stop()
        
        self.models.clear()
        self.tokenizers.clear()
        self.processors.clear()
        
        logger.info("All models unloaded")
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.load_models()
            yield
            # Shutdown
            await self.unload_models()
        
        app = FastAPI(
            title="Encoder Model Server",
            description="High-performance serving for encoder text classification models",
            version="0.1.0",
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
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes."""
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Perform text classification."""
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Select model
            if request.model_name and request.model_name in self.processors:
                model_name = request.model_name
            elif len(self.processors) == 1:
                model_name = list(self.processors.keys())[0]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model name required. Available: {list(self.processors.keys())}"
                )
            
            processor = self.processors[model_name]
            
            # Create inference request
            inference_request = InferenceRequest(
                id=request_id,
                texts=request.texts,
                max_length=request.max_length,
            )
            
            try:
                # Process request
                result = await processor.process_request(inference_request)
                
                processing_time = time.time() - start_time
                
                return PredictionResponse(
                    request_id=request_id,
                    results=result["results"],
                    processing_time=processing_time,
                    model_used=model_name
                )
                
            except Exception as e:
                logger.error(f"Error processing request {request_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            stats = {}
            for name, processor in self.processors.items():
                stats[name] = processor.get_stats()
            
            return HealthResponse(
                status="healthy",
                models_loaded=list(self.models.keys()),
                stats=stats
            )
        
        @app.get("/models")
        async def list_models():
            """List available models."""
            model_info = {}
            for name in self.models.keys():
                model = self.models[name]
                model_info[name] = {
                    "num_parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(next(model.parameters()).device),
                    "dtype": str(next(model.parameters()).dtype)
                }
            return model_info
    
    async def run_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ):
        """Run the server."""
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def run_server_sync(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info"
    ):
        """Run the server synchronously."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level
        )