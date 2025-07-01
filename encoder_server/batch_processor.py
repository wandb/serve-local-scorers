"""Efficient batch processing for encoder models."""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import deque
import threading

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Individual inference request."""
    id: str
    texts: Union[str, List[str]]
    max_length: Optional[int] = None
    truncation: bool = True
    padding: bool = True
    return_attention_mask: bool = True
    future: Optional[asyncio.Future] = None

@dataclass 
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 32
    max_wait_time: float = 0.01  # 10ms max wait time
    preferred_batch_size: int = 8
    max_sequence_length: int = 512
    adaptive_batching: bool = True
    
class BatchProcessor:
    """High-performance batch processor with dynamic batching."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: BatchConfig,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        self.request_queue = deque()
        self.batch_stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_latency": 0.0,
            "throughput": 0.0
        }
        
        self._running = False
        self._processor_task = None
        self._lock = threading.Lock()
        
        # Warmup the model
        self._warmup()
    
    def _warmup(self):
        """Warmup the model with a dummy batch."""
        logger.info("Warming up model...")
        dummy_texts = ["This is a warmup text to initialize the model."] * self.config.preferred_batch_size
        
        with torch.no_grad():
            inputs = self.tokenizer(
                dummy_texts,
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            start_time = time.time()
            _ = self.model(**inputs)
            warmup_time = time.time() - start_time
            
        logger.info(f"Model warmed up in {warmup_time:.3f}s")
    
    async def start(self):
        """Start the batch processor."""
        if self._running:
            return
            
        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches())
        logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        if self._processor_task:
            await self._processor_task
        logger.info("Batch processor stopped")
    
    async def process_request(self, request: InferenceRequest) -> Dict[str, Any]:
        """Add a request to the processing queue and wait for results."""
        if not self._running:
            await self.start()
        
        # Create future for async result
        request.future = asyncio.Future()
        
        # Add to queue
        with self._lock:
            self.request_queue.append(request)
            self.batch_stats["total_requests"] += 1
        
        # Wait for result
        return await request.future
    
    async def _process_batches(self):
        """Main batch processing loop."""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    results = await self._process_batch(batch)
                    self._return_results(batch, results)
                else:
                    # Small sleep if no requests
                    await asyncio.sleep(0.001)  # 1ms
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(0.01)
    
    async def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch using dynamic batching."""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.config.max_batch_size and self._running:
            with self._lock:
                if self.request_queue:
                    batch.append(self.request_queue.popleft())
                
            # Check if we should process the current batch
            if self._should_process_batch(batch, start_time):
                break
                
            # Small sleep to avoid busy waiting
            await asyncio.sleep(0.0001)  # 0.1ms
        
        return batch
    
    def _should_process_batch(self, batch: List[InferenceRequest], start_time: float) -> bool:
        """Determine if we should process the current batch."""
        if not batch:
            return False
            
        current_time = time.time()
        wait_time = current_time - start_time
        
        # Process if:
        # 1. We've hit max batch size
        # 2. We've waited too long
        # 3. We have preferred batch size and some wait time
        # 4. Adaptive batching suggests processing now
        
        if len(batch) >= self.config.max_batch_size:
            return True
            
        if wait_time >= self.config.max_wait_time:
            return True
            
        if len(batch) >= self.config.preferred_batch_size and wait_time >= self.config.max_wait_time * 0.5:
            return True
            
        if self.config.adaptive_batching:
            # Use throughput-based heuristic
            avg_throughput = self.batch_stats.get("throughput", 0)
            if avg_throughput > 0:
                expected_processing_time = len(batch) / avg_throughput
                if wait_time >= expected_processing_time * 0.1:  # 10% of processing time
                    return True
        
        return False
    
    async def _process_batch(self, batch: List[InferenceRequest]) -> List[Dict[str, Any]]:
        """Process a batch of requests."""
        batch_start = time.time()
        
        # Extract texts and parameters
        all_texts = []
        max_lengths = []
        
        for req in batch:
            if isinstance(req.texts, str):
                all_texts.append(req.texts)
            else:
                all_texts.extend(req.texts)
            max_lengths.append(req.max_length or self.config.max_sequence_length)
        
        # Use the minimum max_length for efficiency
        batch_max_length = min(max_lengths)
        
        # Tokenize
        inputs = self.tokenizer(
            all_texts,
            max_length=batch_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        results = self._process_outputs(outputs, batch, all_texts)
        
        # Update stats
        batch_time = time.time() - batch_start
        self._update_stats(len(batch), batch_time)
        
        return results
    
    def _process_outputs(
        self, 
        outputs, 
        batch: List[InferenceRequest], 
        all_texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Process model outputs into results."""
        results = []
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            if logits.dim() == 2:  # Sequence classification
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                text_idx = 0
                for req in batch:
                    if isinstance(req.texts, str):
                        num_texts = 1
                    else:
                        num_texts = len(req.texts)
                    
                    req_results = []
                    for i in range(num_texts):
                        result = {
                            "prediction": predictions[text_idx + i].item(),
                            "probabilities": probs[text_idx + i].cpu().numpy().tolist(),
                            "text": all_texts[text_idx + i]
                        }
                        req_results.append(result)
                    
                    results.append({
                        "request_id": req.id,
                        "results": req_results if len(req_results) > 1 else req_results[0]
                    })
                    
                    text_idx += num_texts
                    
            elif logits.dim() == 3:  # Token classification
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                text_idx = 0
                for req in batch:
                    if isinstance(req.texts, str):
                        num_texts = 1
                    else:
                        num_texts = len(req.texts)
                    
                    req_results = []
                    for i in range(num_texts):
                        result = {
                            "predictions": predictions[text_idx + i].cpu().numpy().tolist(),
                            "probabilities": probs[text_idx + i].cpu().numpy().tolist(),
                            "text": all_texts[text_idx + i]
                        }
                        req_results.append(result)
                    
                    results.append({
                        "request_id": req.id,
                        "results": req_results if len(req_results) > 1 else req_results[0]
                    })
                    
                    text_idx += num_texts
        
        return results
    
    def _return_results(self, batch: List[InferenceRequest], results: List[Dict[str, Any]]):
        """Return results to the corresponding futures."""
        for req, result in zip(batch, results):
            if req.future and not req.future.done():
                req.future.set_result(result)
    
    def _update_stats(self, batch_size: int, batch_time: float):
        """Update batch processing statistics."""
        self.batch_stats["total_batches"] += 1
        
        # Update average batch size
        total_requests = self.batch_stats["total_requests"]
        self.batch_stats["avg_batch_size"] = total_requests / self.batch_stats["total_batches"]
        
        # Update average latency
        current_avg = self.batch_stats["avg_latency"]
        batch_count = self.batch_stats["total_batches"]
        self.batch_stats["avg_latency"] = (current_avg * (batch_count - 1) + batch_time) / batch_count
        
        # Update throughput (requests per second)
        self.batch_stats["throughput"] = batch_size / batch_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current batch processing statistics."""
        return self.batch_stats.copy()