"""Command line interface for the encoder model server."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .server import EncoderModelServer
from .batch_processor import BatchConfig, InferenceRequest

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_model_list(models_str: str) -> List[str]:
    """Parse comma-separated model list."""
    return [model.strip() for model in models_str.split(",") if model.strip()]

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="High-performance encoder model server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names to load (e.g., 'wandb/WeaveFluencyScorerV1,wandb/WeaveBiasScorerV1')"
    )
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    # Batch processing configuration
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--max-wait-time", type=float, default=0.01, help="Maximum wait time in seconds")
    parser.add_argument("--preferred-batch-size", type=int, default=8, help="Preferred batch size")
    parser.add_argument("--max-sequence-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--disable-adaptive-batching", action="store_true", 
                       help="Disable adaptive batching")
    
    # Model optimization
    parser.add_argument("--disable-flash-attention", action="store_true",
                       help="Disable flash attention")
    parser.add_argument("--disable-compile", action="store_true",
                       help="Disable model compilation")
    parser.add_argument("--disable-better-transformer", action="store_true",
                       help="Disable BetterTransformer optimization")
    parser.add_argument("--cache-dir", type=str, help="Model cache directory")
    
    # Performance testing
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark after loading models")
    parser.add_argument("--benchmark-texts", type=int, default=100,
                       help="Number of texts for benchmarking")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Parse models
    models = parse_model_list(args.models)
    
    # Create batch configuration
    batch_config = BatchConfig(
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
        preferred_batch_size=args.preferred_batch_size,
        max_sequence_length=args.max_sequence_length,
        adaptive_batching=not args.disable_adaptive_batching
    )
    
    # Create model loader configuration
    model_loader_config = {
        "use_flash_attention": not args.disable_flash_attention,
        "compile_model": not args.disable_compile,
        "use_better_transformer": not args.disable_better_transformer,
    }
    
    if args.cache_dir:
        model_loader_config["cache_dir"] = args.cache_dir
    
    # Create server
    server = EncoderModelServer(
        models=models,
        batch_config=batch_config,
        model_loader_config=model_loader_config
    )
    
    if args.benchmark:
        # Run benchmark
        asyncio.run(run_benchmark(server, args.benchmark_texts))
    else:
        # Run server
        print(f"Starting server with models: {models}")
        print(f"Server will be available at http://{args.host}:{args.port}")
        print(f"API documentation at http://{args.host}:{args.port}/docs")
        
        server.run_server_sync(
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level
        )

async def run_benchmark(server: EncoderModelServer, num_texts: int):
    """Run performance benchmark."""
    import time
    import random
    
    print(f"Running benchmark with {num_texts} texts...")
    
    # Load models
    await server.load_models()
    
    # Generate test texts
    test_texts = [
        f"This is test text number {i} for benchmarking the model performance. "
        f"It contains some random content to simulate real usage patterns. "
        f"Random number: {random.randint(1, 1000)}"
        for i in range(num_texts)
    ]
    
    results = {}
    
    for model_name, processor in server.processors.items():
        print(f"\nBenchmarking {model_name}...")
        
        # Single requests benchmark
        single_start = time.time()
        for i, text in enumerate(test_texts):
            request = InferenceRequest(
                id=f"bench_{i}",
                texts=text
            )
            await processor.process_request(request)
        single_time = time.time() - single_start
        
        # Batch requests benchmark
        batch_start = time.time()
        tasks = []
        for i, text in enumerate(test_texts):
            request = InferenceRequest(
                id=f"batch_bench_{i}",
                texts=text
            )
            tasks.append(processor.process_request(request))
        
        await asyncio.gather(*tasks)
        batch_time = time.time() - batch_start
        
        # Calculate metrics
        single_throughput = num_texts / single_time
        batch_throughput = num_texts / batch_time
        speedup = batch_time / single_time if single_time > 0 else 1
        
        results[model_name] = {
            "single_request_time": single_time,
            "batch_request_time": batch_time,
            "single_throughput": single_throughput,
            "batch_throughput": batch_throughput,
            "speedup": speedup,
            "stats": processor.get_stats()
        }
        
        print(f"  Single requests: {single_time:.2f}s ({single_throughput:.1f} req/s)")
        print(f"  Batch requests: {batch_time:.2f}s ({batch_throughput:.1f} req/s)")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Avg batch size: {processor.get_stats()['avg_batch_size']:.1f}")
    
    # Save results
    results_file = "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark completed. Results saved to {results_file}")
    
    await server.unload_models()

if __name__ == "__main__":
    main()