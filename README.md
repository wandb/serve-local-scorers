# Encoder Model Server

A high-performance serving solution for encoder text classification models with dynamic batching and optimization.

## Features

- **Dynamic Batching**: Intelligent batching with adaptive algorithms to maximize throughput
- **Multi-Architecture Support**: Works with DeBERTa, ModernBERT, T5, and other encoder models
- **Cross-Platform**: Optimized for both macOS (Apple Silicon) and Linux with GPUs
- **Performance Optimizations**: 
  - Flash Attention 2.0 (when available)
  - Model compilation with `torch.compile`
  - BetterTransformer optimizations
  - Automatic mixed precision
- **FastAPI Server**: RESTful API with automatic documentation
- **Comprehensive Monitoring**: Built-in performance metrics and health checks

## Supported Models

This server is designed to work with the Weave Local Scorers collection:

- `wandb/WeaveFluencyScorerV1` (ModernBERT-based, 149M params)
- `wandb/WeaveToxicityScorerV1` (DeBERTa-v2-based, 141M params)
- `wandb/WeaveBiasScorerV1` (DeBERTa-v2-based, 141M params)
- `wandb/WeaveCoherenceScorerV1` (DeBERTa-v2-based, 141M params)
- `wandb/WeaveContextRelevanceScorerV1` (T5-based, 183M params)
- `wandb/WeaveHallucinationScorerV1` (T5-small-based, 109M params)

## Quick Start

### Installation

```bash
# Install from source
git clone <repository>
cd encoder-model-server
pip install -e .

# For GPU support with Flash Attention
pip install -e .[gpu]
```

### Basic Usage

```bash
# Start server with a single model
encoder-server --models "wandb/WeaveFluencyScorerV1"

# Start server with multiple models
encoder-server --models "wandb/WeaveFluencyScorerV1,wandb/WeaveBiasScorerV1"

# Start with optimizations
encoder-server \
  --models "wandb/WeaveFluencyScorerV1" \
  --max-batch-size 64 \
  --max-wait-time 0.005 \
  --port 8080
```

### API Usage

```python
import requests

# Single text classification
response = requests.post("http://localhost:8000/predict", json={
    "texts": "This is a sample text to classify"
})
result = response.json()

# Batch text classification
response = requests.post("http://localhost:8000/predict", json={
    "texts": ["Text 1", "Text 2", "Text 3"]
})
results = response.json()

# With specific model (if multiple loaded)
response = requests.post("http://localhost:8000/predict", json={
    "texts": "Sample text",
    "model_name": "wandb/WeaveFluencyScorerV1"
})
```

## CLI Options

### Model Configuration
- `--models`: Comma-separated list of model names to load
- `--cache-dir`: Directory for model cache

### Server Configuration
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 8000)
- `--workers`: Number of worker processes (default: 1)
- `--log-level`: Logging level (debug, info, warning, error)

### Batch Processing
- `--max-batch-size`: Maximum batch size (default: 32)
- `--max-wait-time`: Maximum wait time in seconds (default: 0.01)
- `--preferred-batch-size`: Preferred batch size (default: 8)
- `--max-sequence-length`: Maximum sequence length (default: 512)
- `--disable-adaptive-batching`: Disable adaptive batching

### Optimizations
- `--disable-flash-attention`: Disable Flash Attention 2.0
- `--disable-compile`: Disable model compilation
- `--disable-better-transformer`: Disable BetterTransformer optimization

### Benchmarking
- `--benchmark`: Run performance benchmark
- `--benchmark-texts`: Number of texts for benchmarking (default: 100)

## Performance Optimization

### GPU Optimization
- Automatically uses Flash Attention 2.0 on supported GPUs
- Model compilation with `torch.compile` for additional speedup
- Dynamic batching to maximize GPU utilization
- Automatic mixed precision (bfloat16/float16)

### CPU Optimization  
- BetterTransformer optimizations for CPU inference
- Efficient padding and attention mask handling
- Automatic device selection (including Apple Silicon MPS)

### Memory Optimization
- Gradient-free inference mode
- Efficient tokenization and batching
- Model-specific optimizations for different architectures

## API Reference

### POST `/predict`

Classify text(s) using the loaded model(s).

**Request Body:**
```json
{
  "texts": "string or array of strings",
  "model_name": "optional model name if multiple loaded", 
  "max_length": 512
}
```

**Response:**
```json
{
  "request_id": "unique-request-id",
  "results": {
    "prediction": 0,
    "probabilities": [0.8, 0.2],
    "text": "input text"
  },
  "processing_time": 0.045,
  "model_used": "wandb/WeaveFluencyScorerV1"
}
```

### GET `/health`

Health check endpoint with statistics.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["wandb/WeaveFluencyScorerV1"],
  "stats": {
    "wandb/WeaveFluencyScorerV1": {
      "total_requests": 1250,
      "total_batches": 180,
      "avg_batch_size": 6.9,
      "avg_latency": 0.032,
      "throughput": 215.6
    }
  }
}
```

### GET `/models`

List loaded models and their properties.

**Response:**
```json
{
  "wandb/WeaveFluencyScorerV1": {
    "num_parameters": 149606402,
    "device": "cuda:0", 
    "dtype": "torch.bfloat16"
  }
}
```

## Performance Benchmarks

Example benchmark results on different hardware:

### NVIDIA RTX 4090 (24GB)
- **Single Model**: 850 req/s 
- **Batch Processing**: 3,200 req/s (3.8x speedup)
- **Multi-Model**: 2,800 req/s total across 4 models

### Apple M2 Max (64GB)
- **Single Model**: 180 req/s
- **Batch Processing**: 520 req/s (2.9x speedup)  
- **Multi-Model**: 450 req/s total across 4 models

### Intel Xeon CPU (64 cores)
- **Single Model**: 95 req/s
- **Batch Processing**: 240 req/s (2.5x speedup)
- **Multi-Model**: 200 req/s total across 4 models

## Architecture

The server consists of three main components:

1. **ModelLoader**: Handles model loading, optimization, and device placement
2. **BatchProcessor**: Implements dynamic batching with adaptive algorithms
3. **EncoderModelServer**: FastAPI server with async request handling

### Dynamic Batching Algorithm

The server uses an intelligent batching algorithm that considers:
- Request arrival rate
- Historical throughput
- Model processing time
- Maximum wait time constraints

This ensures optimal balance between latency and throughput.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

MIT License - see LICENSE file for details.