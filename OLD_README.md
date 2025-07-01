# Encoder Model Server

An example serving solution for Weave Local Models

## 🚀 Quick Start

## 🚀 Quick Start Example

```bash
# 0. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: pip install uv

# 1. Install dependencies
uv sync

# 2. Download a Weave scorer
uv run python download_models.py download fluency

# 3. Start the server
uv run python encoder_server.py --weave-scorer fluency

# 4. Test it
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This sentence is well-written and clear."}'
```

## Detailed Options

### 1. Install Dependencies
```bash
uv sync
```

### 2. Start the Server

**Option A: Using a Weave Local Scorer (Recommended)**
```bash
# Download and use a specific Weave scorer
uv run python download_models.py download fluency
uv run python encoder_server.py --weave-scorer fluency
```

**Option B: Using custom model path**
```bash
uv run python encoder_server.py --model-path models/your-custom-model
```

**Option C: Using configuration file**
```bash
# Edit encoder_config.yaml with your settings
uv run python encoder_server.py --config encoder_config.yaml
```

**Option D: Override config with CLI arguments**
```bash
uv run python encoder_server.py --config encoder_config.yaml --weave-scorer toxicity --port 8001
```

### 3. Manage Weave Scorers

```bash
# List all available Weave scorers
uv run python download_models.py list

# Download a specific scorer
uv run python download_models.py download fluency

# Download all scorers
uv run python download_models.py download all

# Check downloaded scorers
uv run python download_models.py list --downloaded-only

# Get detailed info about a scorer
uv run python download_models.py info fluency
```

### 4. Test the Server
```bash
# Health check
curl http://localhost:8000/health

# Model and scorer information
curl http://localhost:8000/info

# List available scorers
curl http://localhost:8000/scorers

# Count tokens (validates input length)
curl -X POST http://localhost:8000/count-tokens \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'

# Classify text
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!"}'
```

## 📋 Configuration

### YAML Configuration File (`encoder_config.yaml`)

```yaml
# Model configuration - use either model_path OR weave_scorer, not both
# model_path: "models/your-model"        # Custom model path
weave_scorer: "fluency"                  # Use a Weave local scorer

# Server configuration
host: "0.0.0.0"
port: 8000

# Performance configuration
max_batch_size: 32
batch_timeout: 0.1
max_length: 8192
device: "auto"
num_workers: 4
models_dir: "models"

# Optimization configuration
disable_optimization: false
compile_model: false
```

### Command Line Arguments

All YAML settings can be overridden via command line:

```bash
# Using a Weave scorer
python encoder_server.py \
  --weave-scorer fluency \
  --port 8001 \
  --max-batch-size 64 \
  --max-length 4096 \
  --device cuda

# Using a custom model
python encoder_server.py \
  --model-path models/your-model \
  --port 8001 \
  --max-batch-size 64
```

**Configuration Precedence**: CLI Arguments > YAML Config > Defaults

## 🔧 API Endpoints

### Classification: `POST /classify`
```json
{
  "text": "Your text here",
  "return_all_scores": false
}
```

Batch classification:
```json
{
  "text": ["Text 1", "Text 2", "Text 3"],
  "return_all_scores": false
}
```

### Token Counting: `POST /count-tokens`
```json
{
  "text": "Your text here"
}
```

Returns:
```json
{
  "text_preview": "Your text here",
  "token_count": 14,
  "max_length": 8192,
  "is_valid": true,
  "character_count": 67
}
```

### Server Info: `GET /info`
Returns model and server configuration information.

### Health Check: `GET /health`
Returns server health status.

### Scorer Management: `GET /scorers`
Returns information about all available Weave scorers:
```json
{
  "available_scorers": {
    "fluency": {
      "name": "WeaveFluencyScorerV1",
      "description": "Measures text readability and natural language quality",
      "downloaded": true,
      "valid": true
    }
  },
  "downloaded_count": 2,
  "total_count": 6,
  "current_scorer": "fluency"
}
```

## 🎯 Available Weave Scorers

| Scorer | Description | Task Type |
|--------|-------------|-----------|
| **fluency** | Measures text readability and natural language quality | text-classification |
| **bias** | Detects bias in text related to gender, race, and origin | text-classification |
| **toxicity** | Identifies toxic content across five dimensions | text-classification |
| **hallucination** | Checks for hallucinations in AI system outputs | text-classification |
| **coherence** | Assesses text coherence and logical flow | text-classification |
| **context_relevance** | Evaluates relevance of context in RAG systems | token-classification |

## ⚡ Key Features

### Dynamic Batching
- Automatically groups requests for optimal GPU utilization
- Configurable batch size and timeout
- Maintains low latency for single requests

### Cross-Platform Support
- **macOS**: Uses MPS (Apple Silicon GPU) acceleration
- **Linux**: CUDA GPU support
- **CPU**: Optimized multi-threading

### No Silent Truncation 🚨
- **Rejects oversized inputs** with clear error messages
- Token counting endpoint for pre-validation
- Transparent handling of sequence length limits
- No surprises - you know exactly what text is processed

### Performance Optimizations
- PyTorch 2.0 compilation support
- Fast tokenizers (Rust-based)
- Optimized attention mechanisms
- Thread pool execution

## 🎯 Performance Characteristics

Based on testing with ModernBERT fluency scorer on Apple Silicon:

- **Throughput**: Up to 112+ texts/sec with optimal batching
- **Latency**: Sub-100ms for reasonable batch sizes
- **Scalability**: Handles 16+ concurrent requests efficiently
- **Sequence Lengths**: Supports up to 8192 tokens (model-dependent)

### Batch Size Performance
- **Batch 1**: 43 texts/sec, 23ms latency
- **Batch 32**: 108 texts/sec, 296ms latency
- **2.5x throughput improvement** with batching

## 🧪 Testing

### Basic Functionality Test
```bash
python test_no_truncation.py
```

### Comprehensive Performance Testing
```bash
python performance_test.py --url http://localhost:8000
```

## 🎯 Example Configurations

### High-Throughput Production
```yaml
max_batch_size: 64
batch_timeout: 0.05
num_workers: 8
compile_model: true
```

### Low-Latency Production
```yaml
max_batch_size: 8
batch_timeout: 0.01
num_workers: 2
```

### Development/Testing
```yaml
max_batch_size: 4
batch_timeout: 0.2
max_length: 512
device: "cpu"
```

## 🛠 Troubleshooting

### Common Issues

**1. Model not found**
```
Error: models/your-model is not a local folder
Solution: Ensure model_path points to a valid model directory
```

**2. Input too long (No Silent Truncation!)**
```
Error: Input validation failed: 1 text(s) exceed maximum length
Solution: Use /count-tokens endpoint to check length, then:
  - Truncate inputs to max_length tokens
  - Increase max_length parameter
  - Split long texts into chunks
```

**3. CUDA out of memory**
```
Solution: Reduce max_batch_size or max_length
```

## 🔒 Production Deployment

### Recommended Settings
- Set appropriate `max_length` for your use case
- Configure `max_batch_size` based on GPU memory
- Use `compile_model: true` for production workloads
- Monitor logs for performance optimization

### Security Considerations
- Run behind a reverse proxy in production
- Implement rate limiting
- Validate inputs before sending to server
- Monitor resource usage

### Rate Limiting Example
For production deployments, implement rate limiting to protect against abuse:

```python
# Install: pip install slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add to encoder_server.py
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/classify")
@limiter.limit("100/minute")  # Allow 100 requests per minute per IP
async def classify_text(request: Request, ...):
    # Your existing code
```

**Alternative: Nginx rate limiting**
```nginx
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        location /classify {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://127.0.0.1:8000;
        }
    }
}
```

## 📚 Files Overview

**Core Files:**
- `encoder_server.py` - Main server implementation with Weave scorer support
- `encoder_config.yaml` - Default configuration
- `download_models.py` - Weave scorer manager and CLI tool
- `models/` - Model directory (auto-populated by scorer downloads)

**Testing Files:**
- `tests/test_basic.py` - Basic functionality test
- `tests/test_all_scorers.py` - Test all 6 Weave scorers
- `tests/performance_test.py` - Comprehensive performance testing