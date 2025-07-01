# Weave Scorer Server

An example FastAPI server for [Weave local scorers](https://weave-docs.wandb.ai/guides/evaluation/weave_local_scorers/) with GPU acceleration and device auto-detection.

## 🚀 Quick Start

```bash
# 0. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Authenticate with wandb for Weave logging
wandb login

# 2. Install dependencies
uv sync

# 3. Start server with auto device detection
uv run python weave_scorer_server.py --scorers fluency bias toxicity

# 4. Test it
curl -X POST http://localhost:8001/score/fluency \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a well-written sentence."}'
```

## 🎯 Available Scorers

| Scorer | Description | Use Case |
|--------|-------------|----------|
| **fluency** | Text readability and quality | Content evaluation |
| **bias** | Gender and racial bias detection | Safety filtering |
| **toxicity** | Multi-dimensional toxicity detection | Content moderation |
| **coherence** | Text logical flow assessment | Quality control |
| **hallucination** | AI output accuracy verification | RAG validation |
| **context_relevance** | Query-context relevance in RAG | Search quality |
| **trust** | Composite RAG trust scoring | System reliability |
| **pii** | Personal information detection | Privacy protection |

## 🖥️ Device Support

**Auto-Detection**: Automatically selects optimal device
- **CUDA**: NVIDIA GPUs (batch_size: 64, workers: 8)
- **MPS**: Apple Silicon (batch_size: 32, workers: 6) 
- **CPU**: Fallback mode (batch_size: 16, workers: 4)

```bash
# Specific device selection
uv run python weave_scorer_server.py --device cuda --scorers fluency
uv run python weave_scorer_server.py --device mps --scorers bias toxicity
```

## 📋 API Endpoints

### Individual Scorers
```bash
# Fluency scoring
curl -X POST http://localhost:8001/score/fluency \
  -d '{"text": "Your text here"}'

# Bias detection
curl -X POST http://localhost:8001/score/bias \
  -d '{"text": ["Text 1", "Text 2"]}'

# Toxicity screening
curl -X POST http://localhost:8001/score/toxicity \
  -d '{"text": "Content to evaluate"}'

# PII detection
curl -X POST http://localhost:8001/score/pii \
  -d '{"text": "My email is john@example.com", "language": "en"}'
```

### Server Management
```bash
# Health check with device info
curl http://localhost:8001/health

# List available scorers
curl http://localhost:8001/scorers

# Generic scoring endpoint
curl -X POST http://localhost:8001/score \
  -d '{"text": "Text here", "scorer_type": "fluency"}'
```

## ⚡ Performance Features

### Intelligent Batching
- **Dynamic batching** with configurable timeouts
- **Concurrent processing** with asyncio
- **Device-optimized** batch sizes

### Advanced Caching
- **Result caching** for repeated queries
- **FIFO eviction** with configurable size
- **Request deduplication**

### GPU Optimization
- **Mixed precision** support (CUDA)
- **Model compilation** (PyTorch 2.0+)
- **Memory monitoring** and optimization

## 🔧 Configuration

### Quick Configuration
```bash
# High-throughput GPU setup
uv run python weave_scorer_server.py \
  --device cuda \
  --max-batch-size 128 \
  --batch-timeout 0.02 \
  --enable-optimization \
  --scorers fluency bias toxicity

# Low-latency setup
uv run python weave_scorer_server.py \
  --max-batch-size 4 \
  --batch-timeout 0.01 \
  --scorers fluency
```

### YAML Configuration (`gpu_config.yaml`)
```yaml
device: "auto"           # auto, cuda, mps, cpu
max_batch_size: 64      # GPU-optimized batching
batch_timeout: 0.05     # Fast GPU processing
enable_optimization: true
compile_model: true     # PyTorch 2.0+ optimization
```


## 🛠 Troubleshooting

### Authentication
```bash
# Automatic: .env file (included)
WANDB_API_KEY=your_key_here

# Manual login
wandb login your_api_key
```


## 📁 Project Structure

```
├── weave_scorer_server.py      # Main server with device detection
├── utils/
│   └── device_utils.py         # Device management and optimization
├── tests/
│   ├── weave_performance_test.py   # Comprehensive performance testing
│   ├── test_weave_server.py        # Server functionality tests
│   └── weave_performance_results.*  # Performance test results
├── deployment_scripts/        # GPU VM deployment automation
├── gpu_config.yaml            # GPU-optimized configuration
├── .env                       # WANDB authentication (included)
├── OLD_README.md              # Previous encoder server documentation
├── encoder_server.py          # Original encoder server (preserved)
└── README.md                  # This file
```