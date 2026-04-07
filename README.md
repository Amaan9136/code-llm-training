# CodeLLM Training System

Production-ready end-to-end LLM training system for coding data from GitHub repositories.

## Architecture

```
GitHub Repos → Ingestion → Preprocessing → Dataset → Training → Fine-tuning → Inference API
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/setup.py
python scripts/ingest.py --repo https://github.com/user/repo
python scripts/train.py --config config/training.yaml
python scripts/serve.py --model-path outputs/model
```

## Components

- `core/` — Repository ingestion, code extraction, tokenization
- `pipeline/` — Data pipeline, dataset creation, preprocessing
- `training/` — Training loop, fine-tuning, checkpointing
- `inference/` — Model serving, inference engine
- `api/` — REST API for inference and management
- `ui/` — Web dashboard
- `config/` — Configuration files
- `scripts/` — CLI entry points