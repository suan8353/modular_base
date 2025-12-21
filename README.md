# ModularBase

**Minimal Base + Pluggable Data Packs** — Run Professional AI on 4GB GPUs

[![Status](https://img.shields.io/badge/Status-Active%20Development-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)]()

---

## 🎯 Vision

Current LLM challenges:
- 7B models require 14GB+ VRAM, out of reach for most users
- Adding new capabilities requires full retraining, expensive
- Knowledge updates cause catastrophic forgetting

**ModularBase's Solution**: Decompose LLMs into "Minimal Base + Pluggable Data Packs"

```
Traditional LLM:  [████████████████████████████] 7B params, monolithic

ModularBase:      [Base 0.5B] + [Router] + [Chat] + [Code] + [Medical] + ...
                       ↑           ↑         ↑         ↑          ↑
                   Resident    Resident  Resident  On-demand  On-demand
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Ultra-low VRAM** | Runs smoothly on 4GB GPUs, ~1.6GB resident |
| **Modular Capabilities** | Data packs train and update independently |
| **On-demand Loading** | Only load packs needed for current task |
| **No Forgetting** | New capabilities don't affect existing ones |
| **Easy Extension** | Add new domains by training new data packs |

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Resident (~1.6GB)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Base Core (Understanding + Routing + Fusion) │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌──────────────┐              ┌──────────────┐           │
│  │  Router Pack │              │ General Chat │           │
│  └──────────────┘              └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ On-demand Loading
┌─────────────────────────────────────────────────────────────┐
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │  Code  │ │Reasoning│ │Medical │ │ Legal  │ │Creative│   │
│  │  Pack  │ │  Pack   │ │  Pack  │ │  Pack  │ │  Pack  │   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📊 VRAM Comparison

| Solution | VRAM Required | Capability Extension | Update Cost |
|----------|---------------|---------------------|-------------|
| Llama-7B | 14GB+ | Full retrain | High |
| Qwen-1.8B | 4GB | Full retrain | Medium |
| **ModularBase** | **~2-3GB** | **Add data packs** | **Low** |

## 🚀 Quick Start

```bash
# Clone the project
git clone https://github.com/your-username/ModularBase.git
cd ModularBase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download training data
python scripts/download_data.py

# Train base model
python scripts/train_base.py

# Train data packs
python scripts/train_packs.py

# Test inference
python scripts/test_inference.py
```

## 📁 Project Structure

```
ModularBase/
├── docs_cn/                    # Chinese documentation
├── modular_base/               # Core implementation
│   ├── model/                  # Model definitions (base, packs, router)
│   ├── engine/                 # Inference engine (pack manager, context compression)
│   └── training/               # Training utilities
├── packs/                      # Trained data packs
├── data/                       # Training data

```

## 🗺️ Roadmap

### Phase 1: Architecture Validation ✅
- [x] Core architecture design
- [x] Base + Data Pack + Router implementation
- [x] Inference engine prototype
- [x] Small-scale training validation

### Phase 2: Model Training 🚧 In Progress
- [ ] Large-scale data training (50K-200K)
- [ ] Base model optimization
- [ ] Core data pack training

### Phase 3: Performance Optimization
- [ ] INT8/INT4 quantization
- [ ] KV Cache compression
- [ ] C++ inference engine

### Phase 4: Ecosystem Building
- [ ] Data pack development tools
- [ ] Data pack marketplace
- [ ] Community contribution guide

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Core architecture and design decisions |
| [Optimization](optimization.md) | Performance, engineering, robustness optimization |
| [Progress](progress.md) | Current status and next steps |
| [CardInfer](cardinfer.md) | Serial streaming inference engine |
| [Roadmap](roadmap.md) | Development roadmap |
| [Contributing](contributing.md) | How to contribute |

## 🤝 Contributing

The project is in early development. Welcome to:
- 🌟 Star to follow progress
- 💡 Discuss architecture in Issues
- 🔧 Contribute code via PR

See [Contributing Guide](contributing.md)

## 📄 License

Apache 2.0

---

> **"The future of AI shouldn't belong only to those with top-tier hardware"**


