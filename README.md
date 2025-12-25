# ModularBase

## Vision
ModularBase makes professional-grade AI accessible on consumer hardware.  
Currently led by a single developer, the project is designed to grow into a **community-driven ecosystem** with modular packages, fostering collaboration, innovation, and democratized AI access.

## Current LLM Challenges
- **High Hardware Requirements**: 7B+ models need 14GB+ VRAM, out of reach for most users.  
- **Costly Capability Expansion**: Adding new abilities requires full model retraining, slowing innovation.  
- **Knowledge Updates Cause Forgetting**: Updating models often loses previously learned knowledge, making improvements disruptive.

## ModularBase Solution
Decomposes LLMs into a **lean base model + pluggable data packs**, allowing for:
- Efficient operation on low-end hardware
- Independent module updates
- On-demand loading
- Continuous expansion without forgetting

## Key Features
- **Ultra-low VRAM**: Runs seamlessly on 4GB GPUs (~1.6GB resident)  
- **Modular Capabilities**: Data packs are independent, upgradeable, and extendable  
- **On-demand Loading**: Only load required modules to optimize memory and compute  
- **No Forgetting**: Add new capabilities without disrupting existing knowledge  
- **Easy Extension**: Train new data packs for specialized tasks to future-proof the system

## Architecture Overview
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

````

## VRAM Comparison
| Solution      | VRAM Required | Capability Extension | Update Cost |
|---------------|---------------|--------------------|------------|
| Llama-7B      | 14GB+         | Full retrain        | High       |
| Qwen-1.8B     | 4GB           | Full retrain        | Medium     |
| ModularBase   | ~2-3GB        | Add data packs      | Low        |

## Quick Start
```bash
# Clone the project
git clone https://github.com/suan8353/modular_base.git
cd ModularBase

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download training data
python scripts/download_data.py

# Train the base model
python scripts/train_base.py

# Train data packs
python scripts/train_packs.py

# Test inference
python scripts/test_inference.py
````

## Project Structure

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

## Roadmap

**Phase 1: Architecture Validation ✅**

* Core architecture design
* Base + Data Pack + Router implementation
* Inference engine prototype
* Small-scale training validation

**Phase 2: Model Training 🚧**

* Large-scale data training (50K–200K)
* Base model optimization
* Core data pack training

**Phase 3: Performance Optimization**

* INT8/INT4 quantization
* KV Cache compression
* C++ inference engine

**Phase 4: Ecosystem Building**

* Data pack development tools
* Data pack marketplace
* Community contribution guide

## Contributing

We are excited to grow ModularBase with a **passionate, collaborative community**.
Currently, the project is led by a single developer, but the goal is to **build an open-source ecosystem** where contributors can develop modules, enhance the core engine, and expand documentation.

You can participate by:

* 🌟 Starring the project to follow progress
* 💡 Engaging in discussions around architecture, features, and improvements
* 🔧 Contributing code, enhancements, or new data packs via Pull Requests
* 📚 Expanding documentation and guides to make the project accessible to all

Whether you’re building specialized data packs or improving the core engine, your contributions will directly shape the future of modular AI.

## License

Apache 2.0

> "The future of AI shouldn't belong only to those with top-tier hardware."

