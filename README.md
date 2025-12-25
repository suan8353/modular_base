### Vision

#### Current LLM Challenges:

* **7B models require 14GB+ VRAM, out of reach for most users**
  *Most large language models demand high-end hardware, locking out a vast majority of potential users and developers.*

* **Adding new capabilities requires full retraining, expensive**
  *The high cost and complexity of retraining entire models for new functionalities limits the ability to rapidly innovate and extend AI capabilities.*

* **Knowledge updates cause catastrophic forgetting**
  *Updating models with new knowledge often results in losing previously learned information, making updates inefficient and disruptive.*

#### ModularBase's Solution:

* **Decompose LLMs into "Minimal Base + Pluggable Data Packs"**
  *ModularBase tackles these challenges head-on by modularizing LLMs, allowing for flexible, on-demand expansion while maintaining a lean, efficient base model.*

---

### Key Features

| **Feature**              | **Description**                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **Ultra-low VRAM**       | *Runs seamlessly on 4GB GPUs, requiring only ~1.6GB of resident VRAM for efficient operations*                 |
| **Modular Capabilities** | *Data packs are independent, easily upgradable, and extendable—adding new functionality has never been easier* |
| **On-demand Loading**    | *Only the required data packs are loaded, optimizing memory and processing power for each task*                |
| **No Forgetting**        | *Add new capabilities without disrupting existing knowledge—continuous improvement without trade-offs*         |
| **Easy Extension**       | *Easily extendable by training new data packs for specialized tasks, making this platform future-proof*        |

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Resident (~1.6GB)                                  │
│  ┌─────────────────────────────────────────────────────┐           │
│  │           Base Core (Understanding + Routing + Fusion) │           │
│  └─────────────────────────────────────────────────────┘           │
│  ┌──────────────┐              ┌──────────────┐                   │
│  │  Router Pack │              │ General Chat │                   │
│  └──────────────┘              └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ On-demand Loading
┌─────────────────────────────────────────────────────────────────────┐
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐           │
│  │  Code  │ │Reasoning│ │Medical │ │ Legal  │ │Creative│           │
│  │  Pack  │ │  Pack   │ │  Pack  │ │  Pack  │ │  Pack  │           │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

### VRAM Comparison

| **Solution**    | **VRAM Required** | **Capability Extension** | **Update Cost** |
| --------------- | ----------------- | ------------------------ | --------------- |
| **Llama-7B**    | 14GB+             | Full retrain             | High            |
| **Qwen-1.8B**   | 4GB               | Full retrain             | Medium          |
| **ModularBase** | ~2-3GB            | Add data packs           | Low             |

---

### Quick Start

1. **Clone the project**

   ```bash
   git clone https://github.com/your-username/ModularBase.git
   cd ModularBase
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download training data**

   ```bash
   python scripts/download_data.py
   ```

5. **Train the base model**

   ```bash
   python scripts/train_base.py
   ```

6. **Train data packs**

   ```bash
   python scripts/train_packs.py
   ```

7. **Test inference**

   ```bash
   python scripts/test_inference.py
   ```

---

### Project Structure

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

---

### Roadmap

**Phase 1: Architecture Validation** ✅

* Core architecture design
* Base + Data Pack + Router implementation
* Inference engine prototype
* Small-scale training validation

**Phase 2: Model Training** 🚧 *In Progress*

* Large-scale data training (50K-200K)
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

---

### Documentation

| **Document**     | **Description**                                     |
| ---------------- | --------------------------------------------------- |
| **Architecture** | *Core architecture and design decisions*            |
| **Optimization** | *Performance, engineering, robustness optimization* |
| **Progress**     | *Current status and next steps*                     |
| **CardInfer**    | *Serial streaming inference engine*                 |
| **Roadmap**      | *Development roadmap*                               |
| **Contributing** | *How to contribute*                                 |

---

### Contributing

We are excited to grow ModularBase with the help of a passionate and innovative community. Here’s how you can get involved:

* 🌟 *Star the project to stay updated on progress.*
* 💡 *Engage in discussions around architecture, features, and potential improvements.*
* 🔧 *Contribute code, enhancements, and new data packs through Pull Requests.*
* 📚 *Help expand our documentation and guides to make it easier for everyone to get started.*

We encourage developers, AI researchers, and enthusiasts to explore the project, share feedback, and contribute to building a modular AI future. Whether you’re interested in creating specialized data packs or improving the core engine, your contribution can make a significant impact.

We believe that **ModularBase** can transform the AI ecosystem into a more inclusive and adaptable space, and we want **YOU** to be a part of it!

---

### License

Apache 2.0

**"The future of AI shouldn't belong only to those with top-tier hardware."**

---
