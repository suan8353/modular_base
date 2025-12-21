# ModularBase - Modular Base Model Architecture Design

## 1. Core Philosophy

Decompose large models into a **"Minimal Base + Pluggable Data Packs"** modular architecture:
- Base only handles understanding and routing, doesn't store domain knowledge
- Knowledge and capabilities are encapsulated in independent data packs
- Data packs can be shared, incrementally trained, and loaded on-demand
- Context is compressed and cached to virtual memory for ultra-long context support

## 2. Core Design Decisions

| Question | Decision |
|----------|----------|
| Data Pack Form | **Independent small networks**, not LoRA stacking |
| Base Origin | **Train from scratch** dedicated base |
| Data Pack Essence | Domain knowledge compressed into weights + expert networks |
| Context Management | Compressed and cached to virtual memory (disk) |
| Implementation | **Hybrid**: Python training + C++ inference |
| Training Method | **Standard PyTorch**, compatible with existing ecosystem |

## 2.1 Hybrid Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Layer (Training + High-level Logic)    │
│  - PyTorch model definitions                                     │
│  - Training loops, optimizers                                    │
│  - Pack scheduling logic                                         │
│  - API interfaces                                                │
├─────────────────────────────────────────────────────────────────┤
│                    C++ Layer (Inference + Performance Critical)  │
│  - Attention computation (CUDA accelerated)                      │
│  - Matrix operations (cuBLAS)                                    │
│  - KV Cache compression/decompression                            │
│  - Data pack loading/unloading                                   │
│  - Memory management                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Why this design:**
- Training with PyTorch: Compatible with existing tools (wandb, tensorboard), can use pretrained weights
- Inference with C++: Maximum performance, reduced Python overhead
- Unified weight format: Export after training, load for C++ inference

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Resident (Cannot Unload)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Base Core (~0.3B)                       │  │
│  │     Embedding │ Context Mgmt │ Pack Scheduler │ Decoder    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │ Router Pack  │                    │ General Chat │           │
│  │    ~0.2B     │                    │    ~0.3B     │           │
│  │ (Required)   │                    │  (Required)  │           │
│  └──────────────┘                    └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ On-demand Loading
┌─────────────────────────────────────────────────────────────────┐
│                    On-demand Packs (Pluggable)                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Reasoning│ │ Python │ │Medical │ │ Legal  │ │Creative│  ...  │
│  │  Pack   │ │  Pack  │ │  Pack  │ │  Pack  │ │  Pack  │       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
└─────────────────────────────────────────────────────────────────┘
```


## 4. Detailed Architecture Design

### 4.1 Overall Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Base Core                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │Embedding│ -> │Understand│ -> │Scheduler│ -> │ Fusion  │       │
│  └─────────┘    └─────────┘    └────┬────┘    └────┬────┘       │
└────────────────────────────────────│──────────────│─────────────┘
                                     │              │
                    ┌────────────────┘              │
                    ▼                               │
┌─────────────────────────────────────┐             │
│         Router Pack (Resident)       │             │
│  Analyze intent -> Select packs      │             │
└─────────────────────────────────────┘             │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────┐            │
│    Selected Data Packs (Independent) │ ──────────┘
│  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │General│  │Reason│  │Domain│      │
│  └──────┘  └──────┘  └──────┘      │
└─────────────────────────────────────┘
                    │
                    ▼
               Output
```

### 4.2 Base Core Design

The base is minimal, only handles **understanding** and **routing**:

```
Base Core (~0.5B)
├── Embedding Layer (shared, reused by all packs)
│   └── Vocab: 32K-64K
│   └── Dimension: 1024
│
├── Understanding Layers (4-6 Transformer layers)
│   └── Understand input semantics
│   └── Generate hidden state for data packs
│
├── Scheduler
│   └── Receive router pack decisions
│   └── Manage pack loading/unloading
│
└── Fusion Layers (2 layers)
    └── Merge outputs from multiple packs
    └── Generate final output
```

### 4.3 Data Pack Design (Independent Small Networks)

Each data pack is an **independent small Transformer network**:

```
Data Pack Structure (~0.1-0.3B)
├── Input Adapter Layer
│   └── Receive hidden state from base
│
├── Expert Layers (4-8 Transformer layers)
│   └── Domain knowledge encoded here
│   └── Independent attention and FFN
│
└── Output Adapter Layer
    └── Output to base fusion layer
```

**Data Pack Types:**

| Type | Name | Size | Description |
|------|------|------|-------------|
| Resident | Router Pack | ~0.1B | Analyze intent, select packs |
| Resident | General Chat | ~0.2B | Chitchat, common sense |
| On-demand | Reasoning Pack | ~0.2B | Logic, math |
| On-demand | Domain Packs | ~0.1-0.2B | Python/Medical/Legal... |

### 4.4 Context Compression Cache

**Core idea**: Compress processed context and store in virtual memory (disk), reload when needed

```
┌─────────────────────────────────────────────────────────────────┐
│                      Context Management                          │
│                                                                  │
│   GPU VRAM (Hot)                Virtual Memory/Disk (Cold)       │
│  ┌─────────────────┐        ┌─────────────────────────┐        │
│  │ Current Window  │        │ Compressed History      │        │
│  │ (Recent 2K tok) │  <-->  │ (Up to 100K+ tokens)    │        │
│  │ Full precision  │        │ Quantized storage       │        │
│  └─────────────────┘        └─────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Multi-Pack Collaboration

When multiple packs need to work together:

```
Input: "Write a quicksort in Python and explain the principle"
                │
                ▼
Router decides: Need [General Chat + Python Pack + Reasoning Pack]
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
    ┌──────┐┌──────┐┌──────┐
    │General││Python││Reason│  <- Parallel processing
    └──┬───┘└──┬───┘└──┬───┘
       │       │       │
       └───────┼───────┘
               ▼
         Fusion Layer
               │
               ▼
            Output
```

**Fusion Strategies:**
- Weighted average: Weight by router confidence scores
- Attention fusion: Let pack outputs attend to each other
- Primary-secondary: One main pack, others assist

## 6. VRAM & Storage Allocation

### 6.1 VRAM Allocation (4GB GPU)

| Component | VRAM | Notes |
|-----------|------|-------|
| Base Core | ~1.0GB | Resident |
| Router Pack | ~0.2GB | Resident |
| General Chat | ~0.4GB | Resident |
| **Resident Total** | **~1.6GB** | |
| On-demand slots | ~0.6GB | Max 2-3 packs |
| Current context | ~0.5GB | Hot KV Cache |
| Compute buffer | ~0.3GB | |
| **Total** | **~3.0GB** | 1GB headroom |

## 7. Comparison with Traditional Architecture

| Feature | Traditional LLM | ModularBase |
|---------|-----------------|-------------|
| Knowledge Storage | All in one network | Distributed in packs |
| Update Method | Full fine-tuning | Update relevant pack only |
| VRAM Usage | Fixed | Dynamic on-demand |
| Context Length | Limited by VRAM | Can compress to disk |
| Capability Extension | Retrain | Add new pack |
| Forgetting Problem | Severe | Pack isolation prevents |

---

**Project Codename**: ModularBase  
**Goal**: Minimal base + Independent data packs + Compressed context, achieving modular AI on 4GB GPUs
