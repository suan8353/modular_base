# CardInfer Target Planning

## 1. Project Goal

Enable consumer-grade GPUs to run larger models beyond their normal capacity through serial streaming inference technology.

## 2. GPU Adaptation Targets

### 2.1 Standard Upgrade (Recommended, Speed Priority)

| VRAM | Representative GPU | Normal Capacity | Upgrade Target | Quantization | Expected Speed |
|------|-------------------|-----------------|----------------|--------------|----------------|
| 2GB | GTX 1050 | 0.5B | **4B** | Q2_K | 2-3 tok/s |
| 4GB | GTX 1650 | 1-2B | **7B-14B** | Q2_K | 3-5 tok/s |
| 6GB | GTX 1660 | 3-4B | **14B-20B** | Q3_K | 5-8 tok/s |
| 8GB | RTX 3060 | 7B | **27B-34B** | Q4_K | 8-12 tok/s |
| 12GB | RTX 4070 | 14B | **34B-70B** | Q4_K | 10-15 tok/s |
| 16GB | RTX 4080 | 20B | **70B-120B** | Q4_K | 12-18 tok/s |
| 24GB | RTX 4090 | 34B | **120B-235B** | Q4_K | 15-20 tok/s |

### 2.2 Extreme Upgrade (Slow Mode)

| VRAM | Representative GPU | Extreme Target | Quantization | Expected Speed |
|------|-------------------|----------------|--------------|----------------|
| 2GB | GTX 1050 | **14B** | Q1_K | 0.5-1 tok/s |
| 4GB | GTX 1650 | **20B-27B** | Q2_K | 1-2 tok/s |
| 8GB | RTX 3060 | **70B** | Q3_K | 3-5 tok/s |
| 24GB | RTX 4090 | **480B** | Q4_K | 10-15 tok/s |

## 3. Core Technical Metrics

### 3.1 VRAM Control
- Peak VRAM usage not exceeding **80%** of GPU capacity
- Single layer weights + activations + KV Cache total within limits

### 3.2 Inference Speed
- Minimum usable speed: **3 tok/s**
- Target speed: **2-3x slower** than full loading, but still acceptable

### 3.3 Inference Quality
- **No quality loss** compared to full loading inference
- All layers participate in computation, just loaded in batches

## 4. Technical Approach

### 4.1 Serial Streaming Inference
```
Input → Embedding(load→compute→unload) 
      → Layer0(load→compute→unload) 
      → Layer1(load→compute→unload) 
      → ... 
      → LayerN(load→compute→unload) 
      → LM_Head(load→compute→unload) 
      → Output
```

### 4.2 VRAM Analysis (14B Q4 Example)
- Single layer weights: ~200MB
- Activations: ~50MB
- KV Cache: ~100MB (can offload to RAM)
- **Total: ~350MB** (much less than 6GB limit)

## 5. MVP Acceptance Criteria

### Functional
- [ ] Qwen3-4B model slicing successful
- [ ] Serial streaming inference working
- [ ] Output matches Ollama results

### Performance
- [ ] Peak VRAM < 500MB
- [ ] Inference speed > 3 tok/s
- [ ] No OOM crashes

### Quality
- [ ] Output text semantically correct
- [ ] No garbled text/truncation
- [ ] Multi-turn dialogue working

## 6. Success Criteria

**6GB GPU successfully runs 14B model at 5+ tok/s with no quality loss.**
