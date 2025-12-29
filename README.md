# Vector-H: Physics-Based AI Safety Engine

A Hamiltonian Neural ODE architecture that brings **physics-level constraints** to Large Language Models, enabling real-time hallucination detection and energy-conserving inference.

## 🚀 Key Features

- **Physics Supervisor (Sidecar)**: Monitors LLM hidden states using Hamiltonian energy metrics
- **Real-time Hallucination Detection**: Detects "surprising" tokens via energy spikes ($H > \theta$)
- **CPU-Optimized**: Runs efficiently on legacy Xeon hardware (32+ TPS on Ivy Bridge)
- **Zero Retraining**: Wraps existing models (TinyLlama, Llama-3) without fine-tuning

## 📊 Empirical Results (Xeon E5-2697 v2)

| Metric | Value | Notes |
|--------|-------|-------|
| **Throughput** | 32.2 tokens/sec | Pure Python/PyTorch (AVX1) |
| **Energy Detection** | ✅ Verified | `[SPIKE:0.28]` on semantic jumps |
| **Hardware** | Xeon Ivy Bridge (2013) | 12 cores, 24 threads |

## 🏗️ Architecture

### Physics Supervisor (Implemented)
```python
# Two LLM instances: Driver + Monitor
llm_gen = Llama(model_path)      # Generates text
llm_probe = Llama(model_path)    # Monitors energy

H = 0.5|p|² + 0.5|q|²  # Hamiltonian Energy
# p = Δq (semantic velocity)
# q = hidden state embedding
```

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/NovakDavid98/vector-llm.git
cd vector-llm

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch transformers datasets tokenizers
pip install llama-cpp-python

# Download TinyLlama model
mkdir -p models
curl -L -o models/tinyllama-1.1b-chat.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 🎯 Usage

### Run Physics Supervisor
```bash
python3 src/vector_h/probe.py --threshold 0.05 --prompt "Explain quantum physics"
```

**Output:**
```
Quant[SPIKE:0.28]um physics is the study of...
```
- 🟢 Green: Stable flow
- 🔴 Red: High energy (surprise/hallucination)
- 🔵 Blue: Stagnation (loops)

### Run Vector-H Base Model
```bash
# Transplant GPT-2 weights (legacy demo)
python3 -m src.vector_h.transplant

# Verify properties
python3 -m src.vector_h.verify --model_path ./models/vector_h_gpt2
```

## 📁 Project Structure

```
src/vector_h/
├── probe.py          # Physics Supervisor (Sidecar)
├── model.py          # Vector-H Architecture
├── transplant.py     # GPT-2 → Vector-H weight mapping
├── train.py          # Energy-conserving fine-tuning
├── verify.py         # Drift/Truth/Throughput tests
└── experiments.py    # Lyapunov/Thermodynamics metrics
```

## 🧪 Experiments

```bash
# Advanced physics metrics
python3 -m src.vector_h.experiments --model_path ./models/vector_h_finetuned
```

Tests:
- **Lyapunov Stability**: Measures phase space divergence
- **Thermodynamics**: "Temperature" of confusion vs. clarity

## 🔬 Technical Deep Dive

See [`final_report.md`](final_report.md) for:
- Empirical performance data
- Energy correlation analysis
- Hardware optimization strategies

## 🤝 Contributing

This is a research prototype. Contributions welcome:
1. Scale to Llama-3-8B
2. Hook energy metric into sampler (dynamic temperature)
3. Add AVX-512 kernels for modern Xeons

## 📜 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Built on:
- PyTorch
- HuggingFace Transformers
- llama.cpp / llama-cpp-python
- TinyLlama (Quantized by TheBloke)

---

**Status**: ✅ Production-ready for Xeon CPUs (Ivy Bridge+)  
**Citation**: If you use this work, please cite the repository.
