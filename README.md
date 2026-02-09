# FlashAttention vs Standard Attention

Simple comparison of **standard Transformer attention** and a **FlashAttention-style** implementation in Python/NumPy.

## File

- `CISC_719_IMC03_FlashAttention_Parallel_Algorithm.ipynb`  
  Contains:
  - Short intro to self-attention
  - Standard attention implementation
  - FlashAttention-style implementation
  - Correctness tests
  - Basic benchmarks and plots

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `seaborn`
- Google Colab (or `jupyter` for local setup)

Install dependencies (if running locally):
```bash
pip install numpy matplotlib seaborn jupyter
```

## How to Run

### Using Google Colab (Recommended)

1. Upload the notebook to Google Colab or open it directly from Google Drive
2. Go to **Runtime → Run all**

This will:

- Run correctness tests (standard vs FlashAttention)
- Run benchmarks for a few sequence lengths
- Generate runtime and memory plots

### Using Jupyter (Local)

Start Jupyter:
```bash
jupyter notebook
```

Open the notebook:
```text
CISC_719_IMC03_FlashAttention_Parallel_Algorithm.ipynb
```

In Jupyter, run:

**Kernel → Restart & Run All**

## Main Functions

### standard_attention(Q, K, V, mask=None)

Naive attention:

O = softmax(QK^T)V

### flash_attention(Q, K, V, mask=None, sram_size=...)

Blocked (tiled) version:

- Works on small blocks that fit in fast memory
- Does not store the full N×N attention matrix
- Produces the same output as standard attention

Both return:

- Output matrix O
- A stats dictionary with simple counters (FLOPs, memory, etc.)

## Goal

- Show a very simple, CPU-based illustration of the FlashAttention idea
- Compare correctness, runtime behavior, and memory usage with standard attention
- Implementation and testing done using Google Colab