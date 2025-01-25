# Quantum-Inspired Matrix Product States (MPS)

This repository implements a Quantum-inspired Matrix Product States (MPS) model in PyTorch, combining quantum mechanics principles with tensor networks for sequence modeling tasks.

## Overview

Matrix Product States (MPS) are a type of tensor network that originated in quantum physics for efficiently representing quantum many-body systems. This implementation provides a quantum-inspired approach to sequence modeling, featuring:

- Quantum measurement-based embeddings
- Isometric tensor network architecture
- Efficient sampling and probability calculations
- GPU acceleration support

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Jupyter (for running the tutorial)

## Quick Start

To use the model, follow these steps:

1. Import the necessary modules:

```

## Tutorial

A comprehensive tutorial is provided in `tutorial.ipynb`, which covers:
1. Setup and Basic Usage
2. Quantum Embeddings
3. Training an MPS Model
4. Sampling and Generation
5. Advanced Features

To run the tutorial:

```bash
jupyter notebook tutorial.ipynb
```

## Project Structure

```
.
├── main.py              # Core implementation
├── tutorial.ipynb       # Interactive tutorial
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Key Components

### QuantumEmbed
- Custom embedding layer that represents tokens as quantum measurements
- Supports both one-hot and learned embeddings
- Maintains quantum mechanical properties through QR decomposition

### MPS (Matrix Product States)
- Implements the core MPS architecture
- Features efficient tensor contractions
- Supports both training and generation
- Maintains isometric constraints

## Training

The model supports training with custom loss functions and includes specialized optimizers that maintain quantum mechanical constraints. Example training setup:

```python
import torch.optim as optim
from main import iso_op

optimizer = optim.Adam([
    {'params': model.mps_blocks, 'name': 'iso'},
    {'params': model.emb.parameters(), 'name': 'emb'}
], lr=0.001)

# Training loop with isometric constraints
optimizer.zero_grad()
loss.backward()
iso_op(optimizer, loss)
optimizer.step()
```

## Features

- **Efficient Tensor Operations**: Leverages PyTorch's tensor operations for MPS calculations
- **Quantum-Inspired Design**: Incorporates quantum mechanical principles in classical machine learning
- **Scalable Architecture**: Memory-efficient representation of high-dimensional data
- **GPU Support**: Automatic CUDA acceleration when available
- **Sampling Capabilities**: Supports both complete and step-by-step sequence generation

## Technical Details

- The model maintains isometric constraints through QR decomposition and gradient projection
- Implements efficient sampling through a right-to-left procedure
- Uses complex-valued tensors for quantum state representations
- Supports both complete and incomplete sequence probability calculations
