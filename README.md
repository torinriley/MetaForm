# MetaForm

**MetaForm** is an open-source transformer model and library designed for high-performance natural language processing (NLP) tasks. This project provides a robust and flexible foundation for building and experimenting with transformer-based architectures, suitable for research and production environments.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [Core Components](#core-components)
  - [Layers](#layers)
  - [Tools](#tools)
  - [Training Utilities](#training-utilities)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Modular Design**: Easily customizable transformer blocks and models.
- **Advanced Training**: Built-in support for gradient checkpointing, mixed precision training, and distributed training.
- **Scalability**: Efficient memory management and parallelization for large-scale models.
- **Flexibility**: Tools for easy layer distribution across multiple devices, gradient aggregation, and parameter updates.

## Installation

To install MetaForm, you can clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/torinriley/metaform.git
cd metaform
pip install -r requirements.txt
```

## Quick Start
Here’s a quick example to get started with building and training a transformer model using MetaForm:

```python
from metaform.transformer.core import TransformerModel
from metaform.tools.matrix import Matrix
from metaform.tools.training import TrainingLoop

# Example usage
embed_size = 512
num_heads = 8
ff_hidden_size = 2048
num_layers = 12
dropout = 0.1

model = TransformerModel(embed_size, num_heads, ff_hidden_size, num_layers, dropout)

# Dummy input
input_data = Matrix.random(64, embed_size)  # Batch size of 64

# Forward pass
output = model.forward(input_data)

print("Output shape:", output.shape())
```



# Modules

## Core Components
- **TransformerModel**: The main transformer model class, allowing for the creation of custom transformer architectures with a flexible number of layers and attention heads.
- **GradientCheckpointing**: Utility for memory-efficient training through gradient checkpointing.

## Layers
- **MultiHeadAttention**: Implements the multi-head self-attention mechanism.
- **FeedForward**: A fully connected feedforward network layer.
- **Normalization**: Layer normalization to stabilize training.

## Tools
- **Matrix Operations**: Custom matrix operations designed to replace numpy for matrix manipulation, supporting core linear algebra operations.
- **Activation Functions**: Implementation of activation functions like ReLU, Sigmoid, and Tanh.
- **Memory Management**: Utilities for efficient memory usage during training.

## Training Utilities
- **Distributed Training**: Tools for training models across multiple GPUs or machines.
- **Mixed Precision**: Support for mixed precision training to speed up computations while saving memory.
- **Parallelization**: Methods to distribute layers and operations across multiple devices for efficient parallel processing.

# Usage Examples
Below is an example of how to use MetaForm for a basic training loop:

```python
from metaform.transformer.core import TransformerModel
from metaform.tools.matrix import Matrix
from metaform.tools.training import TrainingLoop

# Initialize model
model = TransformerModel(embed_size=512, num_heads=8, ff_hidden_size=2048, num_layers=12)

# Dummy input data
input_data = Matrix.random(64, 512)

# Forward pass
output = model.forward(input_data)

# Implement your training loop here
training_loop = TrainingLoop(model)
training_loop.train(input_data)
```




# Contributing
Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for information.

# License
MetaForm is released under the [MIT License](LICENSE). See the LICENSE file

