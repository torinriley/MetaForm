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

This flowchart provides a detailed visualization of the MetaForm Transformer Model, specifically designed as a Large Language Model (LLM) for sequence generation tasks. It outlines the core components, from input preprocessing and positional encoding to the stacked transformer blocks and the intricate multi-head attention mechanism. The flow also includes critical decision points such as gradient checkpointing for memory optimization and the End-of-Sequence (EOS) token generation, which determines when the model completes generating text. The diagram showcases the decoder and softmax layers for token prediction, leading to the final sequence output. This flowchart serves as a clear representation of the entire data flow through the MetaForm LLM, offering insights into its architecture and sequence generation process.
<img width="4064" alt="MetaForm" src="https://github.com/user-attachments/assets/43512fbc-01f1-462d-ba24-37759f4f93bc">


## Core Components
- **TransformerModel**: The main transformer model class, allowing for the creation of custom transformer architectures with a flexible number of layers and attention heads.
- **GradientCheckpointing**: Utility for memory-efficient training through gradient checkpointing.

## Layers
- **MultiHeadAttention**: Implements the multi-head self-attention mechanism.
- **FeedForward**: A fully connected feedforward network layer.
- **Normalization**: Layer normalization to stabilize training.

### Layers Overview

In the **MetaForm LLM** model, several layers are used to process the input data and transform it into meaningful output. This includes the **Multi-Head Attention** mechanism, **Feedforward Neural Networks**, and **Layer Normalization**, among other key components. Below is a mathematical breakdown of how these layers function.

#### Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. For each head, the attention mechanism computes **scaled dot-product attention** using the following equation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Where:
\( Q \) (queries), \( K \) (keys), and \( V \) (values) are linear projections of the input embeddings.
\( d_k \) is the dimensionality of the keys.
\( \text{softmax} \) is applied to normalize the attention scores.

In **multi-head attention**, multiple sets of \( Q \), \( K \), and \( V \) matrices are learned, and each head computes its own attention output. These are then concatenated and linearly transformed:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O
$$

Where:
\( W_O \) is the output projection matrix.
\( h \) is the number of attention heads.

Each attention head is calculated as:

$$
{head}i = \text{Attention}(QW_{q,i}, KW_{k,i}, VW_{v,i})
$$

Where:

$$
W_{q,i}, W_{k,i}, W_{v,i}
$$

are learned projection matrices for queries, keys, and values for each head.

---

#### Feedforward Network

After the multi-head attention mechanism, the output is passed through a **Feedforward Neural Network (FFN)**. Each position in the sequence is processed independently by the FFN. The FFN consists of two linear transformations with a ReLU activation in between:

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

Where:
- \( W_1 \) and \( W_2 \) are weight matrices.
- \( b_1 \) and \( b_2 \) are bias terms.
- \( \text{ReLU}(x) = \max(0, x) \) is the Rectified Linear Unit activation function.

The FFN is applied to each position in the sequence separately but with the same parameters, making it position-wise feedforward.

---

#### Layer Normalization

Layer normalization is applied after the multi-head attention and feedforward networks to stabilize and speed up training. The layer normalization is computed as:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$$

Where:
- \( \mu \) and \( \sigma \) are the mean and standard deviation of the activations.
- \( \epsilon \) is a small constant for numerical stability.
- \( \gamma \) and \( \beta \) are learned scale and shift parameters.

---

### Combining Layers

Each transformer block consists of **multi-head attention** followed by a **feedforward network**, with **residual connections** and **layer normalization** applied after both sub-layers:

Multi-head Attention:

$$
\( x = \text{LayerNorm}(x + \text{MultiHead}(Q, K, V)) \)
$$

Feedforward Network: 

$$
\( x = \text{LayerNorm}(x + \text{FFN}(x)) \)
$$

This architecture allows the transformer to capture complex dependencies between tokens in the sequence, facilitating effective sequence modeling for language tasks.

---

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

