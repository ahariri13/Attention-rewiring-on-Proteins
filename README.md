# Attention Rewiring on Proteins

This repository contains the official implementation of the paper:

> **Attention Rewiring on Proteins**  
> *[Author(s)]*, NeurIPS ML4PhysicalSciences 2024  
> [[Paper Link]](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_201.pdf)

The core idea behind this work is to introduce a novel attention-rewiring mechanism tailored for protein structures. By refining the standard attention layers, our approach aims to capture the nuanced geometric and chemical interactions that drive protein folding and function.

## Overview

Modern deep learning techniques (particularly Transformer-based architectures) excel at capturing long-range dependencies in sequence data. However, protein sequences and structures pose unique challenges due to their 3D conformation and residue interactions. This project addresses these complexities by **rewiring** the conventional attention mechanism, ensuring that attention heads respect the physical and biochemical constraints of proteins.

Key goals:
1. **Encourage biologically-plausible attention** patterns aligned with protein contact maps.  
2. **Improve performance** on protein-specific tasks such as secondary structure prediction or contact prediction.  
3. **Retain interpretability** by providing a clear mechanism to visualize and understand how attention is redirected.

---

## Features

- **Attention Rewiring Module**: A specialized layer that modifies standard attention heads to better represent residue-residue interactions.  
- **Configurable Architecture**: Easily plug in or remove the rewiring mechanism from a standard Transformer or GNN backbone.  
- **Benchmarks**: Scripts to evaluate the performance on typical protein modeling tasks (e.g., contact map prediction).  
- **Visualization Tools**: Functions/notebooks (if available) to visualize attention heatmaps and rewiring patterns.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahariri13/Attention-rewiring-on-Proteins.git
   cd Attention-rewiring-on-Proteins
   
2. **Setup environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
