<div align="center">
  <h1>🌌 POLYDIM V3</h1>
  <p><b>A Topological Execution Engine to Bypass the KV-Cache Wall via Clifford Algebras and Hodge Laplacians</b></p>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)]()
  [![Architecture: Hybrid](https://img.shields.io/badge/Architecture-PyTorch%20%7C%20Rust-success.svg)]()
</div>

---

## 🛑 The Core Problem: The KV-Cache Wall

State-of-the-art Large Language Models (LLMs) rely on the classical Transformer architecture. While extremely expressive, they suffer from a fundamental mathematical bottleneck: **Dense Attention**.

In a standard Transformer, every token must attend to every other token, resulting in $\mathcal{O}(N^2)$ time complexity and a massive Memory Wall (The KV-Cache) that grows linearly $\mathcal{O}(N \cdot d)$ with sequence length. At 1M+ tokens, the memory required to maintain the context state exceeds the VRAM of most GPU clusters.

State Space Models (SSMs) like Mamba solve the memory issue by maintaining a fixed-size $\mathcal{O}(1)$ state, but their diagonal linear updates destroy the angular, geometric relationships between distant concepts.

## 📐 The Polydim Solution

**POLYDIM V3** is not just another LLM; it is a **Topological Execution Engine**. It replaces the statistical guessing game of Dense Attention with strict Geometric Computability.

By mapping sequence topologies to **Simplicial Complexes** and running **Chebyshev Spectral Filters** over **Sparse Hodge Laplacians**, POLYDIM achieves global consensus in $\mathcal{O}(N \cdot k)$ time (where $k \ll N$), entirely bypassing the $N^2$ attention matrix. 

Furthermore, memory state is updated via **Clifford Algebra Rotors** (Householder reflections), which strictly preserve the $\mathcal{L}_2$ norm (100% isometry).

### Architectural Paradigm Shift

```mermaid
graph TD
    subgraph "Classical Transformer (O(N²))"
        A[Input Tokens] --> B[Dense Q, K, V Projections]
        B --> C[Softmax Attention Matrix]
        C -->|Memory Explodes| D[KV Cache Storage]
        C --> E[MLP Layers]
    end

    subgraph "POLYDIM V3 (O(N·k))"
        F[Input Tokens] --> G[JL Lemma Hashing: GEO_ID]
        G --> H[Simplicial Complex Topology]
        H --> I[Hodge Laplacian Consensus]
        I -->|O(D) Constant State| J[Clifford Rotors Isometric Routing]
        J --> K[Statistical Feature MLP]
    end
```

## 🧠 The 5-Layer Ontology (The Polydim Stack)

POLYDIM V3 guarantees theoretical rigor by adhering strictly to a 5-layer ontology:
1. **Layer 1 - Foundations (Pure Math):** Category Theory, HoTT, Clifford Algebra, Sheaf Theory.
2. **Layer 2 - Mathematical Theory:** The abstract dynamic system over the Stiefel Manifold.
3. **Layer 3 - Execution Semantics:** Isometric State Transitions and Shadow Loss Entropy.
4. **Layer 4 - Runtime & Protocol:** The Hybrid Engine (PyTorch) and the PMTP Network Protocol (Rust).
5. **Layer 5 - Applications:** Autonomous Agents, Swarm Intelligence, Long-Context AI.

## 🚀 The Hybrid Engine (V3 Core)

To merge rigorous geometry with GPU hardware efficiency, the POLYDIM V3 Engine separates concerns into two distinct spaces:

1. **Statistical Space (Learning):** Uses PyTorch MLPs (SwiGLU) and Autograd to extract features and learn representations. Information is intentionally compressed here.
2. **Topological Space (Routing):** Information is projected onto a $10,000D$ hypersphere. Here, no learning occurs; data is routed across agents/tokens using strictly isometric Clifford rotations to guarantee zero semantic degradation over infinite context lengths.

## 📦 PMTP: Polydim Matrix Transfer Protocol

When Polydim agents communicate, they do not send human-readable text (which suffers from catastrophic entropy collapse, known as *Shadow Loss*). Instead, they transmit their internal geometric states using **PMTP**, a Rust-based networking protocol that serializes topologies into Block-Sparse Row (BSR) tensors, cutting network payload sizes by $\sim 99\%$ compared to dense matrices.

---

### 📚 Documentation & Deep Dive
For the complete mathematical theorems, empirical proofs, and LaSalle/Cheeger convergence bounds, please refer to the foundational document:
- [The Polydim White Book](docs/for_evaluation/POLYDIM_WHITE_BOOK.md)

---
*“Geometry is not a heuristc. It is destiny.”*
