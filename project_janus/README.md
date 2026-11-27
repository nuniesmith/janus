# Project JANUS - Neuromorphic Trading Intelligence

Technical paper and visualization suite for the JANUS (Joint Adaptive Neuro-symbolic Universal System) architecture.

## 📄 Core Files

- **`janus.tex`** - Complete technical specification (LaTeX source)
- **`janus.pdf`** - Generated PDF (auto-compiled via GitHub Actions)
- **`visualization_specification.md`** - Design specification for all 13 visualizations
- **`examples/`** - Python implementation of visualization suite (13/13 complete)

## 🚀 Quick Start

### Compile the Paper

The paper now includes a comprehensive bibliography with 80+ citations. To compile with references:

```bash
cd technical_papers/project_janus

# Full compilation with bibliography (recommended)
pdflatex janus.tex
biber janus          # Process bibliography with biblatex/biber
pdflatex janus.tex   # Resolve citations
pdflatex janus.tex   # Resolve cross-references

# Alternative: Use latexmk for automatic compilation
latexmk -pdf -bibtex janus.tex
```

**Note**: The document uses `biblatex` with the `biber` backend. If you don't have `biber` installed:

```bash
# On Ubuntu/Debian
sudo apt-get install biber

# On macOS with MacTeX
# biber is included in MacTeX

# On Windows with MiKTeX
# Install biber via MiKTeX Package Manager
```

Or simply push to `main` - GitHub Actions will auto-compile and commit the PDF.

### Run Visualizations

```bash
cd examples

# Install dependencies
pip install numpy matplotlib scipy scikit-learn
pip install umap-learn torch  # Optional: for UMAP and attention visuals

# Run individual visualizations
python3 visual_1_gaf_pipeline.py --show
python3 visual_7_opal_decision.py --show
python3 visual_11_umap_evolution.py --show

# Generate all outputs
for script in visual_*.py; do 
    python3 $script --save-all --output-dir ../outputs
done

# Or run the comprehensive test suite
./test_all_visuals.sh --quick
```

## 📊 Visualization Suite (13/13 Complete)

### Phenomenological (Perception Layer)
- **V1** - GAF Transformation Pipeline
- **V2** - ViViT Attention Heatmap
- **V3** - Multimodal Feature Fusion

### Internal State (Cognitive Layer)
- **V4** - LTN Grounding Graph
- **V5** - LTN Truth Surface
- **V6** - Multimodal Fusion Gate
- **V7** - OpAL Decision Pathway
- **V8** - Mahalanobis Anomaly Map

### System (Architecture & Runtime)
- **V9** - Memory Consolidation Cycle
- **V10** - Recall Gate Comparator
- **V11** - UMAP Schema Evolution
- **V12** - Runtime Topology
- **V13** - Microservices Ecosystem

All visualizations are:
- ✅ Production-ready (300 DPI output)
- ✅ WCAG 2.1 AA accessible
- ✅ Color-blind safe palettes
- ✅ Reproducible (fixed random seeds)
- ✅ Documented with inline comments

See [`visualization_specification.md`](visualization_specification.md) for detailed design requirements.

## 📖 What is JANUS?

JANUS is a neuro-symbolic trading intelligence system combining:

- **Deep Learning**: ViViT transformers, GAF image encoding, multimodal fusion
- **Symbolic Logic**: Logic Tensor Networks (LTN) for explainable reasoning
- **Neuromorphic Design**: Brain-inspired architecture (hippocampus, prefrontal cortex, amygdala)
- **Adaptive Learning**: Experience replay, schema consolidation, UMAP manifold learning

The system makes real-time trading decisions while maintaining interpretability through symbolic grounding and logical constraints.

## 🎯 Use Cases

### For Implementation (FKS Project)

The visualization suite provides reference implementations for:
- GAF normalization and encoding
- LTN grounding and fuzzy logic operations
- UMAP manifold learning and schema evolution
- Attention mechanisms and gating functions
- Anomaly detection (Mahalanobis distance)
- Memory consolidation and prioritized replay

All code is modular and can be adapted for your FKS implementation.

### For Academic Publication

The paper (`janus.tex`) and visualizations are publication-ready:
- Complete mathematical formulations
- Biological motivation and neuromorphic mapping
- Implementation details and architectural diagrams
- Performance budgets and accessibility compliance
- ~9,400 lines of reference Python code

## 📚 Repository Structure

```
technical_papers/project_janus/
├── janus.tex                           # Main LaTeX paper
├── janus.pdf                           # Auto-generated PDF
├── README.md                           # This file
├── visualization_specification.md      # Visualization design spec
└── examples/
    ├── visual_1_gaf_pipeline.py       # GAF encoding visualization
    ├── visual_2_vivit_attention.py    # Attention heatmap
    ├── visual_3_multimodal_fusion.py  # Feature fusion
    ├── visual_4_ltn_grounding.py      # Grounding graph
    ├── visual_5_ltn_truth_surface.py  # Truth surface plots
    ├── visual_6_fusion_gate.py        # Gating mechanism
    ├── visual_7_opal_decision.py      # Decision pathway
    ├── visual_8_mahalanobis_map.py    # Anomaly detection
    ├── visual_9_consolidation.py      # Memory consolidation
    ├── visual_10_recall_gate.py       # Recall gating
    ├── visual_11_umap_evolution.py    # Schema evolution
    ├── visual_12_runtime_topology.py  # Concurrency topology
    ├── visual_13_microservices.py     # System architecture
    ├── test_all_visuals.sh            # Comprehensive test suite
    ├── requirements.txt               # Python dependencies
    └── README.md                      # Examples documentation
```

## 🔧 Dependencies

### LaTeX (for paper compilation)
- Standard packages: `amsmath`, `amssymb`, `listings`, `algorithm`, `tcolorbox`, `hyperref`
- Bibliography: `biblatex` with `biber` backend
- Bibliography file: `janus.bib` (80+ sources across ML, neuroscience, finance, and systems)

### Python (for visualizations)
- **Core**: numpy, matplotlib, scipy, scikit-learn
- **Optional**: umap-learn, torch (for UMAP and attention visuals; PCA fallbacks provided)

## 📚 Bibliography

The paper includes 80+ citations organized into categories:

- **Machine Learning & Computer Vision**: GAF encoding, ViViT, Chronos, Vision Transformers
- **Neuro-Symbolic AI**: Logic Tensor Networks, Łukasiewicz logic, fuzzy reasoning
- **Neuroscience**: Complementary Learning Systems, Sharp-Wave Ripples, basal ganglia, dopamine modulation
- **Reinforcement Learning**: Prioritized Experience Replay, Feudal Networks, OpAL
- **Market Microstructure**: Almgren-Chriss execution, VPIN/flow toxicity, wash sale rules
- **Dimensionality Reduction**: UMAP, AlignedUMAP, Parametric UMAP
- **Hardware & Systems**: tch-rs, Candle, Tokio, Polars, FPGA acceleration, JAX-LOB
- **Generative Models**: Diffusion models for LOB simulation
- **Quantum Computing**: Portfolio optimization

See `janus.bib` for full bibliography.

## 🤖 CI/CD

GitHub Actions automatically:
1. Compiles `janus.tex` with full bibliography on every push to `main`
2. Uploads PDF as artifact (90-day retention)
3. Commits PDF back to repository

No manual compilation required!

## 📬 Contact

- **Author**: Jordan Smith
- **Repository**: [github.com/nuniesmith/technical_papers](https://github.com/nuniesmith/technical_papers)

## 🎓 Citation

```bibtex
@article{smith2024janus,
  title={JANUS: Joint Adaptive Neuro-symbolic Universal System for Trading Intelligence},
  author={Smith, Jordan},
  journal={Technical Papers},
  year={2024},
  url={https://github.com/nuniesmith/technical_papers}
}
```

---

*"The god of beginnings and transitions, looking simultaneously to the future and the past."*

**Status**: Paper complete ✅ | Visualizations 13/13 ✅ | Ready for FKS implementation 🚀