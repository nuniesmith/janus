# Visualizing Project JANUS: A Comprehensive Neuromorphic Design and Visualization Specification

## 1. Introduction: The Epistemological Necessity of Visualization in Neuromorphic Finance

The advent of Project JANUS represents a fundamental paradigm shift in the engineering of autonomous financial systems. We are moving away from the era of opaque "black box" algorithms—typified by deep neural networks that function as inscrutable oracles—toward "glass box" architectures rooted in biological plausibility and symbolic reasoning. This transition, however, introduces a crisis of complexity. Project JANUS is not merely a trading algorithm; it is a synthetic cognitive entity comprising bicameral processing services (Forward/Backward), multi-timescale memory systems, and neuro-symbolic logic gates. To validate, debug, and trust such a system, we must develop a rigorous visual language that transcends standard financial charting.

This document serves as the definitive research report and visual specification for Project JANUS. It integrates the architectural definitions from the core technical papers with advanced insights from computational neuroscience, manifold learning, and systems programming. The objective is to define the specific visual artifacts—diagrams, heatmaps, and topological projections—required to contextualize the system's operation for stakeholders ranging from quantitative researchers to regulatory compliance officers.

The visualization strategy is bifurcated to mirror the system's own architecture:

- **Phenomenological Visualization**: How the system perceives the market (e.g., GAF textures, Attention maps).
- **Internal State Visualization**: How the system thinks and remembers (e.g., Basal Ganglia dynamics, Memory consolidation graphs, Neocortical schemas).

By rendering the abstract mathematics of Gramian fields and Łukasiewicz logic into perceptible visual structures, we bridge the gap between high-frequency data streams and human cognitive understanding.

### 1.1. Document Structure and Intended Audience

This document serves three primary audiences:

1. **ML Researchers** (Sections 2-4): Mathematical foundations and theoretical justification for each visualization
2. **Software Engineers** (Section 5): Implementation requirements, data pipelines, and performance considerations
3. **Stakeholders** (All sections): Interpretation guides explaining what each visualization reveals about system behavior

### 1.2. Accessibility and Color Standards

All visualizations conform to **WCAG 2.1 AA standards** for accessibility:

**Color Palette Constraints:**
- Color is NEVER the sole information channel (redundant encoding via shape/texture/annotation)
- Minimum contrast ratio: 4.5:1 for all text elements
- All colormaps tested with deuteranopia and protanopia simulators

**Approved Colormaps:**

| Visualization Type | Colormap | Rationale | Accessibility |
|-------------------|----------|-----------|---------------|
| GASF (Correlation) | YlGnBu, Blues | Sequential, perceptually uniform | Deuteranopia-safe |
| GADF (Flux) | PuOr, RdYlBu | Diverging, avoids red/green confusion | Protanopia-safe |
| Anomaly Highlights | ISO Orange (#FF6600) | Universal warning color | Color-blind safe |
| Attention Weights | Viridis, Plasma | Perceptually uniform scientific standard | Full color-blind safe |
| Neural Pathways | Green (#2E8B57) / Red (#DC143C) | With shape redundancy (arrows/dashes) | Safe with redundant encoding |

**Testing Protocol:** All figures must be validated using [Coblis Color Blindness Simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/) before publication.

**Implementation Example:**
```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Deuteranopia-safe diverging colormap
SAFE_DIVERGING = LinearSegmentedColormap.from_list(
    'safe_div', 
    ['#2166AC', '#F7F7F7', '#B2182B']  # Blue-White-Orange
)

plt.imshow(gadf_matrix, cmap=SAFE_DIVERGING, vmin=-1, vmax=1)
```

## 2. The Forward Service (Janus Bifrons): Visualizing Perception and Action

The Forward Service, or "Janus Bifrons," operates as the system's "wake state," characterized by high-throughput perception, real-time constraint satisfaction, and immediate execution. The visualization challenge here is to depict how 1D time-series data is transmuted into 3D semantic understanding.

### 2.1. The Gramian Angular Field (GAF) Transformation Pipeline

In the domain of deep learning for time series, the primary challenge is encoding temporal dependencies in a format accessible to computer vision architectures. Project JANUS employs Gramian Angular Fields (GAF) to project 1D price histories into 2D spatiotemporal manifolds. This is not merely a data formatting step; it is a projection into a feature space where "market regime" manifests as visual texture.

#### 2.1.1. Theoretical Basis and Visual Construction

The GAF transformation preserves temporal correlation through a polar coordinate system. Standard Cartesian plots obscure the cyclical nature of market volatility. By wrapping the time series into a polar representation, we generate a matrix that represents the interaction between every time step and every other time step.

**Visual 1: The Polar Embedding and Gramian Matrix**

To fully contextualize this, we require a multi-panel figure illustrating the transformation flow:

**Panel A: Learnable Normalization.**

Unlike static min-max scaling, JANUS employs learnable affine transformations:

$$\tilde{x}_t = \gamma \odot \frac{x_t - \mu}{\sigma} + \beta$$

**Visual Specification:** A dual-axis plot showing the raw price series (gray) overlayed with the normalized series (blue). Critical to this visualization is the bounding box of $[-1, 1]$, representing the valid domain for the subsequent arccosine operation. Anomalies or "fat tail" events that saturate this normalization (clipping at -1 or 1) should be highlighted in red, indicating information loss or extreme volatility.

**Panel B: The Polar Clock Representation.**

The normalized value is mapped to an angle: $\phi_t = \arccos(\tilde{x}_t)$.

**Visual Specification:** A polar plot with **discrete clock structure** (not unbounded spiral):

- **Angular coordinate:** $\phi_t = \arccos(\tilde{x}_t) \in [0, \pi]$
  - Encodes normalized price value
  - 0° = maximum price, 180° = minimum price
  
- **Radial coordinate:** $r_t = t/T \in [0, 1]$
  - Normalized timestamp (bounded to unit circle)
  - Center (r=0) = start of window
  - Edge (r=1) = end of window

**Interpretation:**
- **Mean-reverting market:** Points cluster in narrow angular sector (small φ variance)
- **Trending market:** Points sweep through wide angular range (large φ variance)
- **Cyclic pattern:** Spiral exhibits rotational periodicity

**Note:** For continuous Archimedean spiral (r = a + bt), use r_t = t instead of t/T. The normalized version is preferred for fixed-size visualizations.

**Plot Specifications:**
- Type: Polar scatter/line plot
- Markers: Start point (green circle, size=100), end point (red circle, size=100)
- Background: Polar grid with 30° angle divisions
- Annotation: Dashed radial line at mean price angle

**Panel C: The Gramian Fields (GASF vs. GADF).**

The core insight of the GAF is the trigonometric sum/difference, which creates the image texture.

- **GASF (Summation)**: $\mathbf{G}_{ij} = \cos(\phi_i + \phi_j)$. This visualizes the static intensity correlations. A "Quiet" market results in a GASF with large, smooth, monochromatic patches (high correlation).
- **GADF (Difference)**: $\mathbf{G}_{ij} = \sin(\phi_i - \phi_j)$. This visualizes the flux or change. A "Volatile" market results in a GADF with high-frequency, checkerboard-like noise patterns.

**Visual Comparison:** The report must include a side-by-side comparison of a GASF and GADF generated from the same time window. The GADF should be rendered with a diverging colormap (e.g., RdBu) to highlight the directional differences, effectively serving as a visual "derivative" of the market state.

#### 2.1.2. Market Microstructure: LOB Heatmaps vs. GAF

While GAFs encode price history, they must be contextualized against the raw state of liquidity. The Limit Order Book (LOB) heatmap is the standard tool for this, but JANUS integrates both.

| Feature | Limit Order Book (LOB) Heatmap | Gramian Angular Field (GAF) |
|---------|--------------------------------|-----------------------------|
| Data Source | Level 2 Market Depth (Bids/Asks) | Level 1 Executed Price/Volume |
| Dimensionality | Price Levels vs. Time | Time vs. Time (Temporal Correlation) |
| Visual Insight | Liquidity Support/Resistance Levels | Temporal periodicity and volatility regimes |
| Machine Vision | Detects "Walls" and "Spoofing" | Detects "Trend Exhaustion" and "Regime Shift" |
| Representation | Sparse Matrix (mostly empty) | Dense Matrix (fully populated) |

**Visual 2: The Microstructure Fusion View.**

A composite figure showing the LOB heatmap (left) feeding into the fusion layer alongside the GAF texture (right). This visually demonstrates that while the LOB shows intent (where orders are resting), the GAF shows kinematics (how price is actually moving). This distinction is vital for understanding why the Multimodal Fusion layer is necessary.

#### 2.1.3. Common Issues and Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| GAF appears monochrome (all blue/single color) | Insufficient variance in time window | Increase window size W; verify σ > 0.01 |
| Diagonal stripe artifacts | Timestamp misalignment or gaps | Check for missing data; use interpolation |
| Saturation at ±1 boundaries | Normalization overflow on outliers | Use tanh() nonlinearity instead of linear clipping |
| NaN values in GAF matrix | arccos() domain error | Ensure \|x_norm\| ≤ 1 with explicit clipping: `x_norm.clamp(-1, 1)` |
| GASF and GADF look identical | Incorrect formula implementation | Verify: GASF uses cos(φ_i + φ_j), GADF uses sin(φ_i - φ_j) |

### 2.5. Visual Specification Reference Table (Forward Service)

| ID | Name | Type | Input Data | Tools/Libraries | Output Format | Update Frequency |
|----|------|------|------------|-----------------|---------------|------------------|
| V1 | GAF Pipeline | 3-panel static | Price array [T] | Matplotlib, NumPy | PNG 16×5" @ 300 DPI | On-demand |
| V2 | LOB vs GAF Comparison | Side-by-side | LOB dict + price series | Matplotlib | PNG 16×6" @ 300 DPI | Real-time (1s) |
| V3 | ViViT Factorized Attention | Heatmap grid | Attention weights [H,N,N] | PyTorch hooks, Matplotlib | PNG 18×10" @ 300 DPI | Batch (minutes) |
| V4 | LTN Grounding Graph | Bipartite network | Concept-predicate mapping | NetworkX, Graphviz | PNG 12×8" @ 300 DPI | Static (docs) |
| V5 | Łukasiewicz Truth Surface | 3D surface plot | Logic operation name | Matplotlib 3D, NumPy | PNG 16×6" @ 300 DPI | Static (docs) |
| V6 | Multimodal Fusion Gate | Schematic diagram | Gate activations [T] | Matplotlib | PNG 15×5" @ 300 DPI | Real-time (100ms) |

**Legend:**
- **Update Frequency:** Real-time (<100ms), Near-real-time (~1s), Batch (minutes), On-demand (hours), Static (documentation only)
- **Size:** Recommended figure dimensions for publication-quality output at 300 DPI

### 2.2. The Video Vision Transformer (ViViT): Spatiotemporal Factorization

Once the market data is encoded as a sequence of GAF images (a "video"), it is processed by the Video Vision Transformer (ViViT). The specific variant used—the Factorized Encoder—requires precise visualization to distinguish it from computationally expensive 3D CNNs or full-attention Transformers.

#### 2.2.1. Factorized Attention Architecture

The computational cost of attention grows quadratically with the number of tokens. For a video of resolution $H \times W$ with $T$ frames, full attention is $O((T \cdot H \cdot W)^2)$. JANUS utilizes factorization to reduce this to $O(T^2 + (H \cdot W)^2)$.

**Visual 3: The Factorized Encoder Block Diagram.**

This diagram must illustrate the decoupled flow of information:

**Tubelet Extraction:** Instead of 2D patches, the visual must show the GAF video volume being sliced into 3D "tubelets" (cuboids extending through time). This preserves local temporal continuity before the transformer even begins.

**Spatial Encoder (The "Texture" Specialist):**

- **Visual:** A Transformer block attending only to tokens within the same frame.
- **Insight:** Arrows should connect pixels $(x,y)_t$ to $(x',y')_t$, but never to $(x,y)_{t+1}$. This encoder learns the "shape" of the market at a single instant (e.g., detecting a "Head and Shoulders" pattern in the GAF texture).

**Temporal Encoder (The "Sequence" Specialist):**

- **Visual:** The output tokens (CLS) from the Spatial Encoder are stacked. A second Transformer block attends only across the time dimension.
- **Insight:** Arrows connect Frame $t$ to Frame $t+k$. This encoder learns the evolution of the market (e.g., "Texture A" transitions to "Texture B").

This visualization clarifies why JANUS is efficient enough for real-time trading: it doesn't cross-reference every pixel with every other pixel in history, but rather abstracts spatial features first, then analyzes their sequence.

#### 2.2.2. Common Issues and Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| Attention weights all uniform (~1/N) | Model not trained / frozen layers | Verify `model.train()` mode; check gradients with `torch.autograd.grad()` |
| Attention heatmap shows all zeros | Incorrect tensor extraction from hooks | Ensure hook attached to correct layer; verify `.detach()` not blocking gradients |
| Spatial and temporal attention identical | Factorization not implemented | Verify separate encoder blocks; check attention mask shapes |
| NaN values in attention weights | Numerical instability in softmax | Add epsilon to denominator: `softmax(x / sqrt(d) + 1e-8)` |
| Attention doesn't sum to 1.0 across dimension | Wrong softmax dimension | Check `dim` parameter in `torch.softmax()`; should be `-1` for last dim |
| Memory overflow during attention visualization | Full attention matrix too large | Use attention head subsampling; visualize only representative heads |
| Heatmap appears pixelated/blocky | Patch size too large | Reduce patch size `p` or use interpolation for display |
| Temporal attention shows no time dependence | CLS tokens not propagating information | Verify token aggregation between spatial and temporal encoders |

### 2.3. Logic Tensor Networks (LTN): Visualizing Differentiable Reasoning

The most novel aspect of JANUS is the integration of Logic Tensor Networks (LTN), which allow the system to treat logical constraints (e.g., "Do not wash sale") as differentiable loss functions. This moves beyond symbolic AI into "Real Logic".

#### 2.3.1. The Grounding Graph ($\mathcal{G}$)

Standard logic is discrete (True/False). LTN logic is continuous ($[0,1]$). Visualization must depict this "softening" of truth.

**Visual 4: The Bipartite Grounding Graph.**

This figure represents the mapping $\mathcal{G}: \mathcal{C} \to \mathbb{R}^d$.

- **Symbolic Layer (Top):** Nodes representing abstract concepts: $Concept: \text{Bullish}$, $Rule: \forall x (\text{Bullish}(x) \to \text{Buy}(x))$.
- **Neural Layer (Bottom):** Nodes representing tensors and networks.
  - The symbol $x$ maps to a vector embedding $\mathbf{v}_x$.
  - The predicate $\text{Bullish}(\cdot)$ maps to a Multilayer Perceptron (MLP) ending in a Sigmoid activation.

**The Fuzzy Logic Operations:** The visual must explicitly show the replacement of Boolean operators with Łukasiewicz T-norms.

- **AND** ($\land$): Visualized as a computational node performing $\max(0, u+v-1)$.
- **IMPLIES** ($\Rightarrow$): Visualized as $\min(1, 1-u+v)$.

**Visual 5: The Truth Value Surface Plot.**

To provide deep intuition into how the network learns logic, we must visualize the loss landscape of the Łukasiewicz T-norm.

**Plot:** A 3D surface plot comparing Łukasiewicz logic (differentiable) to Boolean logic (non-differentiable).

**Axes:**
- X: Truth(P) ∈ [0,1], 50 subdivisions
- Y: Truth(Q) ∈ [0,1], 50 subdivisions  
- Z: Truth(P∧Q) = max(0, P+Q-1) for Łukasiewicz

**Overlays:**
1. **Primary surface:** Łukasiewicz T-norm (smooth gradient surface, colormap: viridis)
2. **Comparison wireframe:** Boolean AND (Z = P·Q, gray wireframe)
3. **Gradient vectors:** Red arrows showing ∇Z at sample points (0.3,0.3), (0.5,0.5), (0.7,0.7)
4. **Optimizer trajectory:** Dotted line showing gradient descent path

**Viewpoint:** Azimuth 45°, Elevation 30° (standard 3D convention)

**Insight:** Unlike a boolean step function (flat plateau, vertical cliff), the Łukasiewicz surface has a continuous gradient. Visualizing this gradient "ramp" explains how the optimizer can "slide" the network parameters toward satisfying the constraint via backpropagation.

### 2.4. Multimodal Fusion: The Gated Cross-Attention

Project JANUS is not unimodal; it must reconcile the GAF video, the temporal price stream, and textual sentiment. The "Gated Cross-Attention" mechanism is the arbiter of this fusion.

**Visual 6: The Attention Gate Schematic.**

- **Mechanism:** Show three streams ($\mathbf{v}, \mathbf{t}, \mathbf{s}$) entering a "Fusion Hub."
- **The Gate:** A specific sub-network (MLP + Sigmoid) that takes the concatenated state and outputs a scalar $g \in [0,1]$.
- **The Multiplier:** This scalar $g$ element-wise multiplies the attention weights of the secondary modalities.
- **Interpretation:** If the Gate outputs $0$, the Visual and Textual inputs are suppressed. This visualizes the system's ability to "focus" or "ignore." For example, during a high-latency liquidity crunch, the Gate might close off the (slower) Textual stream to prioritize Order Book latency, a dynamic rebalancing of cognitive resources.

## 3. The Decision Engine: Neuromorphic Basal Ganglia Dynamics

Moving from perception to action, the JANUS Decision Engine mimics the Basal Ganglia's direct and indirect pathways. This biological fidelity provides robust action selection that simple "ArgMax" classifiers cannot match.

### 3.1. The OpAL / TD2Q Circuitry

The Opponent Actor Learning (OpAL) and TD2Q models posit that the brain learns "Benefits" (G) and "Costs" (N) separately.

#### 3.1.1. Dual-Pathway Architecture

**Visual 7: The Direct/Indirect Pathway Flowchart.**

This is the central diagram for the Decision Engine. It must depict the bifurcation of the state signal $\mathbf{s}_t$.

**Direct Pathway (Go / Green):**

- Represents D1-type Striatal Projection Neurons (SPNs).
- Function: Potentiates action. $\mathbf{d}_{direct} = \text{ReLU}(\mathbf{W}_d \mathbf{h})$.
- **Visual:** A thick green arrow originating from the Cortex (Input) and terminating at the Thalamus (Output).

**Indirect Pathway (No-Go / Red):**

- Represents D2-type SPNs.
- Function: Suppresses action. $\mathbf{d}_{indirect} = \text{ReLU}(\mathbf{W}_i \mathbf{h})$.
- **Visual:** A thick red arrow that passes through an intermediate node (the Subthalamic Nucleus/GPe) before inhibiting the Thalamus.

**Dopamine Modulation (RPE):**

- **Visual:** A node representing the Substantia Nigra pars compacta (SNc) releasing Dopamine ($\delta$).
- **Crucial Detail:** The diagram must show opposing plasticity rules.
  - Positive RPE ($\delta > 0$): Strengthens Direct ($\uparrow G$), Weakens Indirect ($\downarrow N$).
  - Negative RPE ($\delta < 0$): Weakens Direct ($\downarrow G$), Strengthens Indirect ($\uparrow N$).

This visualization is essential to explain why JANUS can handle volatility. In a high-volatility environment (negative RPEs), the Indirect pathway (N-matrix) builds up charge, effectively acting as a biological "circuit breaker" that suppresses trading even if a buy signal is technically present.

#### 3.1.2. Common Issues and Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| G = N = 0 (both pathways inactive) | No gradients flowing / dead neurons | Check weight initialization; verify ReLU not saturating at zero; use LeakyReLU |
| G and N are identical (pathways not differentiated) | Shared weights between pathways | Verify separate weight matrices W_d and W_i; check model architecture |
| Decision always HOLD (stuck in neutral) | Thresholds too wide (θ_buy, θ_sell) | Reduce threshold magnitude; typical values: θ ∈ [0.1, 0.3] |
| Decision oscillates rapidly (Buy→Sell→Buy) | No temporal smoothing | Add exponential moving average to G and N; use momentum term |
| Dopamine signal δ always zero | RPE not computed correctly | Verify TD error calculation: δ = r + γV(s') - V(s) |
| Indirect pathway never activates (N ≈ 0) | Negative RPEs not strengthening N-weights | Check plasticity rule signs: negative δ should increase W_i |
| Circuit diagram arrows don't match activations | Visualization not synced with model state | Ensure visualization reads from actual model tensors, not hardcoded values |
| Historical scatter shows no separation by outcome | Decision boundaries poorly calibrated | Retrain model with balanced reward signal; adjust threshold hyperparameters |

#### 3.1.3. Decision Space Visualization

**Panel B (NEW): Decision Boundary Plot**

To complement the circuit diagram, add a quantitative decision space visualization:

**Plot Type:** 2D scatter with shaded decision regions

**Axes:**
- X: Direct pathway activation (G) ∈ [0, 1]
- Y: Indirect pathway activation (N) ∈ [0, 1]

**Decision Regions (shaded):**
- **BUY zone:** {(G,N) | G - N > θ_buy = 0.2} → Light green (#90EE90, α=0.3)
- **SELL zone:** {(G,N) | G - N < θ_sell = -0.2} → Light red (#FFB6C1, α=0.3)
- **HOLD zone:** {(G,N) | |G - N| ≤ 0.2} → Light gray (#D3D3D3, α=0.3)

**Overlays:**
- Decision boundaries: Green dashed line (G - N = 0.2), red dashed line (G - N = -0.2)
- Historical states: Scatter plot of last 1000 (G,N) tuples, colored by outcome (profit=green, loss=red, hold=gray)
- Current state: Large blue star marker (size=200) with black outline (linewidth=2)

**Annotation:** Text box displaying current decision: "BUY" / "SELL" / "HOLD" in corresponding color

**Interpretation:** Clustering of historical points reveals calibration quality. Well-calibrated system shows profitable trades concentrated in decision zones, not scattered near boundaries.

### 3.3. Visual Specification Reference Table (Decision Engine)

| ID | Name | Type | Input Data | Tools/Libraries | Output Format | Update Frequency |
|----|------|------|------------|-----------------|---------------|------------------|
| V7 | OpAL Dual Pathways | 2-panel: Circuit + Decision Space | G, N, δ scalars + history | Matplotlib, patches | PNG 16×8" @ 300 DPI | Real-time (100ms) |
| V8 | Mahalanobis Ellipsoid | 2-panel comparison | 2D data + covariance | Matplotlib, SciPy | PNG 14×6" @ 300 DPI | Batch (5min) |

**Legend:** Same as Section 2.5

### 3.2. The Amygdala: Mahalanobis Distance and Risk

While the Basal Ganglia handles routine decisions, the "Amygdala" module handles existential threats via anomaly detection.

**Visual 8: The Mahalanobis Ellipsoid.**

To explain why Euclidean distance is insufficient for financial risk, we must visualize the Mahalanobis distance.

#### 3.2.1. Mahalanobis Distance Visualization

**Plot:** A 2D scatter plot of two correlated variables (e.g., Spread vs. Volatility). The "Normal" regime forms a diagonal cloud.

- **Euclidean Fallacy:** A circle centered on the mean. Show a point inside the circle but outside the natural correlation cloud (an anomaly). Euclidean distance calls this "Safe."
- **Mahalanobis Insight:** An ellipse aligned with the covariance of the cloud. This ellipse correctly excludes the anomalous point.
- **Formula Annotation:** $D_M(\mathbf{s}_t) = \sqrt{(\mathbf{s}_t - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{s}_t - \boldsymbol{\mu})}$. This visual proves that JANUS understands correlation structure, not just magnitude. A widening spread is normal during high volatility, but a widening spread during low volatility is a crisis signal—only Mahalanobis distance captures this.

#### 3.2.2. Common Issues and Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| Ellipse appears as circle (no orientation) | Covariance matrix is identity / diagonal | Verify using correlated features; check data has been normalized independently |
| Singular matrix error when computing Σ⁻¹ | Insufficient data or perfect collinearity | Use regularization: Σ_reg = Σ + λI (λ = 1e-6); or remove redundant features |
| Ellipse axes reversed (major/minor swapped) | Eigenvalue ordering incorrect | Sort eigenvalues descending; ensure eigenvectors correspond to sorted order |
| All points flagged as anomalies | Threshold too strict | Increase chi-squared threshold; use 99% confidence (χ²₀.₉₉) instead of 95% |
| No anomalies detected despite obvious outliers | Covariance contaminated by outliers | Use robust covariance estimation (Minimum Covariance Determinant) |
| Ellipse doesn't align with data cloud | Running statistics not converged | Increase warmup period before computing Σ; use larger batch size |
| Mahalanobis distance negative or complex | Covariance matrix not positive definite | Check for numerical errors; ensure Σ is symmetric; add jitter to diagonal |
| Distance comparison misleading | Feature scales vastly different | Standardize features before computing distances (mean=0, std=1) |

## 4. The Backward Service (Janus Consivius): Visualizing Memory and Sleep

The "Sleep" state of JANUS is where data is transmuted into wisdom. This involves the transfer of memory from the fast-learning Hippocampus to the slow-learning Neocortex.

### 4.1. The Three-Timescale Memory Hierarchy

Standard RL uses a simple replay buffer. JANUS uses a bio-mimetic hierarchy.

**Visual 9: The Consolidation Cycle.**

A cyclic diagram showing the flow of information across timescales.

| Memory Component | Biological Analogue | Computational Structure | Function | Visual Representation |
|------------------|---------------------|------------------------|----------|----------------------|
| Episodic Buffer | Hippocampus (CA3) | Circular Buffer (FIFO) | Fast recording of raw experience $(s_t, a_t, r_t)$ | Small, rapidly rotating ring |
| SWR Simulator | Sharp-Wave Ripples | Prioritized Sampler | Selection of "high error" events for replay | Bursts/Spikes transferring data |
| Neocortical Schema | Neocortex | Vector Database (Qdrant) | Slow integration of generalized patterns | Large, stable network/grid |

**Visual Detail: Prioritized Experience Replay (PER) Distribution.**

Within this diagram, we must visualize the sampling probability distribution of the SWR Simulator.

**Plot:** A histogram of the replay probabilities $P(i) \propto |\delta_i|^\alpha$.

**Shape:** It should be a "Heavy-Tailed" Power Law distribution. This visual confirms that the system spends the majority of its "sleep" processing the rare, high-surprise events (black swans) rather than the bulk of routine data. This aligns with biological theories that dreams are threat simulations.

### 4.2. Recall-Gated Consolidation

The transfer from Hippocampus to Neocortex is not automatic; it is gated.

**Visual 10: The Recall Gate.**

**Concept:** The Neocortex only updates its weights if the new memory is consistent with (or provides a useful update to) existing schemas.

**Equation:** $\mathbf{z}_k \leftarrow \mathbf{z}_k + \eta \cdot \mathbb{1}[\text{recall} > \theta] \cdot (\mathbf{h}_{new} - \mathbf{z}_k)$.

**Diagram:** A comparator node. The "New Memory" and the "Reconstructed Memory" (from the Schema) are compared. If the cosine similarity is below threshold $\theta$ (too alien/noise), the gate remains closed. This visualization explains the system's resistance to "catastrophic forgetting" and noise overfitting.

### 4.3. UMAP and the Cognitive Dashboard

Finally, the "state of mind" of the system is visualized using Uniform Manifold Approximation and Projection (UMAP).

**Visual 11: The Schema Manifold Evolution.**

**Input:** The high-dimensional vectors stored in the Neocortical Qdrant database.

**Projection:** UMAP projects these to 2D.

**Temporal Evolution:** A series of three scatter plots showing the manifold at $T=0$ (Random initialization), $T=100$ (Emerging clusters), and $T=1000$ (Converged Schemas).

**Insight:** Points should be colored by market regime. The visual success condition is seeing distinct, separated islands of "Bullish", "Bearish", and "Choppy" schemas. This provides the "Explainability" layer—we can literally see what the AI knows.

#### 4.3.1. Common Issues and Troubleshooting

| Symptom | Probable Cause | Fix |
|---------|----------------|-----|
| All points collapse to single cluster | n_neighbors too high or min_dist too large | Reduce n_neighbors (try 5-15); reduce min_dist (try 0.01-0.1) |
| Projection is extremely distorted (stretching) | Incompatible high-D and 2D topologies | Increase n_components to 3D; check trustworthiness metric T(k) < 0.6 |
| Different runs produce completely different layouts | Random initialization dominates | Set random_state=42; increase n_epochs; use init='spectral' |
| Clusters overlap despite being distinct in high-D | UMAP parameters don't preserve global structure | Increase n_neighbors for more global view; reduce min_dist |
| Runtime extremely slow (>10 minutes) | Too many points for exact UMAP | Use approximate UMAP with low_memory=True; or sample data |
| "Disconnected graph" error | Data has isolated points / outliers | Remove outliers before UMAP; or use metric='cosine' instead of 'euclidean' |
| Manifold doesn't evolve over time (T=0 looks like T=1000) | Model not actually learning | Verify training loss decreasing; check weight updates; inspect schema diversity |
| Color labels don't match clusters | Labeling error or regime detection failure | Manually inspect embeddings; use unsupervised clustering (HDBSCAN) to verify |

#### 4.3.2. UMAP Quality Metrics

Raw UMAP projections can be misleading without validation. We measure:

**1. Trustworthiness T(k)**
- **Definition:** Proportion of k-nearest neighbors preserved from high-D to 2D
- **Formula:** T(k) = (1/N) Σᵢ |NN_highD(i,k) ∩ NN_2D(i,k)| / k
- **Thresholds:** T(15) > 0.8 (good), > 0.6 (acceptable), < 0.6 (reject)
- **Visualization:** Line plot of T(k) for k ∈ {5, 10, 20, 50}

**2. Continuity C(k)**
- **Definition:** Proportion of 2D neighbors that are valid in high-D
- **Purpose:** Detects "false neighbors" introduced by projection

**3. Distortion Heatmap**
- **Metric:** For each point, compute ratio_i = mean_j(d_2D(i,j) / d_highD(i,j))
- **Visualization:** Overlay on UMAP scatter plot
  - Blue points: ratio ≈ 1 (faithful projection)
  - Red points: ratio >> 1 (inflated distance, unreliable)
- **Action threshold:** If >20% of points are red, increase `n_neighbors` parameter

**Visual Specification:** Create companion 2-panel figure:
- **Left:** UMAP with distortion heatmap overlay (colorbar showing ratio)
- **Right:** Trustworthiness curve T(k) vs k with threshold lines

### 4.4. Visual Specification Reference Table (Backward Service)

| ID | Name | Type | Input Data | Tools/Libraries | Output Format | Update Frequency |
|----|------|------|------------|-----------------|---------------|------------------|
| V9 | Memory Consolidation Cycle | Flow diagram + histogram | Experience buffer, PER weights | Matplotlib, NetworkX | PNG 14×8" @ 300 DPI | Batch (1hr) |
| V10 | Recall Gate Comparator | Block diagram | Similarity scores [T] | Matplotlib | PNG 10×6" @ 300 DPI | Batch (1hr) |
| V11 | UMAP Schema Evolution | 4-panel time series | Embeddings [N,D] at T=[0,100,500,1000] | UMAP, Matplotlib | PNG 14×12" @ 300 DPI | Batch (daily) |
| V11b | UMAP Quality Metrics | 2-panel: Heatmap + Curve | High-D and 2D embeddings | Scikit-learn, Matplotlib | PNG 14×6" @ 300 DPI | Batch (daily) |

**Legend:** Same as Section 2.5

## 5. The Rust Implementation: Visualizing Concurrency

The choice of Rust is strategic, leveraging its ownership model for safety and its dual runtime capability for performance.

### 5.1. The "Two Minds" of Rust: Tokio vs. Rayon

JANUS employs a unique "Async Sandwich" architecture which requires specific visualization to guide developers.

**Visual 12: The Runtime Topology.**

This diagram contrasts the concurrency models of the Forward and Backward services.

**Forward Service (Tokio - The "Wait" Engine):**

- **Visual:** An Event Loop (Reactor) pattern.
- **Flow:** Many tasks are spawned, but they are mostly "Hollow" bars (awaiting I/O). The single thread hops rapidly between them.
- **Context:** Optimized for high-frequency WebSocket data where latency comes from the network, not the CPU.

**Backward Service (Rayon - The "Think" Engine):**

- **Visual:** A Thread Pool pattern (Work Stealing).
- **Flow:** All cores are 100% utilized with "Solid" bars (CPU compute). The tasks (UMAP projection, Gradient calculation) are computationally heavy.
- **Context:** Optimized for throughput. The visual must show that blocking the Tokio reactor with these tasks would catastrophic (the "Async Sandwich" anti-pattern), justifying the architectural separation.

### 5.2. Architecture Diagram and Data Flow

**Visual 13: The Microservices Ecosystem.**

A high-level C4 container diagram illustrating the physical deployment.

- **Python Gateway:** PyTorch/HuggingFace for initial model training and ONNX export.
- **Forward Pod (Rust):** Loads model.onnx. Ingests Market Data (TCP). Outputs Orders (Fix) and Experiences (Redis/RingBuffer).
- **Backward Pod (Rust):** Consumes Experiences. Runs SWR/Replay. Updates Qdrant DB.
- **Qdrant:** Persistent Vector Storage.
- **Feedback Loop:** The Backward Pod periodically pushes updated Schema Vectors back to the Forward Pod (Hot Reload), closing the learning loop.

### 5.3. Performance Requirements and Computational Budget

| Visual | Update Frequency | Max Latency (p99) | Compute Tier | Implementation | CPU/GPU |
|--------|------------------|-------------------|--------------|----------------|---------|
| V2 (LOB Heatmap) | 100ms | 50ms | **Tier 1** (Real-time) | Rust + WebGL shader | GPU |
| V6 (Gate Values) | 100ms | 50ms | **Tier 1** (Real-time) | Rust incremental plot | CPU |
| V1 (GAF Transform) | 1s | 500ms | **Tier 2** (Near-real-time) | Rust compute, Canvas2D | CPU |
| V3 (Attention Maps) | 1s | 800ms | **Tier 2** (Near-real-time) | PyTorch hooks, cached | GPU |
| V7 (OpAL Circuit) | 1s | 500ms | **Tier 2** (Near-real-time) | Rust, pre-rendered templates | CPU |
| V11 (UMAP Update) | 60s | 30s | **Tier 3** (Batch) | Python/UMAP, Matplotlib | CPU |
| V8 (Mahalanobis) | 300s | - | **Tier 3** (Batch) | NumPy/SciPy offline | CPU |
| V9 (Memory Flow) | 3600s | - | **Tier 3** (Batch) | NetworkX static | CPU |
| V5 (Truth Surface) | On-demand | - | **Tier 4** (Static) | Jupyter notebook | CPU |
| V4 (Grounding Graph) | On-demand | - | **Tier 4** (Static) | Graphviz static | CPU |

**Tier Definitions:**
- **Tier 1 (Real-time):** Required for live trading; Rust-only; GPU-accelerated rendering where applicable
- **Tier 2 (Near-real-time):** Dashboard monitoring; Mixed Rust/Python; CPU rendering acceptable
- **Tier 3 (Batch):** Nightly/hourly consolidation; Python/NumPy; No interactivity required
- **Tier 4 (Static):** Documentation/exploration; Any language; Print-quality only

**Monitoring and Alerts:**
- Use `cargo flamegraph` for Rust profiling (Tier 1-2)
- Use `cProfile` for Python bottlenecks (Tier 2-3)
- **Alert threshold:** If p99 latency > 2× budget, page on-call engineer
- **Fallback behavior:** If Tier 1 visualization exceeds budget, fall back to text-only display

### 5.4. Visual Specification Reference Table (Rust Implementation)

| ID | Name | Type | Input Data | Tools/Libraries | Output Format | Update Frequency |
|----|------|------|------------|-----------------|---------------|------------------|
| V12 | Runtime Topology | 2-panel comparison | Task traces from Tokio Console | Custom SVG generator | SVG diagram | Static (docs) |
| V13 | Microservices Ecosystem | C4 container diagram | Deployment config | PlantUML, Graphviz | PNG 16×10" @ 300 DPI | Static (docs) |

**Legend:** Same as Section 2.5

## 6. Validation Methodology

To ensure visualizations are effective, accurate, and meet their intended purpose, we employ a multi-tiered validation protocol combining quantitative metrics, user testing, and expert review.

### 6.1. Quantitative Performance Metrics

Each visualization must meet measurable performance criteria:

#### 6.1.1. Task Completion Time (TCT)

**Definition:** Median time for a trained user to complete a representative task using the visualization.

**Measurement Protocol:**
1. Recruit 10 users familiar with trading concepts
2. Provide 5-minute training on visualization interpretation
3. Present 10 test scenarios per visualization
4. Record time from presentation to decision

**Success Criteria:**

| Visualization | Task | Target TCT | Baseline (Traditional) |
|---------------|------|------------|------------------------|
| V1 (GAF) | Identify regime change | < 10s | 15s (candlestick) |
| V7 (OpAL) | Predict next action | < 5s | 8s (simple indicator) |
| V11 (UMAP) | Assess learning progress | < 30s | 60s (loss curves) |
| V8 (Mahalanobis) | Detect anomaly | < 8s | 12s (z-score) |

**Data Collection:**
```python
import time

def measure_tct(user_id, visualization_id, scenario_id):
    display_scenario(scenario_id)
    start = time.time()
    decision = get_user_decision()
    elapsed = time.time() - start
    
    log_result({
        'user': user_id,
        'visual': visualization_id,
        'scenario': scenario_id,
        'time': elapsed,
        'decision': decision,
        'timestamp': datetime.now()
    })
    return elapsed
```

#### 6.1.2. Error Rate

**Definition:** Percentage of incorrect interpretations or decisions made using the visualization.

**Measurement:**
- Present ground-truth scenarios with known outcomes
- Compare user interpretation to ground truth
- Calculate: Error Rate = (Incorrect / Total) × 100%

**Success Criteria:**
- **Excellent:** Error rate < 5%
- **Acceptable:** Error rate < 10%
- **Needs improvement:** Error rate ≥ 10%

**Example Test Cases:**

```python
test_cases = [
    {
        'visual': 'V1_GAF',
        'scenario': 'Trending market with regime change at t=50',
        'ground_truth': {'regime_change': True, 'timestep': 50},
        'difficulty': 'medium'
    },
    {
        'visual': 'V7_OpAL',
        'scenario': 'High volatility triggering circuit breaker',
        'ground_truth': {'action': 'HOLD', 'reason': 'indirect_pathway'},
        'difficulty': 'hard'
    }
]
```

#### 6.1.3. Subjective Usability Scale (SUS)

**Definition:** Standardized 10-question survey measuring perceived usability.

**Questionnaire (5-point Likert scale: Strongly Disagree to Strongly Agree):**

1. I think that I would like to use this visualization frequently
2. I found the visualization unnecessarily complex
3. I thought the visualization was easy to use
4. I think that I would need technical support to use this visualization
5. I found the various functions in this visualization well integrated
6. I thought there was too much inconsistency in this visualization
7. I would imagine that most people would learn to use this very quickly
8. I found the visualization very cumbersome to use
9. I felt very confident using the visualization
10. I needed to learn a lot of things before I could use this effectively

**Scoring:**
- Odd questions: contribution = (score - 1)
- Even questions: contribution = (5 - score)
- SUS Score = sum(contributions) × 2.5

**Interpretation:**
- **Excellent:** SUS ≥ 85
- **Good:** SUS ≥ 70
- **Acceptable:** SUS ≥ 50
- **Poor:** SUS < 50

### 6.2. A/B Testing Framework

Compare JANUS visualizations against traditional alternatives to validate superiority claims.

#### 6.2.1. GAF vs. Candlestick for Regime Detection

**Hypothesis:** GAF textures enable faster and more accurate regime change detection than candlestick charts.

**Experimental Design:**
- **Participants:** 20 quantitative traders (10 per condition)
- **Materials:** 20 market sequences (10 with regime change, 10 without)
- **Procedure:**
  - Group A: Identify regime changes using GAF visualizations
  - Group B: Identify regime changes using candlestick charts
  - Randomize sequence order to control for learning effects

**Measured Variables:**
1. **Time to decision** (seconds)
2. **Accuracy** (% correct identifications)
3. **Confidence** (1-5 self-report)
4. **False positive rate** (incorrectly flagging regime change)
5. **False negative rate** (missing actual regime change)

**Statistical Analysis:**
```python
from scipy import stats

# Two-sample t-test for time comparison
gaf_times = [8.2, 7.5, 9.1, ...]  # Group A times
candlestick_times = [12.3, 14.1, 11.8, ...]  # Group B times

t_stat, p_value = stats.ttest_ind(gaf_times, candlestick_times)

# Success criteria: p < 0.05 AND mean(GAF) < mean(Candlestick)
if p_value < 0.05 and np.mean(gaf_times) < np.mean(candlestick_times):
    print("GAF is significantly faster")
```

**Success Criteria:**
- GAF accuracy ≥ Candlestick accuracy (non-inferiority)
- GAF time ≤ 0.8 × Candlestick time (20% improvement)
- GAF confidence ≥ Candlestick confidence

#### 6.2.2. OpAL Circuit vs. Simple Indicator

**Hypothesis:** OpAL dual-pathway visualization provides better insight into decision rationale than simple buy/sell/hold indicators.

**Metrics:**
- **Explainability score:** Can user explain WHY system made decision? (0-10)
- **Trust calibration:** User confidence matches actual system performance
- **Debugging speed:** Time to identify why system made error

**Control Condition:** Simple traffic light (green=buy, red=sell, yellow=hold)

### 6.3. Expert Review Protocol

Each visualization undergoes review by domain experts before deployment.

#### 6.3.1. Neuroscience Validation

**Reviewer Qualifications:** PhD in neuroscience with computational modeling experience

**Checklist:**
- [ ] Biological analogies are accurate (not misleading)
- [ ] Brain region mappings are appropriate
- [ ] Terminology is used correctly (e.g., "dopamine modulation")
- [ ] Plasticity rules match known biology
- [ ] No oversimplification that creates misconceptions

**Example Feedback Form:**
```markdown
## Neuroscience Review: Visual 7 (OpAL Decision Pathway)

**Biological Accuracy:** 9/10
- Direct/Indirect pathway distinction: Correct ✓
- Dopamine RPE signaling: Correct ✓
- Minor issue: SNc also projects to cortex, not shown

**Recommendations:**
- Add disclaimer: "Simplified model for illustration"
- Consider showing cortical dopamine projections (optional)

**Approval:** ✅ Approved with minor recommendations
```

#### 6.3.2. Trading Domain Validation

**Reviewer Qualifications:** 5+ years experience in quantitative trading

**Checklist:**
- [ ] Visualization provides actionable insights
- [ ] Information density is appropriate (not overwhelming)
- [ ] Critical information is immediately visible
- [ ] Latency requirements are realistic for live trading
- [ ] Visualization wouldn't cause decision paralysis

#### 6.3.3. HCI/Visualization Expert Review

**Reviewer Qualifications:** Published research in information visualization

**Checklist:**
- [ ] Visual encoding follows best practices (position > color > size)
- [ ] Color palettes are perceptually uniform
- [ ] Gestalt principles applied correctly (proximity, similarity)
- [ ] No chart junk or unnecessary decoration
- [ ] Accessibility standards met (WCAG 2.1 AA)
- [ ] Interactive elements have clear affordances

### 6.4. Automated Validation Tests

Programmatic tests to catch regressions and ensure consistency.

#### 6.4.1. Visual Regression Testing

**Tool:** Pytest + pytest-mpl (Matplotlib comparison)

```python
import pytest
import matplotlib.pyplot as plt
from visual_5_ltn_truth_surface import visualize_truth_surface

@pytest.mark.mpl_image_compare(baseline_dir='baseline_images',
                                tolerance=5)
def test_truth_surface_and():
    """Ensure AND surface hasn't changed unexpectedly."""
    fig = visualize_truth_surface(operation='and', 
                                   show_gradients=True)
    return fig

@pytest.mark.mpl_image_compare(tolerance=5)
def test_truth_surface_or():
    """Ensure OR surface matches baseline."""
    fig = visualize_truth_surface(operation='or')
    return fig
```

**Baseline Generation:**
```bash
# Generate baseline images (one time)
pytest --mpl-generate-path=baseline_images test_visualizations.py

# Compare against baseline (in CI/CD)
pytest --mpl test_visualizations.py
```

#### 6.4.2. Colormap Accessibility Validation

**Automated Check:** Ensure all colormaps pass deuteranopia/protanopia simulation

```python
from colorspacious import cspace_convert

def validate_colormap_accessibility(cmap, name):
    """Check if colormap is distinguishable under color blindness."""
    
    # Sample colormap
    colors_rgb = cmap(np.linspace(0, 1, 10))[:, :3]
    
    # Simulate deuteranopia
    colors_deuter = cspace_convert(colors_rgb, "sRGB1", "sRGB1+CVD",
                                   CVD_type='deuteranomaly',
                                   severity=100)
    
    # Check if adjacent colors are distinguishable
    min_diff = float('inf')
    for i in range(len(colors_deuter) - 1):
        diff = np.linalg.norm(colors_deuter[i] - colors_deuter[i+1])
        min_diff = min(min_diff, diff)
    
    # Threshold for distinguishability
    threshold = 0.1  # Perceptual difference
    
    assert min_diff > threshold, \
        f"{name} fails deuteranopia test: min_diff={min_diff:.3f}"
    
    return True

# Test all approved colormaps
def test_all_colormaps():
    approved = ['YlGnBu', 'viridis', 'plasma']
    for name in approved:
        cmap = plt.cm.get_cmap(name)
        validate_colormap_accessibility(cmap, name)
```

#### 6.4.3. Performance Budget Validation

**Automated Benchmark:** Ensure visualizations meet latency requirements

```python
import time
import pytest

def test_gaf_transform_latency():
    """V1 GAF must compute in <500ms (Tier 2)."""
    price_series = np.random.randn(100)
    
    start = time.time()
    gaf_matrix = compute_gaf(price_series)
    elapsed = time.time() - start
    
    assert elapsed < 0.5, \
        f"GAF transform too slow: {elapsed:.3f}s > 0.5s budget"

def test_opal_visualization_latency():
    """V7 OpAL must render in <500ms (Tier 2)."""
    G, N, delta = 0.7, 0.3, 0.1
    
    start = time.time()
    fig = visualize_opal_circuit(G, N, delta)
    plt.savefig('/tmp/test.png')
    elapsed = time.time() - start
    
    assert elapsed < 0.5, \
        f"OpAL rendering too slow: {elapsed:.3f}s"
```

### 6.5. Iterative Refinement Process

Visualizations are not static; they evolve based on validation results.

**Process Flow:**
1. **Initial Implementation** (based on specification)
2. **Internal Review** (team members)
3. **Automated Tests** (regression, accessibility, performance)
4. **Expert Review** (neuroscience, trading, HCI)
5. **User Testing** (TCT, error rate, SUS)
6. **A/B Testing** (vs. baseline alternatives)
7. **Analysis & Iteration** (address failures)
8. **Re-validation** (repeat steps 3-6)
9. **Deployment Approval**

**Minimum Passing Criteria for Deployment:**
- [ ] All automated tests pass (100%)
- [ ] Expert reviews: Average score ≥ 8/10
- [ ] SUS score ≥ 70 (good usability)
- [ ] Error rate < 10%
- [ ] Performance within budget (latency)
- [ ] A/B test: Non-inferior to baseline (p < 0.05)

### 6.6. Continuous Monitoring (Post-Deployment)

**Production Metrics:**
- **Usage frequency:** How often is visualization accessed?
- **Dwell time:** How long do users examine it?
- **Click-through rate:** Do users interact with elements?
- **Correlation with outcomes:** Does using visualization improve trading performance?

**Feedback Loop:**
```python
class VisualizationMetrics:
    def log_view(self, user_id, visual_id, duration_ms):
        """Track visualization usage."""
        self.db.insert({
            'user': user_id,
            'visual': visual_id,
            'duration': duration_ms,
            'timestamp': datetime.now()
        })
    
    def analyze_effectiveness(self, visual_id):
        """Correlate visualization use with performance."""
        
        # Get users who frequently use this visualization
        heavy_users = self.get_users_by_usage(visual_id, 
                                               percentile=75)
        
        # Get users who rarely use it
        light_users = self.get_users_by_usage(visual_id,
                                               percentile=25)
        
        # Compare trading performance
        heavy_perf = self.get_trading_returns(heavy_users)
        light_perf = self.get_trading_returns(light_users)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(heavy_perf, light_perf)
        
        return {
            'heavy_mean': np.mean(heavy_perf),
            'light_mean': np.mean(light_perf),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

### 6.7. Validation Documentation Template

For each visualization, maintain a validation record:

```markdown
# Validation Report: V5 (Łukasiewicz Truth Surface)

## Test Date: 2024-12-28
## Version: 1.0.0

### Automated Tests
- [x] Visual regression: PASS
- [x] Colormap accessibility: PASS (viridis)
- [x] Performance budget: PASS (renders in 2.3s, budget=N/A for Tier 4)

### Expert Reviews
- Neuroscience (Dr. Smith): 9/10 - "Excellent pedagogical tool"
- HCI (Dr. Johnson): 8/10 - "Clear 3D rendering, good gradient overlay"

### User Testing (n=10 users)
- Task: "Explain why Łukasiewicz AND is differentiable"
- Success rate: 90% (9/10 correct explanations)
- Average time: 3.2 minutes
- SUS Score: 82 (Good)

### A/B Test: vs. 2D contour plot
- Understanding score: 8.1 vs 6.3 (p=0.02) ✓
- Preference: 80% prefer 3D surface

### Deployment Decision: ✅ APPROVED

### Recommendations:
- Consider adding interactive rotation in web version
- Add caption explaining gradient vectors
```

## 7. Conclusion

The visualization of Project JANUS is not a supplementary exercise but a core requirement for its comprehensibility and validation. By rigorously defining the visual representations of GAF textures, OpAL decision pathways, and Recall-Gated memory flows, we transform the system from a collection of equations into a coherent, observable synthetic organism. This report provides the blueprint for generating these artifacts, ensuring that the visual narrative of the project is as robust as its code.

The integration of these specific visualizations—particularly the heavy-tailed PER distribution and the Mahalanobis risk ellipsoid—directly addresses the "extra context" required by the original query, grounding the high-level claims of "Neuromorphic Intelligence" in demonstrable, graphical reality.

---

## References and Citations

**Core Visualization Foundations:**

1. **Wang, Z., & Oates, T.** (2015). "Imaging time-series to improve classification and imputation." *Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 3939-3945.

2. **Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C.** (2021). "ViViT: A Video Vision Transformer." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 6836-6846.

3. **McInnes, L., Healy, J., & Melville, J.** (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv preprint arXiv:1802.03426*.

**Neuromorphic Architectures:**

4. **Collins, A. G., & Frank, M. J.** (2014). "Opponent actor learning (OpAL): Modeling interactive effects of striatal dopamine on reinforcement learning and choice incentive." *Psychological Review*, 121(3), 337-366.

5. **Badreddine, S., d'Avila Garcez, A., Serafini, L., & Spranger, M.** (2022). "Logic Tensor Networks." *Artificial Intelligence*, 303, 103649.

**Fuzzy Logic and Reasoning:**

6. **Hájek, P.** (1998). *Metamathematics of Fuzzy Logic*. Springer, Dordrecht.

**Anomaly Detection:**

7. **Mahalanobis, P. C.** (1936). "On the generalized distance in statistics." *Proceedings of the National Institute of Sciences of India*, 2(1), 49-55.

**Memory Systems:**

8. **Schaul, T., Quan, J., Antonoglou, I., & Silver, D.** (2015). "Prioritized Experience Replay." *arXiv preprint arXiv:1511.05952*.

9. **Kumaran, D., Hassabis, D., & McClelland, J. L.** (2016). "What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated." *Trends in Cognitive Sciences*, 20(7), 512-534.

**For complete technical and architectural details, refer to:**
- Project JANUS Technical Specification (main paper): `janus.tex`

---

## Appendix A: Implementation Checklist

Before implementing any visualization, verify:

- [ ] Colormap is color-blind safe (tested with Coblis)
- [ ] Text contrast ratio ≥ 4.5:1 (WCAG AA)
- [ ] Figure has descriptive alt-text for accessibility
- [ ] Performance budget is specified and achievable
- [ ] Failure modes are documented
- [ ] Sample output is generated and validated
- [ ] Code is type-hinted and documented
- [ ] Unit tests cover edge cases (NaN, empty data, saturation)

---

## Appendix B: Quick Reference - All Visual Specifications

**Total Visualizations:** 13 primary + 1 quality metric companion

**By Update Frequency:**
- Real-time (Tier 1): V2, V6 (2 visuals)
- Near-real-time (Tier 2): V1, V3, V7 (3 visuals)
- Batch (Tier 3): V8, V9, V10, V11, V11b (5 visuals)
- Static/On-demand (Tier 4): V4, V5, V12, V13 (4 visuals)

**By System Component:**
- Forward Service: V1-V6 (6 visuals)
- Decision Engine: V7-V8 (2 visuals)
- Backward Service: V9-V11b (4 visuals)
- Infrastructure: V12-V13 (2 visuals)

**Critical Path (Minimum Viable Dashboard):**
- V2 (LOB Heatmap)
- V6 (Gate Values)
- V7 (OpAL Decision)
- V11 (UMAP Evolution)

**Priority Implementation Order:**
1. V5 (Truth Surface) - Easiest, pedagogical value
2. V1 (GAF Pipeline) - Core transformation
3. V7 (OpAL Circuit) - Decision transparency
4. V11 (UMAP) - Learning validation
5. Remainder as needed