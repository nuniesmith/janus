# JANUS Whitepaper Update Plan
## Goal: Comprehensive Theory Document for the JANUS Codebase (Academic Peer Review)

---

## Current State Assessment

Your paper is 1,720 lines of LaTeX organized into 5 Parts with 51 bib entries (13 of which are "Author, Various" placeholders). It's already strong on technical implementation detail — the math is clean, the brain-region mapping is well-structured, and the Rust implementation sections are thorough. What's missing for an academic audience are the **theoretical grounding**, **literature positioning**, **crowding immunity argument**, and **honest engagement with limitations**.

---

## Structural Changes: New Document Outline

The current 5-Part structure should expand to **7 Parts**. The new parts are marked with ★.

```
PART I    — Philosophical & Theoretical Foundations (EXPAND existing Overview)
  ★ 1.1   The Epistemological Transition to Quant 4.0 (exists, tighten)
  ★ 1.2   NEW: Resilience to Strategy Crowding and Co-Impact
  ★ 1.3   NEW: Neuroscience-Inspired AI — The Doya-Hassabis Framework
  ★ 1.4   The Dual-Process Architecture (exists, add citations)

PART II   — Forward Service (Janus Bifrons) — EXISTS, minor updates

PART III  — Backward Service (Janus Consivius) — EXISTS, minor updates

PART IV   — Neuromorphic Architecture — EXISTS, significant expansion

PART V    — Rust Implementation — EXISTS, minor updates

PART VI ★ — NEW: Limitations, Open Problems, and Reviewer Preemptions

PART VII ★ — NEW: Validation Framework and Experimental Design
```

---

## Phase 1: Fix the Bibliography (Do This First)

This is the lowest-effort, highest-impact change. You have **13 placeholder entries** that will immediately disqualify the paper in peer review. Here's every one and what to replace it with:

### Placeholders to Fix

| Current Key | Current "Author, Various" | Replace With |
|---|---|---|
| `fusion2025gaf` | GAF validation, Pattern Recognition 2025 | Chen & Tsai (2020) "Encoding candlesticks as images..." *Financial Innovation* 6:26, DOI: 10.1186/s40854-020-00187-0 — or find the actual 2025 paper you intended |
| `garcez2024mapping` | Neural Networks 2024 | Garcez & Lamb (2023) "Neurosymbolic AI: The 3rd Wave" *AI Review*; or Marra et al. (2024) "From Statistical Relational to Neurosymbolic AI" *Artificial Intelligence* 328:104062 |
| `dopamine2020reward` | Nature Neuroscience 2020 | Schultz, Dayan & Montague (1997) "A Neural Substrate of Prediction and Reward" *Science* 275:1593–1599; or Masset et al. (2025) "Multi-timescale RL in the brain" *Nature* 642:682–690 |
| `thalamus2018attention` | J. Neuroscience 2018 | Halassa & Kastner (2017) "Thalamic functions in distributed cognitive control" *Nature Neuroscience* 20:1669–1679 (already have `halassa2017thalamic` key — may be duplicate) |
| `substantia2016connections` | Neural Computation 2024 | Check what this actually cites — likely should be Caligiore et al. (2019) "The super-learning hypothesis" *Neuroscience & Biobehavioral Reviews* 100:19–34 |
| `amygdala2019fear` | IEEE TNNLS 2019 | LeDoux (2000) "Emotion Circuits in the Brain" *Annual Review of Neuroscience* 23:155–184; or Monfils et al. (2009) "Extinction-Reconsolidation Boundaries" *Science* 324:951–955 (you already cite `monfils2009extinction`) |
| `kar2023selection` | Nature 2016 | This key doesn't match "Nature 2016" — likely Doya (1999) "What are the computations of the cerebellum, the basal ganglia and the cerebral cortex?" *Neural Networks* 12:961–974 |
| `diffusion2024lob` | arXiv 2024 | Find the actual paper. If it's about diffusion models for LOB data, search for specific authors. |
| `gatedattention2023` | CVPR 2023 | Likely Hua et al. (2022) or similar — needs specific paper. If about gated cross-attention, cite Alayrac et al. (2022) "Flamingo" *NeurIPS* |
| `quantum2024portfolio` | Science Advances 2024 | Specific QAOA/VQE finance paper needed — check what you actually implemented from |
| `alignedumap2023` | UMAP docs | McInnes et al. (2018) already cited. For AlignedUMAP specifically, cite the UMAP documentation as software: `@software{mcinnes2018umap_software, ...}` |
| `tradfi2023hft` | tradfi.com 2023 | Remove or replace with an actual academic source |
| `candle2024` | Internal science link | Cite as software: `@software{candle2024, title={Candle: Minimalist ML Framework for Rust}, author={Hugging Face}, year={2024}, url={https://github.com/huggingface/candle}}` |

### New References to Add (from research)

These are verified, high-quality citations you should add to `janus.bib`:

```bibtex
% === CO-IMPACT AND CROWDING ===

@article{bucci2020coimpact,
  author={Bucci, Fr{\'e}d{\'e}ric and Mastromatteo, Iacopo and Eisler, Zolt{\'a}n
          and Lillo, Fabrizio and Bouchaud, Jean-Philippe and Lehalle, Charles-Albert},
  title={Co-impact: Crowding effects in institutional trading activity},
  journal={Quantitative Finance},
  volume={20},
  number={2},
  pages={193--205},
  year={2020},
  doi={10.1080/14697688.2019.1660398}
}

@article{vankralingen2021crowded,
  author={van Kralingen, Marc and Garlaschelli, Diego and Scholtus, Karolina
          and van Lelyveld, Iman},
  title={Crowded Trades, Market Clustering, and Price Instability},
  journal={Entropy},
  volume={23},
  number={3},
  pages={336},
  year={2021},
  doi={10.3390/e23030336}
}

@article{stillman2024neurosymbolic,
  author={Stillman, Namid R. and Baggott, Rory},
  title={Neuro-Symbolic Traders: Assessing the Wisdom of AI Crowds in Markets},
  journal={arXiv preprint arXiv:2410.14587},
  year={2024},
  doi={10.48550/arXiv.2410.14587}
}

@article{khandani2011quants,
  author={Khandani, Amir E. and Lo, Andrew W.},
  title={What Happened to the Quants in August 2007?},
  journal={Journal of Financial Markets},
  volume={14},
  number={1},
  pages={1--46},
  year={2011},
  doi={10.1016/j.finmar.2010.07.005}
}

@article{stein2009crowding,
  author={Stein, Jeremy C.},
  title={Presidential Address: Sophisticated Investors and Market Efficiency},
  journal={Journal of Finance},
  volume={64},
  number={4},
  pages={1517--1548},
  year={2009},
  doi={10.1111/j.1540-6261.2009.01472.x}
}

@article{wagner2011diversity,
  author={Wagner, Wolf},
  title={Systemic Liquidation Risk and the Diversity--Diversification Trade-Off},
  journal={Journal of Finance},
  volume={66},
  number={4},
  pages={1141--1175},
  year={2011},
  doi={10.1111/j.1540-6261.2011.01666.x}
}

@article{demiguel2021crowding,
  author={DeMiguel, Victor and Martin-Utrera, Alberto and Uppal, Raman},
  title={What Alleviates Crowding in Factor Investing?},
  journal={CEPR Discussion Paper 16527},
  year={2021}
}

@article{baltas2019arp,
  author={Baltas, Nick},
  title={The Impact of Crowding in Alternative Risk Premia Investing},
  journal={Financial Analysts Journal},
  volume={75},
  number={3},
  pages={50--65},
  year={2019},
  doi={10.1080/0015198X.2019.1600955}
}

@article{volpati2020zooming,
  author={Volpati, Valerio and Benzaquen, Michael and Eisler, Zolt{\'a}n
          and Mastromatteo, Iacopo and T{\'o}th, Bence and Bouchaud, Jean-Philippe},
  title={Zooming In on Equity Factor Crowding},
  journal={arXiv preprint arXiv:2001.04185},
  year={2020}
}

% === SQUARE-ROOT IMPACT LAW ===

@article{toth2011anomalous,
  author={T{\'o}th, Bence and Lempi{\'e}ri{\`e}re, Yves and Deremble, Cyril
          and de Lataillade, Joachim and Kockelkoren, Julien
          and Bouchaud, Jean-Philippe},
  title={Anomalous Price Impact and the Critical Nature of Liquidity
         in Financial Markets},
  journal={Physical Review X},
  volume={1},
  number={2},
  pages={021006},
  year={2011},
  doi={10.1103/PhysRevX.1.021006}
}

@book{bouchaud2018trades,
  author={Bouchaud, Jean-Philippe and Bonart, Julius and Donier, Jonathan
          and Gould, Martin},
  title={Trades, Quotes and Prices: Financial Markets Under the Microscope},
  publisher={Cambridge University Press},
  year={2018}
}

% === NEUROSCIENCE FOUNDATIONS ===

@article{hassabis2017neuroscience,
  author={Hassabis, Demis and Kumaran, Dharshan and Summerfield, Christopher
          and Botvinick, Matthew},
  title={Neuroscience-Inspired Artificial Intelligence},
  journal={Neuron},
  volume={95},
  number={2},
  pages={245--258},
  year={2017},
  doi={10.1016/j.neuron.2017.06.011}
}

@article{doya1999computations,
  author={Doya, Kenji},
  title={What are the computations of the cerebellum, the basal ganglia
         and the cerebral cortex?},
  journal={Neural Networks},
  volume={12},
  number={7--8},
  pages={961--974},
  year={1999},
  doi={10.1016/S0893-6080(99)00046-5}
}

@article{doya2002metalearning,
  author={Doya, Kenji},
  title={Metalearning and neuromodulation},
  journal={Neural Networks},
  volume={15},
  number={4--6},
  pages={495--506},
  year={2002},
  doi={10.1016/S0893-6080(02)00044-8}
}

@article{jaskir2023opal,
  author={Jaskir, Adam and Frank, Michael J.},
  title={On the normative advantages of dopamine and striatal opponency
         for learning and choice},
  journal={eLife},
  volume={12},
  pages={e85107},
  year={2023}
}

@article{schultz1997reward,
  author={Schultz, Wolfram and Dayan, Peter and Montague, P. Read},
  title={A Neural Substrate of Prediction and Reward},
  journal={Science},
  volume={275},
  number={5306},
  pages={1593--1599},
  year={1997},
  doi={10.1126/science.275.5306.1593}
}

@article{masset2025multitimescale,
  author={Masset, Paul and others},
  title={Multi-timescale reinforcement learning in the brain},
  journal={Nature},
  volume={642},
  pages={682--690},
  year={2025},
  doi={10.1038/s41586-025-08929-9}
}

@article{wilson1994replay,
  author={Wilson, Matthew A. and McNaughton, Bruce L.},
  title={Reactivation of Hippocampal Ensemble Memories During Sleep},
  journal={Science},
  volume={265},
  number={5172},
  pages={676--679},
  year={1994},
  doi={10.1126/science.8036517}
}

@article{kumaran2016cls,
  author={Kumaran, Dharshan and Hassabis, Demis and McClelland, James L.},
  title={What Learning Systems Do Intelligent Agents Need?
         Complementary Learning Systems Theory Updated},
  journal={Trends in Cognitive Sciences},
  volume={20},
  number={7},
  pages={512--534},
  year={2016},
  doi={10.1016/j.tics.2016.05.004}
}

@article{wolpert1998cerebellum,
  author={Wolpert, Daniel M. and Miall, R. Chris and Kawato, Mitsuo},
  title={Internal models in the cerebellum},
  journal={Trends in Cognitive Sciences},
  volume={2},
  number={9},
  pages={338--347},
  year={1998}
}

@article{caligiore2019superlearning,
  author={Caligiore, Daniele and Arbib, Michael A. and Miall, R. Chris
          and Baldassarre, Gianluca},
  title={The super-learning hypothesis: Integrating learning processes
         across cortex, cerebellum and basal ganglia},
  journal={Neuroscience \& Biobehavioral Reviews},
  volume={100},
  pages={19--34},
  year={2019}
}

% === ADAPTIVE MARKETS ===

@article{lo2004adaptive,
  author={Lo, Andrew W.},
  title={The Adaptive Markets Hypothesis},
  journal={Journal of Portfolio Management},
  volume={30},
  number={5},
  pages={15--29},
  year={2004},
  doi={10.3905/jpm.2004.442611}
}

% === KAHNEMAN (you have it, but add the original Prospect Theory) ===

@article{kahneman1979prospect,
  author={Kahneman, Daniel and Tversky, Amos},
  title={Prospect Theory: An Analysis of Decision under Risk},
  journal={Econometrica},
  volume={47},
  number={2},
  pages={263--291},
  year={1979},
  doi={10.2307/1914185}
}

% === BACKTEST OVERFITTING ===

@article{bailey2015backtest,
  author={Bailey, David H. and Borwein, Jonathan M.
          and L{\'o}pez de Prado, Marcos and Zhu, Qiji Jim},
  title={The Probability of Backtest Overfitting},
  journal={Journal of Computational Finance},
  year={2015}
}

% === LTN SCALABILITY ===

@article{wan2024ltn_efficiency,
  author={Wan, Zishen and others},
  title={Towards Efficient Neuro-Symbolic AI: From Workload
         Characterization to Hardware Architecture},
  journal={IEEE Trans. CASAI},
  year={2024}
}

% === NEUROMORPHIC FINANCE (competing work) ===

@article{mohan2025snn_portfolio,
  author={Mohan, A. and others},
  title={Spiking Neural Network for Cross-Market Portfolio Optimization},
  journal={arXiv preprint arXiv:2510.15921},
  year={2025}
}

@article{bi2025financial_connectome,
  author={Bi, Yingjia and Calhoun, Vince D.},
  title={The Financial Connectome: A Brain-Inspired Framework
         for Modeling Latent Market Dynamics},
  journal={arXiv preprint arXiv:2508.02012},
  year={2025}
}
```

---

## Phase 2: Expand Part I — Theoretical Foundations

This is the biggest content addition. Your current Part I is ~40 lines. For academic peer review, it should be **~800–1200 lines** covering the "why" before the "what."

### Section 1.1: Quant 4.0 (Tighten Existing)

Your existing timeline is fine but needs citations for Quant 3.0 claims. Add:
- Cite Lo (2004) Adaptive Markets Hypothesis as the intellectual bridge between EMH and your approach
- Cite Hassabis et al. (2017) as the programmatic statement for neuroscience-inspired AI
- Frame JANUS as the convergence: brain-inspired AI (Hassabis) + adaptive markets (Lo) + neuro-symbolic reasoning (Garcez)

### Section 1.2: NEW — Resilience to Strategy Crowding and Co-Impact

This is the section your notes describe. Structure it as:

**1.2.1 The Problem: Co-Impact and Algorithmic Herding**
- Define co-impact using Bucci et al. (2020): N identical Q/N trades produce same impact as one Q trade
- Cite the 2007 quant quake (Khandani & Lo 2011) as empirical proof
- Cite Stein (2009) for the theoretical coordination problem
- Cite van Kralingen et al. (2021) for market clustering → tail risk

**1.2.2 Why Current Systems Are Vulnerable**
- Static rules, shared foundation models → homogeneous order flow
- ARP crowding evidence: Baltas (2019), Volpati et al. (2020)
- Even diverse AI agents can herd: Stillman & Baggott (2024)

**1.2.3 JANUS's Defense: Engineered Heterogeneity**
- Each deployment is an idiosyncratic agent
- Wagner (2011): rational agents should choose heterogeneous portfolios
- DeMiguel et al. (2021): trading diversification increases capacity by 45%
- Include the heterogeneity table from your notes (thalamus, hypothalamus, basal ganglia, prefrontal, hippocampus, cerebellum)
- **IMPORTANT: Use "crowding resistance" not "crowding immunity"** — the literature doesn't support full immunity

**1.2.4 Quantitative Framework**
- Define expected cross-correlation metric
- Reference your LOB simulator for validation (defer details to Part VII)

### Section 1.3: NEW — The Doya-Hassabis Framework

This section establishes the neuroscience foundation. It should explain **why** mapping brain regions to trading subsystems is scientifically grounded, not just metaphorical.

**1.3.1 Functional Brain-Region Decomposition**
- Doya (1999): cerebellum = supervised learning, basal ganglia = RL, cortex = unsupervised learning
- Doya (2002): neuromodulators regulate meta-parameters
- Caligiore et al. (2019): the "super-learning hypothesis" for multi-region integration

**1.3.2 From Neuroscience to Architecture**
- Hassabis et al. (2017): neuroscience as inspiration, not simulation
- Explicitly state: JANUS uses *functional* mappings, not biological simulation
- This is the preemptive defense against the "brain metaphor" critique

**1.3.3 Complementary Learning Systems**
- McClelland et al. (1995) and Kumaran et al. (2016) CLS 2.0
- Map to your Forward/Backward split
- Masset et al. (2025) multi-timescale RL as biological validation

### Section 1.4: Dual-Process Architecture (Expand Existing)

- Add Kahneman & Tversky (1979) Prospect Theory
- Add Evans & Stanovich (2013) for the modern dual-process debate
- Explicitly note: **no prior work operationalizes dual-process as a computational trading architecture** — this is a novelty claim

---

## Phase 3: Expand Part IV — Neuromorphic Architecture

Your current Part IV (lines 845–1085) has the right structure but each brain region subsection needs:

1. **Neuroscience citation** — the original paper establishing what that region does
2. **Why this mapping** — a sentence connecting the neuroscience function to the trading function
3. **What's novel** — explicitly state if this is the first application to trading

### Per-Region Citation Additions

| Region | Add Citation |
|---|---|
| Visual Cortex | Dosovitskiy (2021) ViT, Arnab (2021) ViViT — already cited, fine |
| Cortex | Doya (1999) for unsupervised learning role |
| Hippocampus | Buzsáki (2015) SWR review, Wilson & McNaughton (1994) replay discovery |
| Thalamus | Halassa & Kastner (2017), George et al. (2025) thalamic microcircuits |
| Hypothalamus | Sterling (2012) allostasis — already cited, fine |
| Basal Ganglia | Collins & Frank (2014) OpAL — already cited. Add Jaskir & Frank (2023) OpAL*. Add Schultz et al. (1997) for dopamine = RPE |
| Prefrontal | Badreddine et al. (2022) LTN — already cited. Note: first LTN application to finance |
| Amygdala | LeDoux (2000) fear circuits |
| Cerebellum | Wolpert et al. (1998) forward models, Caligiore et al. (2019) |

### New Subsection: 4.X — Comparison with Existing Neuromorphic Finance Systems

Add a subsection explicitly comparing JANUS to the only existing work:
- Mohan et al. (2025): SNNs for portfolio optimization — uses spiking neurons but no brain-region mapping
- Bi & Calhoun (2025): Financial Connectome — uses brain connectivity analysis methods on market data, not brain-region functional decomposition
- Ezinwoke & Rhodes (2025): SNNs for HFT price prediction — pure SNN, no multi-region architecture

**Conclusion:** No published system maps multiple brain regions to distinct trading subsystems. JANUS is the first.

---

## Phase 4: NEW Part VI — Limitations and Open Problems

This is **critical for academic credibility**. A paper that doesn't acknowledge its limitations will be rejected. Structure:

### 6.1 The Brain Metaphor Critique
- Acknowledge that every generation projects its technology onto the brain (Cobb 2021)
- Counter with Hassabis et al. (2017): the goal is functional inspiration, not biological fidelity
- Cite Caligiore et al. (2019) as validation that multi-region integration is scientifically productive

### 6.2 Integration Complexity
- Glasmachers (2017): end-to-end training of multi-module systems can fail
- Acknowledge the coupling challenge across 10 brain regions
- Describe your mitigation: modular training, brain wiring pipeline, gRPC service boundaries

### 6.3 Non-Stationarity and Regime Shifts
- Financial RL in non-stationary environments is fundamentally hard (Padakandla et al. 2020)
- Acknowledge: JANUS's regime detector is an ensemble heuristic, not a solved problem
- The three-timescale memory helps but doesn't guarantee adaptation to novel regimes

### 6.4 Backtest Overfitting Risk
- Bailey et al. (2015) CSCV framework
- López de Prado (2018): "ML will always find a pattern, even if there is none"
- Describe your planned walk-forward validation protocol

### 6.5 Crowding Resistance Is Not Crowding Immunity
- Khandani & Lo (2011): independently developed strategies still converged in 2007
- Heterogeneity is necessary but possibly insufficient in tail events
- Acknowledge: empirical validation needed with multi-agent LOB simulations

### 6.6 LTN Scalability
- Wan et al. (2024): LTN workloads create hardware utilization challenges
- Acknowledge: as axiom count grows, LTN inference latency may conflict with hot-path requirements
- Describe your mitigation: dual-mode LTN (strict/approximate)

### 6.7 Adversarial Robustness
- Goldblum et al. (2021): adversarial attacks on ML trading systems
- Acknowledge: JANUS's neural components are vulnerable to adversarial order flow
- Describe planned defenses: anomaly detection, FNI-RL fear network

---

## Phase 5: NEW Part VII — Validation Framework

For academic peer review, you need to describe (even if not yet executed) a rigorous experimental protocol.

### 7.1 Multi-Agent LOB Simulation Protocol
- Use your existing LOB simulator
- Condition A: 50–200 identical JANUS instances → measure co-impact
- Condition B: 50–200 personalized instances → measure cross-correlation
- Metrics: return kurtosis, net signed volume vs. price move, √Q law preservation

### 7.2 Walk-Forward Backtesting
- Combinatorially Symmetric Cross-Validation (Bailey et al. 2015)
- Out-of-sample Sharpe ratio degradation analysis
- Regime-conditional performance breakdown

### 7.3 Ablation Studies
- Brain region ablation: remove each region and measure performance degradation
- This directly demonstrates the value of multi-region integration
- Memory tier ablation: CLS vs. single-tier replay

### 7.4 Benchmark Comparisons
- Against standard RL baselines (PPO, SAC on the same market data)
- Against rule-based systems (your Quant 1.0/2.0 comparisons)
- Against single-brain-region systems (e.g., pure basal ganglia RL without cerebellar timing)

---

## Phase 6: Polish and Consistency Fixes

### Title Update
Current: "Project JANUS: Complete Technical Specification"
Proposed: "Project JANUS: A Neuromorphic Architecture for Crowding-Resistant Autonomous Trading"

### Abstract
You currently have no document-level abstract. Add one (~250 words) on page 2 before the TOC.

### Notation Table
Add a notation/symbols table after the abstract. With 10 brain regions, LTN operators, and impact models, reviewers need a reference.

### Consistent Terminology
- Use "crowding resistance" everywhere, never "immunity"
- Use "neuromorphic" consistently (some sections say "brain-inspired," others "neuromorphic")
- Standardize on "Forward Process" / "Backward Process" or "Forward Service" / "Backward Service" — pick one

### "Already implemented since initial specification" Block
The massive paragraph at line 1692 listing everything you've built is impressive but **does not belong in an academic paper**. Convert it into a proper "Implementation Status" appendix table with columns: Component | Status | Lines of Code | Key Dependencies.

---

## Recommended Execution Order

| Priority | Task | Effort | Impact |
|---|---|---|---|
| 1 | Fix all 13 "Author, Various" bib entries | 1–2 hours | Prevents immediate rejection |
| 2 | Add new bib entries (copy the block above) | 30 min | Enables all new sections |
| 3 | Write Section 1.2 (Crowding Resistance) | 3–4 hours | The new commercial + academic differentiator |
| 4 | Write Section 1.3 (Doya-Hassabis Framework) | 2–3 hours | Theoretical foundation for entire paper |
| 5 | Write Part VI (Limitations) | 3–4 hours | Academic credibility — reviewers check this first |
| 6 | Expand Part IV brain region citations | 2 hours | Strengthens the core architecture argument |
| 7 | Write Part VII (Validation Framework) | 2–3 hours | Shows you know how to test your claims |
| 8 | Add abstract and notation table | 1 hour | Standard academic paper requirements |
| 9 | Refactor the conclusion "already implemented" block | 1 hour | Professionalism |
| 10 | Title update + terminology pass | 1 hour | Final polish |

**Total estimated effort: 16–22 hours of writing across ~2 weeks.**

---

## What I Can Help With Next

I can draft any of these sections as LaTeX ready to paste into your `.tex` file. The highest-value items I'd recommend tackling first:

1. **The new bib entries** — I can produce a complete updated `janus.bib` with all placeholders replaced and new entries added
2. **Section 1.2 (Crowding Resistance)** — ready-to-insert LaTeX with the table, equations, and citations
3. **Section 1.3 (Doya-Hassabis Framework)** — the theoretical backbone
4. **Part VI (Limitations)** — the section that will most impress reviewers
