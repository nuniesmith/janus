# Academic research foundations for the JANUS neuromorphic trading system

**Project JANUS occupies a genuinely novel position in the literature: no existing system maps multiple brain regions to distinct trading subsystems.** The research below covers 100+ verified citations across co-impact/crowding, neuromorphic finance, neuro-symbolic AI, and critical gap analysis. Three core citations (Bucci et al., van Kralingen et al., Stillman & Baggott) have been validated with full bibliographic details. The most significant finding for positioning JANUS is that while individual neuroscience concepts (experience replay, dopamine-modulated RL, dual-process theory) have been applied piecemeal in AI, **no published system integrates basal ganglia decision-making, hippocampal replay, cerebellar timing, thalamic gating, and hypothalamic risk regulation into a unified trading architecture**. This represents the core novelty claim, but it also carries substantial reviewer risk around integration complexity and the "brain metaphor" critique.

---

## 1. Co-impact and strategy crowding: verified core citations

### Bucci et al. — Co-Impact (VERIFIED)
- **Authors:** Frédéric Bucci, Iacopo Mastromatteo, Zoltán Eisler, Fabrizio Lillo, Jean-Philippe Bouchaud, Charles-Albert Lehalle
- **Title:** "Co-impact: Crowding effects in institutional trading activity"
- **Journal:** *Quantitative Finance*, Vol. 20, No. 2, pp. 193–205 (2020)
- **arXiv:** 1804.09565 (April 2018 preprint)
- **DOI:** 10.1080/14697688.2019.1660398

This paper establishes how simultaneous institutional metaorders interact through net order flow. The market responds to aggregate flow without distinguishing individual metaorders. The square-root impact law survives in the presence of many simultaneous metaorders, but co-impact introduces a **finite intercept I₀** that grows with sign correlation — meaning crowded, correlated trading amplifies impact costs beyond what single-agent models predict. This is the theoretical backbone for JANUS's crowding-immunity claim.

### van Kralingen et al. — Crowded Trades (VERIFIED)
- **Authors:** Marc van Kralingen, Diego Garlaschelli, Karolina Scholtus, Iman van Lelyveld
- **Title:** "Crowded Trades, Market Clustering, and Price Instability"
- **Journal:** *Entropy*, Vol. 23, No. 3, 336 (2021)
- **arXiv:** 2002.03319 (February 2020)
- **DOI:** 10.3390/e23030336

Proposes a market clustering measure capturing trading overlap among investors, benchmarked against a maximum-entropy null model. **Market clustering has a causal effect on tail risk**, even after controlling for standard risk drivers. Reduced investor pool diversity negatively affects stock price stability — directly supporting JANUS's heterogeneity rationale.

### Stillman & Baggott — Neuro-Symbolic Traders (VERIFIED)
- **Authors:** Namid R. Stillman, Rory Baggott
- **Title:** "Neuro-Symbolic Traders: Assessing the Wisdom of AI Crowds in Markets"
- **arXiv:** 2410.14587 (October 2024)
- **DOI:** 10.48550/arXiv.2410.14587

Develops neuro-symbolic agents using vision-language models to discover SDE models of fundamental value. Groups of homogeneous neuro-symbolic traders in virtual markets produce **price suppression**, highlighting AI crowd risks. This paper directly supports JANUS's thesis that diverse, heterogeneous AI agents are necessary for market stability.

---

## 2. The square-root impact law and its modification by co-impact

The square-root impact law — stating that the price impact of a metaorder scales as the square root of its volume fraction — is one of the most robust empirical regularities in market microstructure. JANUS's crowding framework builds on how co-impact modifies this law.

**Foundational square-root law papers:**

- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J., & Bouchaud, J.-P. (2011). "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets." *Physical Review X*, 1(2), 021006. arXiv: 1105.1694. Proposes the ε-intelligence model showing that latent liquidity vanishes around the current price, providing the theoretical foundation.

- Farmer, J.D., Gerig, A., Lillo, F., & Waelbroeck, H. (2013). "How Efficiency Shapes Market Impact." *Quantitative Finance*, 13(11), 1743–1758. Derives the square-root law from information-efficiency arguments as an alternative derivation.

- Bucci, F., Benzaquen, M., Lillo, F., & Bouchaud, J.-P. (2019). "Crossover from Linear to Square-Root Market Impact." *Physical Review Letters*, 122(10), 108302. arXiv: 1811.05230. Explains the crossover from linear (single-trade) to square-root (metaorder) impact as a function of participation rate.

- Donier, J., Bonart, J., Mastromatteo, I., & Bouchaud, J.-P. (2015). "A Fully Consistent, Minimal Model for Non-Linear Market Impact." *Quantitative Finance*, 15(7), 1109–1121. Produces the square-root law from first principles via dynamic supply-demand theory.

- Mastromatteo, I., Tóth, B., & Bouchaud, J.-P. (2014). "Agent-based models for latent liquidity and concave price impact." *Physical Review E*, 89(4), 042805. arXiv: 1311.6262.

- Bouchaud, J.-P., Farmer, J.D., & Lillo, F. (2009). "How Markets Slowly Digest Changes in Supply and Demand." In *Handbook of Financial Markets: Dynamics and Evolution*, Elsevier, pp. 57–160. arXiv: 0809.0822.

- Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). *Trades, Quotes and Prices: Financial Markets Under the Microscope*. Cambridge University Press. The comprehensive textbook treatment including Chapter 12 on co-impact.

---

## 3. Strategy crowding, the 2007 quant quake, and ARP capacity decay

### The 2007 Quant Meltdown
The canonical case study for strategy crowding is August 2007:

- **Khandani, A.E. & Lo, A.W.** (2007/2011). "What Happened to the Quants in August 2007?: Evidence from Factors and Transactions Data." *Journal of Financial Markets*, 14(1), 1–46. DOI: 10.1016/j.finmar.2010.07.005. NBER Working Paper 14465. Documents how coordinated deleveraging of similarly constructed long/short equity portfolios caused temporary market dislocation. Quant funds that developed models independently were suddenly strongly correlated — **the defining empirical demonstration of crowding-induced systemic risk**.

### Theoretical foundations of crowding

- **Stein, J.C.** (2009). "Presidential Address: Sophisticated Investors and Market Efficiency." *Journal of Finance*, 64(4), 1517–1548. DOI: 10.1111/j.1540-6261.2009.01472.x. The seminal theoretical formalization: individual traders cannot know how many others pursue the same strategy, creating a coordination problem that generates negative externalities through crowded-trade effects and leverage decisions.

- **Kyle, A.S., Obizhaeva, A.A., & Wang, Y.** (2025, forthcoming). "Trading in Crowded Markets." *Journal of Financial and Quantitative Analysis*. Theoretical model showing that when correlation arises from information content, traders trade less aggressively and markets are less liquid.

### Crowding measurement and empirical evidence

- **Lou, D. & Polk, C.** (2022). "Comomentum: Inferring Arbitrage Activity from Return Correlations." *Review of Financial Studies*, 35(7), 3272–3302. DOI: 10.1093/rfs/hhab117. Proposes "comomentum" — high-frequency abnormal return correlation among momentum stocks — as a crowding measure. When comomentum is high, momentum stocks revert sharply.

- **Brown, G., Howard, P., & Lundblad, C.** (2022). "Crowded Trades and Tail Risk." *Review of Financial Studies*, 35, 3231–3271. DOI: 10.1093/rfs/hhab107. Crowded stocks outperform non-crowded stocks but with substantially elevated tail risk.

- **Volpati, V., Benzaquen, M., Eisler, Z., Mastromatteo, I., Tóth, B., & Bouchaud, J.-P.** (2020). "Zooming In on Equity Factor Crowding." arXiv: 2001.04185. Identifies significant crowding in Fama-French factors, especially Momentum, with crowding percentages increasing over time.

- **van Kralingen et al.** (2021) [see above]. Market clustering directly causes tail risk.

### ARP crowding and capacity decay

- **Baltas, N.** (2019). "The Impact of Crowding in Alternative Risk Premia Investing." *Financial Analysts Journal*, 75(3), 50–65. DOI: 10.1080/0015198X.2019.1600955. Classifies ARP into **divergence premia** (e.g., momentum — self-reinforcing, no fundamental anchor) and **convergence premia** (e.g., value — self-correcting). Divergence premia underperform following crowded periods.

- **DeMiguel, V., Martin-Utrera, A., & Uppal, R.** (2021). "What Alleviates Crowding in Factor Investing?" CEPR Discussion Paper 16527; SSRN 3928838. Identifies "trading diversification" — institutions exploiting different characteristics reduce each other's price-impact costs. **Trading diversification increases capacity by 45%, optimal investment by 43%, and profits by 22%.** This is the strongest empirical support for JANUS's heterogeneity mechanism.

- **Korajczyk, R. & Sadka, R.** (2004). "Are Momentum Profits Robust to Trading Costs?" *Journal of Finance*, 59(3), 1039–1082. Early work establishing momentum capacity limits.

- **Kang, W., Rouwenhorst, K.G., & Tang, K.** (2021). "Crowding and Factor Returns." Using CFTC data, shows crowding has strong negative predictive impact on expected factor returns — returns accumulate primarily during low-crowding periods.

### Heterogeneous agents and systemic risk reduction

- **Wagner, W.** (2011). "Systemic Liquidation Risk and the Diversity–Diversification Trade-Off." *Journal of Finance*, 66(4), 1141–1175. DOI: 10.1111/j.1540-6261.2011.01666.x. **Key paper for JANUS.** Shows investors should rationally choose heterogeneous portfolios and forgo diversification benefits to avoid joint liquidation risk. Portfolio heterogeneity directly reduces systemic risk.

- **Caccioli, F., Shrestha, M., Moore, C., & Farmer, J.D.** (2014). "Stability analysis of financial contagion due to overlapping portfolios." *Journal of Banking & Finance*, 46, 233–245. Portfolio overlap (crowding/homogeneity) creates contagion channels.

- **Caccioli, F., Farmer, J.D., Foti, N., & Rockmore, D.** (2015). "Overlapping portfolios, contagion, and financial stability." *Journal of Economic Dynamics and Control*, 51, 50–63.

- **Hommes, C.** (2006/2011). "Heterogeneous Agent Models in Finance." *Journal of Economic Dynamics and Control*, 35(1), 1–24; also *Handbook of Computational Economics*, Vol. 2. Demonstrates how agent diversity is central to market stability.

---

## 4. Neuromorphic and brain-inspired trading systems: the landscape

### What exists today

The intersection of neuromorphic computing and finance is extremely sparse. Only **spiking neural network (SNN) applications** have been published:

- **Mohan, A. et al.** (2025). "Spiking Neural Network for Cross-Market Portfolio Optimization." arXiv: 2510.15921. Applies SNNs with Leaky Integrate-and-Fire dynamics, STDP, and lateral inhibition to cross-market portfolio optimization.

- **Ezinwoke, B. & Rhodes, O.** (2025). "Predicting Price Movements in High-Frequency Financial Data with Spiking Neural Networks." arXiv: 2512.05868. Three SNN architectures for HFT price-spike forecasting.

- **Bi, Y. & Calhoun, V.D.** (2025). "The Financial Connectome: A Brain-Inspired Framework for Modeling Latent Market Dynamics." arXiv: 2508.02012. The closest existing work — uses brain connectivity analysis methods (not brain-region mapping) for market dynamics.

No published papers use Intel Loihi or IBM TrueNorth for finance. **No system maps multiple brain regions to different trading subsystems** — this is JANUS's primary novelty claim.

### The Doya framework: JANUS's theoretical foundation

Kenji Doya's work provides the neuroscience foundation for mapping brain regions to computational roles:

- **Doya, K.** (1999). "What are the computations of the cerebellum, the basal ganglia and the cerebral cortex?" *Neural Networks*, 12(7–8), 961–974. DOI: 10.1016/S0893-6080(99)00046-5. **The canonical paper:** cerebellum = supervised learning, basal ganglia = reinforcement learning, cerebral cortex = unsupervised learning.

- **Doya, K.** (2000). "Complementary roles of basal ganglia and cerebellum in learning and motor control." *Current Opinion in Neurobiology*, 10(6), 732–739. DOI: 10.1016/S0959-4388(00)00153-7.

- **Doya, K.** (2002). "Metalearning and neuromodulation." *Neural Networks*, 15(4–6), 495–506. DOI: 10.1016/S0893-6080(02)00044-8. Proposes that neuromodulators (dopamine, serotonin, norepinephrine, acetylcholine) regulate meta-parameters of RL — learning rate, discount factor, exploration-exploitation.

- **Fermin, A.S.R. et al.** (2016). "Model-based action planning involves cortico-cerebellar and basal ganglia networks." *Scientific Reports*, 6, 31378. fMRI confirmation of Doya's framework.

### OpAL model: the basal ganglia decision engine

- **Collins, A.G.E. & Frank, M.J.** (2014). "Opponent Actor Learning (OpAL): Modeling Interactive Effects of Striatal Dopamine on Reinforcement Learning and Choice Incentive." *Psychological Review*, 121(3), 337–366. DOI: 10.1037/a0037015. The canonical OpAL paper. Dual opponent actors representing striatal D1/D2 populations differentially specialize in discriminating positive and negative action values.

- **Jaskir, A. & Frank, M.J.** (2023). "On the normative advantages of dopamine and striatal opponency for learning and choice." *eLife*, 12, e85107. OpAL* extends the model with dynamic dopamine modulation, normalized prediction errors, and uncertainty-adjusted learning rates. Shows **robust advantages in sparse reward environments and large action spaces** — directly relevant to trading where rewards are sparse and actions numerous.

- **Frank, M.J.** (2005). "Dynamic dopamine modulation in the basal ganglia." *Journal of Cognitive Neuroscience*, 17(1). Earlier basal ganglia model.

**Critical gap:** No papers apply OpAL or any basal ganglia computational model to financial decision-making. JANUS is the first.

### Dual-process architecture

The Kahneman framework is well-established but has **never been implemented as a computational trading architecture**:

- **Kahneman, D.** (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- **Kahneman, D. & Tversky, A.** (1979). "Prospect Theory: An Analysis of Decision under Risk." *Econometrica*, 47(2), 263–291. DOI: 10.2307/1914185.
- **Evans, J.St.B.T. & Stanovich, K.E.** (2013). "Dual-Process Theories of Higher Cognition: Advancing the Debate." *Perspectives on Psychological Science*, 8(3), 223–241.

Despite extensive searching, **no papers formalize dual-process theory into a computational trading algorithm**. Dual-process is discussed in behavioral finance (explaining trader biases) but never operationalized into System 1 / System 2 trading subsystems. This is a genuine novelty for JANUS.

---

## 5. Technical component citations: LTN, ViViT, replay, and multi-timescale memory

### Logic Tensor Networks

- **Serafini, L. & d'Avila Garcez, A.S.** (2016). "Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge." arXiv: 1606.04422. Also in *AI*IA 2016*, pp. 334–348.

- **Badreddine, S., d'Avila Garcez, A., Serafini, L., & Spranger, M.** (2022). "Logic Tensor Networks." *Artificial Intelligence*, 303, 103649. DOI: 10.1016/j.artint.2021.103649. The definitive journal paper with Real Logic — a fully differentiable first-order logic language grounded onto neural computational graphs.

- **Donadello, I., Serafini, L., & d'Avila Garcez, A.** (2017). "Logic Tensor Networks for Semantic Image Interpretation." arXiv: 1705.08968.

**No papers apply LTN to finance.** This is a novel contribution of JANUS.

### ViViT and vision-based trading

- **Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C.** (2021). "ViViT: A Video Vision Transformer." *ICCV*, pp. 6836–6846. arXiv: 2103.15691.

- **Dosovitskiy, A. et al.** (2021). "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. arXiv: 2010.11929. Foundation for ViViT.

- **Chen, J.H. & Tsai, Y.C.** (2020). "Encoding candlesticks as images for pattern classification using convolutional neural networks." *Financial Innovation*, 6, 26. DOI: 10.1186/s40854-020-00187-0. Uses Gramian Angular Field (GAF) encoding — relevant to DiffGAF naming.

**No papers apply ViViT or video vision transformers to financial chart analysis.** Treating temporal market data as "video" for transformer-based processing is novel.

### DiffGAF

**"DiffGAF" does not appear in any published literature.** The closest related work:

- **Mantri, K.S.I. et al.** (2024). "DiGRAF: Diffeomorphic Graph-Adaptive Activation Function." *NeurIPS 2024*. arXiv: 2407.02013. Learnable graph-adaptive activation functions using CPAB diffeomorphisms.

- **Ramachandran, P., Zoph, B., & Le, Q.V.** (2017). "Searching for Activation Functions." arXiv: 1710.05941. The "Swish" paper — automated activation function search.

- **Liu, H., Simonyan, K., & Yang, Y.** (2019). "DARTS: Differentiable Architecture Search." *ICLR*. arXiv: 1806.09055.

If DiffGAF refers to "Differentiable GAF" (Gramian Angular Field), then Chen & Tsai (2020) above is the relevant GAF citation.

### Sharp-wave ripple replay and experience replay

- **Buzsáki, G.** (2015). "Hippocampal Sharp Wave-Ripple: A Cognitive Biomarker for Episodic Memory and Planning." *Hippocampus*, 25(10), 1073–1188. DOI: 10.1002/hipo.22488. The comprehensive review (116 pages) on SWR.

- **Buzsáki, G.** (1989). "Two-Stage Model of Memory Trace Formation." *Neuroscience*, 31(3), 551–570. DOI: 10.1016/0306-4522(89)90423-5.

- **Wilson, M.A. & McNaughton, B.L.** (1994). "Reactivation of Hippocampal Ensemble Memories During Sleep." *Science*, 265(5172), 676–679. DOI: 10.1126/science.8036517. First direct evidence for experience replay in the brain.

- **Káli, S. & Dayan, P.** (2022). "Hippocampal sharp wave-ripples and the associated sequence replay emerge from structured synaptic interactions." *eLife*, 11, e71850.

- **Mnih, V. et al.** (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529–533. DOI: 10.1038/nature14236. Experience replay explicitly inspired by hippocampal replay.

- **Mattar, M.G. & Daw, N.D.** (2018). "Prioritized memory access explains planning and hippocampal replay." *Nature Neuroscience*, 21, 1609–1617. Models replay as utility-prioritized — relevant to JANUS's prioritized replay design.

### Complementary Learning Systems and multi-timescale memory

- **McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C.** (1995). "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex." *Psychological Review*, 102(3), 419–457. DOI: 10.1037/0033-295X.102.3.419.

- **Kumaran, D., Hassabis, D., & McClelland, J.L.** (2016). "What Learning Systems Do Intelligent Agents Need? Complementary Learning Systems Theory Updated." *Trends in Cognitive Sciences*, 20(7), 512–534. DOI: 10.1016/j.tics.2016.05.004. CLS 2.0.

- **Masset, P. et al.** (2025). "Multi-timescale reinforcement learning in the brain." *Nature*, 642, 682–690. DOI: 10.1038/s41586-025-08929-9. Shows dopaminergic neurons encode RPEs at diverse temporal discount factors — direct biological validation for three-timescale architecture.

- **Deb, R., Ganesh, S., & Bhatnagar, S.** (2021). "Multi Timescale Stochastic Approximation: Stability and Convergence." arXiv: 2112.03515. First sufficient conditions for N-timescale SA stability — theoretical guarantee for three-timescale algorithms.

---

## 6. Gap analysis: what reviewers will challenge

### The brain metaphor critique

Reviewers will likely challenge the fundamental premise of mapping brain regions to trading subsystems. Key critiques:

- **Schuman, C.D. et al.** (2022). "Opportunities for neuromorphic computing algorithms and applications." *Nature Computational Science*, 2, 10–19. Notes most neuromorphic work focuses on hardware with insufficient algorithm development.

- **Kudithipudi, D. et al.** (2025). "Roadmap for neuromorphic computing at scale." *Nature*. States neuromorphic computing "needs to scale up if it is to effectively compete with current computing methods" and there is no one-size-fits-all solution.

The brain metaphor literature (Daugman, Cobb 2021, Salles et al. 2020) documents how each generation projects its latest technology onto the brain. **JANUS should preemptively address this by citing Hassabis et al. (2017) and arguing the mapping is functional, not literal.**

### RL in non-stationary financial environments

This is likely the strongest technical objection:

- Financial RL surveys (arXiv: 2411.12746, 2025; arXiv: 2512.10913, 2025) document: heavy-tailed rewards, non-stationarity degrading performance, sample inefficiency, and strategies optimized for one period failing in the next.

- **Padakandla, S. et al.** (2020). "Reinforcement learning algorithm for non-stationary environments." *Applied Intelligence*, 50, 3590–3606. arXiv: 1905.03970.

- **Riva, A. et al.** (2022). "Addressing Non-Stationarity in FX Trading with Online Model Selection of Offline RL Experts." *ICAIF 2022*. Proposes an ensemble approach — conceptually aligned with JANUS's heterogeneity.

### Backtest overfitting and ML trading failures

- **Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J.** (2015). "The Probability of Backtest Overfitting." *Journal of Computational Finance*. SSRN: 2326253. The foundational CSCV framework.

- **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley. Comprehensive treatment.

- **López de Prado, M.** (2018). "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management*, 44(6), 120–133. Warning: "ML algorithms will always find a pattern, even if there is none."

### Logic Tensor Networks scalability

- **Wan, Z. et al.** (2024). "Towards Efficient Neuro-Symbolic AI: From Workload Characterization to Hardware Architecture." *IEEE Trans. CASAI*. arXiv: 2409.13153. **Critical finding:** LTN workloads show symbolic operations are memory-bounded while neural operations are compute-bounded, creating fundamental hardware utilization challenges. LTN suffers from "inefficiencies on off-the-shelf hardware."

- **Roth, S. et al.** (2025). "Enhancing Symbolic Machine Learning by Neural Embeddings." arXiv: 2506.14569. Explicitly notes LTN is "less efficient in simpler settings... in particular in domains with many constants."

- Neuro-symbolic scalability is identified as the field's "holy grail" barrier by Colelough (2024) survey of 167 papers.

### Does engineered heterogeneity truly prevent crowding?

The literature is **supportive but not unconditional**:

- **DeMiguel et al.** (2021) provides the strongest support: trading diversification alleviates crowding by 45% capacity increase.

- **Wagner** (2011) shows rational agents should choose heterogeneous portfolios.

- **However**, the 2007 quant quake (Khandani & Lo) demonstrated that independently developed strategies converged on similar factors despite nominal diversity. Evolutionary dynamics research (PLoS ONE, 2023, PMC: 10351734) shows crowding effects can still emerge in heterogeneous populations when memory sizes are small.

- **Scholl, M.P. et al.** in the PNAS special issue (2021) on evolutionary financial models shows how even diverse strategy types can produce market malfunctions through competitive interactions.

JANUS should acknowledge that heterogeneity is a **necessary but possibly insufficient** condition and specify mechanisms preventing convergent evolution of its agents.

### Experience replay in non-stationary settings

- **Isele, D. & Cosgun, A.** (2018). "Selective Experience Replay for Lifelong Learning." *AAAI*. arXiv: 1802.10269. Rare events fall out of buffers; as tasks grow, storing all experiences becomes infeasible.

- **Hayes, T.L. et al.** (2021). "Replay in Deep Learning: Current Approaches and Missing Biological Elements." *Neural Computation*. PMC: 9074752. Documents gaps between biological and artificial replay.

- **van de Ven, G.M. et al.** (2024). "Continual Learning and Catastrophic Forgetting." arXiv: 2403.05175. No existing approach fully solves the stability-plasticity dilemma.

### Adversarial robustness

- **Goldblum, M. et al.** (2021). "Adversarial Attacks on Machine Learning Systems for High-Frequency Trading." *ICAIF*. arXiv: 2002.09565. Adversarial traders can fool automated systems by placing adversarial orders on public exchanges.

- **Nehemya, E. et al.** (2021). "Taking Over the Stock Market: Adversarial Perturbations Against Algorithmic Traders." arXiv: 2010.09246. Universal perturbations work across algorithms.

- **Ataiefard, F. & Hemmati, H.** (2023). "Gray-box Adversarial Attack of Deep Reinforcement Learning-based Trading Agents." *ICMLA 2023*, pp. 675–682.

### Integration complexity

- **Glasmachers, T.** (2017). "Limits of End-to-End Learning." *PMLR*, vol. 77. Shows experimentally that end-to-end training can fail even for small multi-module systems due to non-trivial couplings.

---

## 7. Foundational references a peer reviewer expects

### Neuroscience-AI bridge (THE key citation)
- **Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M.** (2017). "Neuroscience-Inspired Artificial Intelligence." *Neuron*, 95(2), 245–258. DOI: 10.1016/j.neuron.2017.06.011.

### Cognitive architectures
- **Baars, B.J.** (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press. Global Workspace Theory.
- **Dehaene, S. & Naccache, L.** (2001). "Towards a Cognitive Neuroscience of Consciousness." *Cognition*, 79(1–2), 1–37. DOI: 10.1016/S0010-0277(00)00123-2. Global Neuronal Workspace.

### Brain region integration
- **Caligiore, D. et al.** (2017). "Consensus Paper: Towards a Systems-Level View of Cerebellar Function." *The Cerebellum*, 16, 203–229.
- **Caligiore, D., Arbib, M.A., Miall, R.C., & Baldassarre, G.** (2019). "The super-learning hypothesis: Integrating learning processes across cortex, cerebellum and basal ganglia." *Neuroscience & Biobehavioral Reviews*, 100, 19–34.
- **Yamakawa, H.** (2021). "The whole brain architecture approach." *Neural Networks*, 144, 478–495.

### Canonical RL and dopamine
- **Sutton, R.S. & Barto, A.G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- **Schultz, W., Dayan, P., & Montague, P.R.** (1997). "A Neural Substrate of Prediction and Reward." *Science*, 275(5306), 1593–1599. DOI: 10.1126/science.275.5306.1593.

### Adaptive markets and agent-based models
- **Lo, A.W.** (2004). "The Adaptive Markets Hypothesis." *Journal of Portfolio Management*, 30(5), 15–29. DOI: 10.3905/jpm.2004.442611.
- **Lo, A.W.** (2017). *Adaptive Markets: Financial Evolution at the Speed of Thought*. Princeton University Press.
- **Farmer, J.D., Patelli, P., & Zovko, I.I.** (2005). "The Predictive Power of Zero Intelligence in Financial Markets." *PNAS*, 102(6), 2254–2259.
- **LeBaron, B.** (2006). "Agent-Based Computational Finance." *Handbook of Computational Economics*, Vol. 2, pp. 1187–1233.

### Rust language
- **Matsakis, N.D. & Klock, F.S., II.** (2014). "The Rust Language." *ACM SIGAda Ada Letters*, 34(3), 103–104. DOI: 10.1145/2692956.2663188.
- **Jung, R., Jourdan, J.-H., Krebbers, R., & Dreyer, D.** (2018). "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages*, 2(POPL), Article 66. DOI: 10.1145/3158154.

### Neuro-symbolic AI
- **d'Avila Garcez, A.S., Lamb, L.C., & Gabbay, D.M.** (2009). *Neural-Symbolic Cognitive Reasoning*. Springer.
- **Garcez, A.d'A. & Lamb, L.C.** (2023). "Neurosymbolic AI: The 3rd Wave." *Artificial Intelligence Review*.
- **Marra, G., Dumančić, S., Manhaeve, R., & De Raedt, L.** (2024). "From Statistical Relational to Neurosymbolic Artificial Intelligence: A Survey." *Artificial Intelligence*, 328, 104062.

### Cerebellar forward models
- **Wolpert, D.M., Miall, R.C., & Kawato, M.** (1998). "Internal models in the cerebellum." *Trends in Cognitive Sciences*, 2(9), 338–347.
- **Sokolov, A.A. et al.** (2017). "The Cerebellum: Adaptive Prediction for Movement and Cognition." *Trends in Cognitive Sciences*, 21(5), 313–332.

### Thalamic gating
- **George, D. et al.** (2025). "A detailed theory of thalamic and cortical microcircuits for predictive visual inference." *Science Advances*. DOI: 10.1126/sciadv.adr6698.

### Continual learning
- **van de Ven, G.M. et al.** (2020). "Brain-inspired replay for continual learning with artificial neural networks." *Nature Communications*, 11, 4069.

---

## 8. Where JANUS stands: novelty map and strategic positioning

Based on this comprehensive literature review, JANUS's novelty claims can be ranked by strength:

**Strong novelty (no prior work found):**
1. Multi-brain-region architecture for trading — no competing system exists
2. OpAL basal ganglia model applied to financial decision-making — first application
3. Dual-process theory operationalized as computational trading architecture — never done
4. LTN constraints applied to trading — no prior work
5. ViViT for financial temporal visual analysis — no prior work
6. Cerebellar forward models for trade execution timing — no prior work
7. Thalamic gating for market data fusion — no prior work
8. Hypothalamic homeostatic risk regulation — no prior work

**Moderate novelty (extends existing work):**
9. Three-timescale CLS memory for trading — CLS theory exists, application to trading is new
10. Sharp-wave ripple-inspired replay for financial RL — experience replay is standard, the biological specificity is new
11. Engineered heterogeneity for crowding immunity — theoretical support exists (Wagner 2011, DeMiguel 2021), but operationalization as a system design principle for trading AI is new

**Points requiring careful argumentation:**
12. "Crowding immunity" as a claim — literature supports heterogeneity reducing crowding but does not support immunity. Recommend softening to "crowding resistance" or "crowding mitigation."
13. Integration of 7+ brain-region modules into a coherent system — Glasmachers (2017) shows multi-module integration is fundamentally hard. JANUS needs empirical evidence or theoretical argument for why integration works.

### Critical recommendation for the whitepaper

The strongest positioning strategy is to frame JANUS within the Hassabis et al. (2017) program of neuroscience-inspired AI, using Doya's (1999) functional brain-region decomposition as the architectural blueprint. The Bucci et al. (2020) co-impact framework provides the financial motivation, while DeMiguel et al. (2021) and Wagner (2011) provide the theoretical mechanism for why heterogeneity works. **Peer reviewers will expect preemptive responses to four objections**: (1) the brain metaphor is too loose, (2) integration complexity, (3) backtest overfitting risk, and (4) heterogeneity may not guarantee immunity in tail events. Addressing these directly, with the citations provided in the gap analysis, will substantially strengthen the paper.