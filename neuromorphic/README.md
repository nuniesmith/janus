# JANUS Neuromorphic Architecture

This module implements a neuromorphic architecture for the JANUS trading system, inspired by cognitive neuroscience.

## Brain Regions

Each directory represents a brain region with specific responsibilities:

### Cortex

**Function:** Strategic Planning & Long-term Memory (Manager Agent)

**Neuroscience:** Executive function, strategic planning, declarative memory

**Components:**
- `manager/`
- `memory/`
- `planning/`

### Hippocampus

**Function:** Episodic Memory & Experience Replay (Worker Agent)

**Neuroscience:** Memory formation, spatial navigation, replay during sleep

**Components:**
- `worker/`
- `replay/`
- `episodes/`
- `swr/`

### Basal Ganglia

**Function:** Action Selection & Reinforcement Learning (Actor-Critic)

**Neuroscience:** Action selection, habit formation, reward processing

**Components:**
- `actor/`
- `critic/`
- `praxeological/`
- `selection/`

### Thalamus

**Function:** Attention & Multimodal Fusion (Sensory Relay)

**Neuroscience:** Sensory relay, attention gating, arousal regulation

**Components:**
- `attention/`
- `gating/`
- `routing/`
- `fusion/`

### Prefrontal

**Function:** Logic, Planning & Compliance (Executive Control)

**Neuroscience:** Logical reasoning, planning, impulse control, ethics

**Components:**
- `ltn/`
- `conscience/`
- `planning/`
- `goals/`

### Amygdala

**Function:** Fear, Threat Detection & Circuit Breakers (Emotional Response)

**Neuroscience:** Fear conditioning, threat detection, emotional memory

**Components:**
- `fear/`
- `vpin/`
- `circuit_breakers/`
- `threat_detection/`

### Hypothalamus

**Function:** Homeostasis & Risk Appetite (Internal Regulation)

**Neuroscience:** Homeostatic regulation, motivation, energy balance

**Components:**
- `homeostasis/`
- `position_sizing/`
- `risk_appetite/`
- `energy/`

### Cerebellum

**Function:** Motor Control & Execution (Fine-tuned Actions)

**Neuroscience:** Motor coordination, procedural learning, error correction

**Components:**
- `execution/`
- `impact/`
- `forward_models/`
- `error_correction/`

### Visual Cortex

**Function:** Pattern Recognition & Vision (Visual Processing)

**Neuroscience:** Visual processing, hierarchical feature extraction, object recognition

**Components:**
- `eyes/`
- `gaf/`
- `vivit/`
- `visualization/`

### Integration

**Function:** Brainstem & Global Coordination (System Integration)

**Neuroscience:** Basic life functions, global coordination, arousal/sleep cycles

**Components:**
- `workflow/`
- `state/`
- `api/`
- `engine/`


## Architecture Principles

1. **Hierarchical Processing**: Low-level → Mid-level → High-level
2. **Parallel Processing**: Multiple regions process simultaneously
3. **Feedback Loops**: Bidirectional information flow
4. **Memory Consolidation**: Sleep state transfers memories
5. **Homeostatic Regulation**: Maintain internal balance
6. **Emotional Gating**: Fear can override rational processing

## Usage

```rust
use janus_neuromorphic::cortex::manager::StrategicPolicy;
use janus_neuromorphic::hippocampus::replay::PrioritizedReplayBuffer;
use janus_neuromorphic::visual_cortex::gaf::DiffGAF;

// Create components
let manager = StrategicPolicy::new();
let replay_buffer = PrioritizedReplayBuffer::new();
let gaf_encoder = DiffGAF::new();
```

## See Also

- [Neuromorphic Architecture Documentation](../../../docs/NEUROMORPHIC_ARCHITECTURE.md)
- [JANUS README](../../../README.md)
