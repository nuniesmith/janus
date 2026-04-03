//! Feudal / Hierarchical Reinforcement Learning manager
//!
//! Part of the Cortex region
//! Component: manager
//!
//! Implements a two-level feudal RL hierarchy (Vezhnevets et al., 2017 style)
//! where a **Manager** sets high-level goals (target return, risk budget,
//! regime stance) and one or more **Workers** translate those goals into
//! concrete trading actions (position sizing, entry/exit timing).
//!
//! Key features:
//! - Manager policy selects a goal vector every `manager_horizon` steps
//! - Worker policy maps (state, goal) → action at every step
//! - Intrinsic reward for worker: cosine similarity between state transition
//!   direction and the goal vector set by the manager
//! - Extrinsic reward for manager: cumulative environment reward over its horizon
//! - EMA-smoothed tracking of manager/worker performance
//! - Sliding window of recent goal–outcome pairs for analysis
//! - ε-greedy exploration with configurable decay
//! - Running statistics for audit and diagnostics

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the hierarchical RL engine
#[derive(Debug, Clone)]
pub struct HierarchicalRlConfig {
    /// Dimensionality of the state representation
    pub state_dim: usize,
    /// Dimensionality of the goal vector set by the manager
    pub goal_dim: usize,
    /// Dimensionality of the action space for the worker
    pub action_dim: usize,
    /// Number of worker steps per manager decision
    pub manager_horizon: usize,
    /// Learning rate for manager value updates
    pub manager_lr: f64,
    /// Learning rate for worker value updates
    pub worker_lr: f64,
    /// Discount factor γ for future rewards
    pub discount: f64,
    /// Initial exploration rate ε
    pub epsilon_start: f64,
    /// Minimum exploration rate
    pub epsilon_min: f64,
    /// Multiplicative decay applied to ε each manager cycle
    pub epsilon_decay: f64,
    /// EMA decay factor for performance tracking (0 < α < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent goal–outcome records
    pub window_size: usize,
    /// Intrinsic reward scale for worker cosine-similarity reward
    pub intrinsic_reward_scale: f64,
    /// Number of discrete goal options the manager can choose from
    pub num_goal_options: usize,
    /// PRNG seed for reproducibility (0 = use step counter)
    pub seed: u64,
}

impl Default for HierarchicalRlConfig {
    fn default() -> Self {
        Self {
            state_dim: 8,
            goal_dim: 4,
            action_dim: 3,
            manager_horizon: 10,
            manager_lr: 0.01,
            worker_lr: 0.05,
            discount: 0.99,
            epsilon_start: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            ema_decay: 0.1,
            window_size: 200,
            intrinsic_reward_scale: 1.0,
            num_goal_options: 8,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Goal & Action types
// ---------------------------------------------------------------------------

/// A goal vector issued by the manager to the worker
#[derive(Debug, Clone)]
pub struct GoalVector {
    /// Goal components (length = goal_dim)
    pub components: Vec<f64>,
    /// Human-readable label (optional)
    pub label: String,
    /// Index into the discrete goal set
    pub index: usize,
}

/// An action produced by the worker
#[derive(Debug, Clone)]
pub struct Action {
    /// Action components (length = action_dim)
    pub components: Vec<f64>,
}

/// A transition record used for learning
#[derive(Debug, Clone)]
pub struct Transition {
    /// State before the step
    pub state: Vec<f64>,
    /// Goal active during this step
    pub goal: GoalVector,
    /// Action taken
    pub action: Action,
    /// Extrinsic reward from environment
    pub extrinsic_reward: f64,
    /// Intrinsic reward (cosine similarity with goal)
    pub intrinsic_reward: f64,
    /// Next state after the step
    pub next_state: Vec<f64>,
}

/// Record of a completed manager cycle (one goal → N worker steps)
#[derive(Debug, Clone)]
pub struct GoalOutcome {
    /// Goal that was set
    pub goal: GoalVector,
    /// Cumulative extrinsic reward over the horizon
    pub cumulative_reward: f64,
    /// Average intrinsic reward of the worker over the horizon
    pub avg_intrinsic_reward: f64,
    /// State at goal start
    pub start_state: Vec<f64>,
    /// State at goal end
    pub end_state: Vec<f64>,
    /// Number of steps actually taken (may be < horizon at episode end)
    pub steps_taken: usize,
    /// Whether the goal was achieved (cosine sim of state delta vs goal > 0.5)
    pub achieved: bool,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the hierarchical RL engine
#[derive(Debug, Clone)]
pub struct HierarchicalRlStats {
    /// Total worker steps executed
    pub total_steps: u64,
    /// Total manager cycles completed
    pub total_cycles: u64,
    /// EMA of cumulative extrinsic reward per manager cycle
    pub ema_cycle_reward: f64,
    /// EMA of worker intrinsic reward
    pub ema_intrinsic_reward: f64,
    /// EMA of goal achievement rate
    pub ema_goal_achievement: f64,
    /// Current exploration rate ε
    pub current_epsilon: f64,
    /// Best single-cycle reward observed
    pub best_cycle_reward: f64,
    /// Worst single-cycle reward observed
    pub worst_cycle_reward: f64,
    /// Goal selection counts (indexed by goal index)
    pub goal_counts: Vec<u64>,
    /// Manager value estimates per goal (Q-values)
    pub manager_q_values: Vec<f64>,
    /// Worker cumulative intrinsic reward per goal
    pub worker_intrinsic_by_goal: Vec<f64>,
}

impl HierarchicalRlStats {
    fn new(num_goals: usize) -> Self {
        Self {
            total_steps: 0,
            total_cycles: 0,
            ema_cycle_reward: 0.0,
            ema_intrinsic_reward: 0.0,
            ema_goal_achievement: 0.0,
            current_epsilon: 1.0,
            best_cycle_reward: f64::NEG_INFINITY,
            worst_cycle_reward: f64::INFINITY,
            goal_counts: vec![0; num_goals],
            manager_q_values: vec![0.0; num_goals],
            worker_intrinsic_by_goal: vec![0.0; num_goals],
        }
    }

    /// Most frequently selected goal index
    pub fn most_selected_goal(&self) -> usize {
        self.goal_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Goal with the highest Q-value
    pub fn best_goal(&self) -> usize {
        self.manager_q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Average reward per cycle
    pub fn avg_cycle_reward(&self) -> f64 {
        if self.total_cycles == 0 {
            return 0.0;
        }
        self.ema_cycle_reward
    }

    /// Goal achievement rate (EMA-smoothed)
    pub fn goal_achievement_rate(&self) -> f64 {
        self.ema_goal_achievement
    }
}

// ---------------------------------------------------------------------------
// Simple PRNG (xoshiro128-style, lightweight, no external dep)
// ---------------------------------------------------------------------------

struct Rng {
    s: [u64; 2],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let s0 = seed.wrapping_add(0x9E3779B97F4A7C15);
        let s1 = s0.wrapping_mul(0xBF58476D1CE4E5B9);
        Self {
            s: [s0 | 1, s1 | 1],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let s0 = self.s[0];
        let mut s1 = self.s[1];
        let result = s0.wrapping_add(s1);
        s1 ^= s0;
        self.s[0] = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s[1] = s1.rotate_left(37);
        result
    }

    /// Uniform [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n)
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal via Box-Muller
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

#[allow(dead_code)]
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_normalize(v: &[f64]) -> Vec<f64> {
    let n = norm(v);
    if n < 1e-15 {
        return v.to_vec();
    }
    vec_scale(v, 1.0 / n)
}

// ---------------------------------------------------------------------------
// Hierarchical RL Engine
// ---------------------------------------------------------------------------

/// Hierarchical (feudal) reinforcement learning engine.
///
/// The Manager selects goals every `manager_horizon` worker steps.
/// The Worker maps (state, goal) → action each step and receives intrinsic
/// reward based on how well its state transition aligns with the goal.
pub struct HierarchicalRl {
    config: HierarchicalRlConfig,
    rng: Rng,
    /// Discrete goal library (generated at init)
    goal_library: Vec<Vec<f64>>,
    /// Currently active goal index
    active_goal_idx: usize,
    /// Steps within the current manager cycle
    cycle_step: usize,
    /// Accumulated extrinsic reward in the current cycle
    cycle_reward: f64,
    /// Accumulated intrinsic reward in the current cycle
    cycle_intrinsic: f64,
    /// State at the start of the current cycle
    cycle_start_state: Vec<f64>,
    /// EMA initialized flag
    ema_initialized: bool,
    /// Sliding window of recent goal outcomes
    recent: VecDeque<GoalOutcome>,
    /// Running statistics
    stats: HierarchicalRlStats,
    /// Worker weight matrix: action_dim × (state_dim + goal_dim)
    worker_weights: Vec<Vec<f64>>,
}

impl Default for HierarchicalRl {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalRl {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        let config = HierarchicalRlConfig::default();
        Self::build(config)
    }

    /// Create with explicit configuration
    pub fn with_config(config: HierarchicalRlConfig) -> Result<Self> {
        Self::validate_config(&config)?;
        Ok(Self::build(config))
    }

    fn validate_config(config: &HierarchicalRlConfig) -> Result<()> {
        if config.state_dim == 0 {
            return Err(Error::InvalidInput("state_dim must be > 0".into()));
        }
        if config.goal_dim == 0 {
            return Err(Error::InvalidInput("goal_dim must be > 0".into()));
        }
        if config.action_dim == 0 {
            return Err(Error::InvalidInput("action_dim must be > 0".into()));
        }
        if config.manager_horizon == 0 {
            return Err(Error::InvalidInput("manager_horizon must be > 0".into()));
        }
        if config.manager_lr <= 0.0 || config.manager_lr > 1.0 {
            return Err(Error::InvalidInput("manager_lr must be in (0, 1]".into()));
        }
        if config.worker_lr <= 0.0 || config.worker_lr > 1.0 {
            return Err(Error::InvalidInput("worker_lr must be in (0, 1]".into()));
        }
        if config.discount < 0.0 || config.discount > 1.0 {
            return Err(Error::InvalidInput("discount must be in [0, 1]".into()));
        }
        if config.epsilon_start < 0.0 || config.epsilon_start > 1.0 {
            return Err(Error::InvalidInput(
                "epsilon_start must be in [0, 1]".into(),
            ));
        }
        if config.epsilon_min < 0.0 || config.epsilon_min > 1.0 {
            return Err(Error::InvalidInput("epsilon_min must be in [0, 1]".into()));
        }
        if config.epsilon_decay <= 0.0 || config.epsilon_decay > 1.0 {
            return Err(Error::InvalidInput(
                "epsilon_decay must be in (0, 1]".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.num_goal_options == 0 {
            return Err(Error::InvalidInput("num_goal_options must be > 0".into()));
        }
        Ok(())
    }

    fn build(config: HierarchicalRlConfig) -> Self {
        let mut rng = Rng::new(config.seed);

        // Generate goal library: unit vectors in goal_dim space
        let goal_library: Vec<Vec<f64>> = (0..config.num_goal_options)
            .map(|_| {
                let raw: Vec<f64> = (0..config.goal_dim).map(|_| rng.next_normal()).collect();
                vec_normalize(&raw)
            })
            .collect();

        // Initialise worker weights with small random values
        let input_dim = config.state_dim + config.goal_dim;
        let worker_weights: Vec<Vec<f64>> = (0..config.action_dim)
            .map(|_| (0..input_dim).map(|_| rng.next_normal() * 0.1).collect())
            .collect();

        let num_goals = config.num_goal_options;
        let mut stats = HierarchicalRlStats::new(num_goals);
        stats.current_epsilon = config.epsilon_start;
        Self {
            config,
            rng,
            goal_library,
            active_goal_idx: 0,
            cycle_step: 0,
            cycle_reward: 0.0,
            cycle_intrinsic: 0.0,
            cycle_start_state: Vec::new(),
            ema_initialized: false,
            recent: VecDeque::new(),
            stats,
            worker_weights,
        }
    }

    /// Main processing function (trait conformance entry point)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Manager: goal selection
    // -----------------------------------------------------------------------

    /// Select a goal for the current manager cycle.
    ///
    /// Uses ε-greedy: with probability ε choose a random goal,
    /// otherwise choose the goal with the highest Q-value.
    pub fn select_goal(&mut self, state: &[f64]) -> Result<GoalVector> {
        if state.len() != self.config.state_dim {
            return Err(Error::InvalidInput(format!(
                "state dim mismatch: expected {}, got {}",
                self.config.state_dim,
                state.len()
            )));
        }

        let idx = if self.rng.next_f64() < self.stats.current_epsilon {
            // Explore: random goal
            self.rng.next_usize(self.config.num_goal_options)
        } else {
            // Exploit: best Q-value
            self.stats.best_goal()
        };

        self.active_goal_idx = idx;
        self.cycle_step = 0;
        self.cycle_reward = 0.0;
        self.cycle_intrinsic = 0.0;
        self.cycle_start_state = state.to_vec();

        self.stats.goal_counts[idx] += 1;

        Ok(GoalVector {
            components: self.goal_library[idx].clone(),
            label: format!("goal_{}", idx),
            index: idx,
        })
    }

    /// Get the currently active goal vector (if any)
    pub fn active_goal(&self) -> GoalVector {
        let idx = self.active_goal_idx;
        GoalVector {
            components: self.goal_library[idx].clone(),
            label: format!("goal_{}", idx),
            index: idx,
        }
    }

    // -----------------------------------------------------------------------
    // Worker: action selection
    // -----------------------------------------------------------------------

    /// Select an action given current state and the active goal.
    ///
    /// The worker uses a simple linear policy: action = W · [state; goal]
    /// with tanh activation and optional exploration noise.
    pub fn select_action(&mut self, state: &[f64]) -> Result<Action> {
        if state.len() != self.config.state_dim {
            return Err(Error::InvalidInput(format!(
                "state dim mismatch: expected {}, got {}",
                self.config.state_dim,
                state.len()
            )));
        }

        // Concatenate state and goal
        let goal = &self.goal_library[self.active_goal_idx];
        let mut input = state.to_vec();
        input.extend_from_slice(goal);

        // Linear forward pass + tanh
        let mut action_components = Vec::with_capacity(self.config.action_dim);
        for row in &self.worker_weights {
            let raw: f64 = dot(row, &input);
            let activated = raw.tanh();
            action_components.push(activated);
        }

        // Add exploration noise proportional to epsilon
        let noise_scale = self.stats.current_epsilon * 0.1;
        for c in action_components.iter_mut() {
            *c += self.rng.next_normal() * noise_scale;
            *c = c.clamp(-1.0, 1.0);
        }

        Ok(Action {
            components: action_components,
        })
    }

    // -----------------------------------------------------------------------
    // Step: process one environment transition
    // -----------------------------------------------------------------------

    /// Process a single worker step.
    ///
    /// Computes intrinsic reward, accumulates rewards, and if the manager
    /// horizon is reached, completes the cycle and updates the manager.
    ///
    /// Returns `Some(GoalOutcome)` when a manager cycle completes, `None`
    /// otherwise.
    pub fn step(
        &mut self,
        state: &[f64],
        action: &Action,
        extrinsic_reward: f64,
        next_state: &[f64],
    ) -> Result<Option<GoalOutcome>> {
        if state.len() != self.config.state_dim || next_state.len() != self.config.state_dim {
            return Err(Error::InvalidInput("state dimension mismatch".into()));
        }
        if action.components.len() != self.config.action_dim {
            return Err(Error::InvalidInput("action dimension mismatch".into()));
        }

        // Compute intrinsic reward: cosine similarity between state delta and goal
        let state_delta = vec_sub(next_state, state);
        // Project state delta to goal_dim if needed (use first goal_dim components)
        let delta_proj: Vec<f64> = if state_delta.len() >= self.config.goal_dim {
            state_delta[..self.config.goal_dim].to_vec()
        } else {
            let mut v = state_delta.clone();
            v.resize(self.config.goal_dim, 0.0);
            v
        };

        let goal = &self.goal_library[self.active_goal_idx];
        let intrinsic = cosine_similarity(&delta_proj, goal) * self.config.intrinsic_reward_scale;

        self.cycle_step += 1;
        self.cycle_reward += extrinsic_reward;
        self.cycle_intrinsic += intrinsic;
        self.stats.total_steps += 1;

        // Worker weight update (simple gradient toward intrinsic reward)
        self.update_worker(state, action, intrinsic);

        // Check if manager cycle is complete
        if self.cycle_step >= self.config.manager_horizon {
            let outcome = self.complete_cycle(next_state);
            return Ok(Some(outcome));
        }

        Ok(None)
    }

    /// Force-complete the current manager cycle (e.g. at episode end)
    pub fn force_complete_cycle(&mut self, final_state: &[f64]) -> GoalOutcome {
        self.complete_cycle(final_state)
    }

    fn complete_cycle(&mut self, end_state: &[f64]) -> GoalOutcome {
        let steps = self.cycle_step;
        let cum_reward = self.cycle_reward;
        let avg_intrinsic = if steps > 0 {
            self.cycle_intrinsic / steps as f64
        } else {
            0.0
        };

        // Check goal achievement: cosine similarity of overall state delta vs goal
        let state_delta = vec_sub(end_state, &self.cycle_start_state);
        let delta_proj: Vec<f64> = if state_delta.len() >= self.config.goal_dim {
            state_delta[..self.config.goal_dim].to_vec()
        } else {
            let mut v = state_delta;
            v.resize(self.config.goal_dim, 0.0);
            v
        };
        let goal = &self.goal_library[self.active_goal_idx];
        let achievement_sim = cosine_similarity(&delta_proj, goal);
        let achieved = achievement_sim > 0.5;

        let outcome = GoalOutcome {
            goal: GoalVector {
                components: goal.clone(),
                label: format!("goal_{}", self.active_goal_idx),
                index: self.active_goal_idx,
            },
            cumulative_reward: cum_reward,
            avg_intrinsic_reward: avg_intrinsic,
            start_state: self.cycle_start_state.clone(),
            end_state: end_state.to_vec(),
            steps_taken: steps,
            achieved,
        };

        // Update manager Q-value for the selected goal
        self.update_manager(cum_reward);

        // Update EMA statistics
        self.update_stats(&outcome);

        // Decay epsilon
        self.stats.current_epsilon =
            (self.stats.current_epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        // Store in sliding window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(outcome.clone());

        // Reset cycle accumulators
        self.cycle_step = 0;
        self.cycle_reward = 0.0;
        self.cycle_intrinsic = 0.0;

        self.stats.total_cycles += 1;

        outcome
    }

    fn update_manager(&mut self, cumulative_reward: f64) {
        let idx = self.active_goal_idx;
        let q = &mut self.stats.manager_q_values[idx];
        // Simple incremental mean update: Q ← Q + α(R - Q)
        *q += self.config.manager_lr * (cumulative_reward - *q);
    }

    fn update_worker(&mut self, state: &[f64], action: &Action, intrinsic_reward: f64) {
        // Simple policy gradient step: W += lr * reward * action ⊗ input
        let goal = &self.goal_library[self.active_goal_idx];
        let mut input = state.to_vec();
        input.extend_from_slice(goal);

        let lr = self.config.worker_lr;
        for (i, row) in self.worker_weights.iter_mut().enumerate() {
            let a = action.components[i];
            // Gradient of tanh: 1 - tanh²
            let grad_tanh = 1.0 - a * a;
            let scale = lr * intrinsic_reward * grad_tanh;
            for (j, w) in row.iter_mut().enumerate() {
                *w += scale * input[j];
                // Weight clipping for stability
                *w = w.clamp(-5.0, 5.0);
            }
        }

        // Track worker intrinsic by goal
        let idx = self.active_goal_idx;
        self.stats.worker_intrinsic_by_goal[idx] += intrinsic_reward;
    }

    fn update_stats(&mut self, outcome: &GoalOutcome) {
        let alpha = self.config.ema_decay;
        let achieved_f = if outcome.achieved { 1.0 } else { 0.0 };

        if !self.ema_initialized {
            self.stats.ema_cycle_reward = outcome.cumulative_reward;
            self.stats.ema_intrinsic_reward = outcome.avg_intrinsic_reward;
            self.stats.ema_goal_achievement = achieved_f;
            self.stats.best_cycle_reward = outcome.cumulative_reward;
            self.stats.worst_cycle_reward = outcome.cumulative_reward;
            self.ema_initialized = true;
        } else {
            self.stats.ema_cycle_reward =
                alpha * outcome.cumulative_reward + (1.0 - alpha) * self.stats.ema_cycle_reward;
            self.stats.ema_intrinsic_reward = alpha * outcome.avg_intrinsic_reward
                + (1.0 - alpha) * self.stats.ema_intrinsic_reward;
            self.stats.ema_goal_achievement =
                alpha * achieved_f + (1.0 - alpha) * self.stats.ema_goal_achievement;

            if outcome.cumulative_reward > self.stats.best_cycle_reward {
                self.stats.best_cycle_reward = outcome.cumulative_reward;
            }
            if outcome.cumulative_reward < self.stats.worst_cycle_reward {
                self.stats.worst_cycle_reward = outcome.cumulative_reward;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &HierarchicalRlStats {
        &self.stats
    }

    /// Configuration reference
    pub fn config(&self) -> &HierarchicalRlConfig {
        &self.config
    }

    /// Recent goal outcomes (sliding window)
    pub fn recent_outcomes(&self) -> &VecDeque<GoalOutcome> {
        &self.recent
    }

    /// Current exploration rate
    pub fn epsilon(&self) -> f64 {
        self.stats.current_epsilon
    }

    /// Goal library (all discrete goal options)
    pub fn goal_library(&self) -> &[Vec<f64>] {
        &self.goal_library
    }

    /// Manager Q-values for each goal
    pub fn q_values(&self) -> &[f64] {
        &self.stats.manager_q_values
    }

    /// EMA-smoothed cycle reward
    pub fn smoothed_cycle_reward(&self) -> f64 {
        self.stats.ema_cycle_reward
    }

    /// EMA-smoothed intrinsic reward
    pub fn smoothed_intrinsic_reward(&self) -> f64 {
        self.stats.ema_intrinsic_reward
    }

    /// EMA-smoothed goal achievement rate
    pub fn smoothed_goal_achievement(&self) -> f64 {
        self.stats.ema_goal_achievement
    }

    /// Windowed average cycle reward
    pub fn windowed_avg_reward(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|o| o.cumulative_reward).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed goal achievement rate
    pub fn windowed_achievement_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let achieved = self.recent.iter().filter(|o| o.achieved).count();
        achieved as f64 / self.recent.len() as f64
    }

    /// Whether cycle reward is trending upward (second half > first half)
    pub fn is_reward_improving(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|o| o.cumulative_reward)
            .sum::<f64>()
            / mid as f64;
        let second_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|o| o.cumulative_reward)
            .sum::<f64>()
            / (n - mid) as f64;
        second_avg > first_avg
    }

    /// Whether exploration has converged (epsilon near minimum)
    pub fn is_exploration_converged(&self) -> bool {
        (self.stats.current_epsilon - self.config.epsilon_min).abs() < 0.01
    }

    /// Reset all state but keep configuration and goal library
    pub fn reset(&mut self) {
        self.active_goal_idx = 0;
        self.cycle_step = 0;
        self.cycle_reward = 0.0;
        self.cycle_intrinsic = 0.0;
        self.cycle_start_state.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = HierarchicalRlStats::new(self.config.num_goal_options);
        self.stats.current_epsilon = self.config.epsilon_start;
        self.rng = Rng::new(self.config.seed);

        // Re-initialise worker weights
        let input_dim = self.config.state_dim + self.config.goal_dim;
        self.worker_weights = (0..self.config.action_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| self.rng.next_normal() * 0.1)
                    .collect()
            })
            .collect();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state(dim: usize) -> Vec<f64> {
        vec![0.1; dim]
    }

    fn next_state(dim: usize) -> Vec<f64> {
        let mut s = vec![0.1; dim];
        // Shift first few dims to create a delta
        for i in 0..dim.min(4) {
            s[i] += 0.05 * (i as f64 + 1.0);
        }
        s
    }

    fn small_config() -> HierarchicalRlConfig {
        HierarchicalRlConfig {
            state_dim: 4,
            goal_dim: 2,
            action_dim: 2,
            manager_horizon: 3,
            manager_lr: 0.1,
            worker_lr: 0.05,
            discount: 0.99,
            epsilon_start: 0.5,
            epsilon_min: 0.05,
            epsilon_decay: 0.9,
            ema_decay: 0.2,
            window_size: 10,
            intrinsic_reward_scale: 1.0,
            num_goal_options: 4,
            seed: 123,
        }
    }

    // -- Construction --

    #[test]
    fn test_new_default() {
        let h = HierarchicalRl::new();
        assert!(h.process().is_ok());
        assert_eq!(h.stats().total_steps, 0);
        assert_eq!(h.stats().total_cycles, 0);
    }

    #[test]
    fn test_with_config() {
        let h = HierarchicalRl::with_config(small_config());
        assert!(h.is_ok());
        let h = h.unwrap();
        assert_eq!(h.config().state_dim, 4);
        assert_eq!(h.goal_library().len(), 4);
    }

    #[test]
    fn test_goal_library_normalized() {
        let h = HierarchicalRl::with_config(small_config()).unwrap();
        for g in h.goal_library() {
            let n = norm(g);
            assert!(
                (n - 1.0).abs() < 1e-10,
                "goal not unit vector: norm = {}",
                n
            );
        }
    }

    // -- Config validation --

    #[test]
    fn test_invalid_state_dim() {
        let mut cfg = small_config();
        cfg.state_dim = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_goal_dim() {
        let mut cfg = small_config();
        cfg.goal_dim = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_action_dim() {
        let mut cfg = small_config();
        cfg.action_dim = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_manager_horizon() {
        let mut cfg = small_config();
        cfg.manager_horizon = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_manager_lr_zero() {
        let mut cfg = small_config();
        cfg.manager_lr = 0.0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_manager_lr_over_one() {
        let mut cfg = small_config();
        cfg.manager_lr = 1.5;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_discount() {
        let mut cfg = small_config();
        cfg.discount = -0.1;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_zero() {
        let mut cfg = small_config();
        cfg.ema_decay = 0.0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_ema_decay_one() {
        let mut cfg = small_config();
        cfg.ema_decay = 1.0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_window_size() {
        let mut cfg = small_config();
        cfg.window_size = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_num_goal_options() {
        let mut cfg = small_config();
        cfg.num_goal_options = 0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_epsilon_negative() {
        let mut cfg = small_config();
        cfg.epsilon_start = -0.1;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_epsilon_decay_zero() {
        let mut cfg = small_config();
        cfg.epsilon_decay = 0.0;
        assert!(HierarchicalRl::with_config(cfg).is_err());
    }

    // -- Goal selection --

    #[test]
    fn test_select_goal() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let goal = h.select_goal(&state);
        assert!(goal.is_ok());
        let goal = goal.unwrap();
        assert_eq!(goal.components.len(), 2);
        assert!(goal.index < 4);
    }

    #[test]
    fn test_select_goal_wrong_dim() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = vec![0.1; 10]; // Wrong dim (expected 4)
        assert!(h.select_goal(&state).is_err());
    }

    #[test]
    fn test_active_goal() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let goal = h.select_goal(&state).unwrap();
        let active = h.active_goal();
        assert_eq!(goal.index, active.index);
    }

    #[test]
    fn test_goal_counts_increment() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        h.select_goal(&state).unwrap();
        let total: u64 = h.stats().goal_counts.iter().sum();
        assert_eq!(total, 1);
    }

    // -- Action selection --

    #[test]
    fn test_select_action() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state);
        assert!(action.is_ok());
        let action = action.unwrap();
        assert_eq!(action.components.len(), 2);
        // Actions should be bounded [-1, 1]
        for c in &action.components {
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }

    #[test]
    fn test_select_action_wrong_dim() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        h.select_goal(&state).unwrap();
        assert!(h.select_action(&[0.1; 7]).is_err());
    }

    // -- Step --

    #[test]
    fn test_step_no_cycle_complete() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();

        // horizon = 3, so first step should not complete
        let result = h.step(&state, &action, 1.0, &ns).unwrap();
        assert!(result.is_none());
        assert_eq!(h.stats().total_steps, 1);
    }

    #[test]
    fn test_step_cycle_completes_at_horizon() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();

        // Do `manager_horizon` steps (= 3)
        let mut outcome = None;
        for _ in 0..3 {
            outcome = h.step(&state, &action, 1.0, &ns).unwrap();
        }
        assert!(outcome.is_some());
        let outcome = outcome.unwrap();
        assert_eq!(outcome.steps_taken, 3);
        assert!((outcome.cumulative_reward - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_state_dim_mismatch() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        h.select_goal(&state).unwrap();
        let action = Action {
            components: vec![0.0; 2],
        };
        assert!(h.step(&state, &action, 1.0, &[0.0; 7]).is_err());
    }

    #[test]
    fn test_step_action_dim_mismatch() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);
        h.select_goal(&state).unwrap();
        let action = Action {
            components: vec![0.0; 5], // Wrong dim
        };
        assert!(h.step(&state, &action, 1.0, &ns).is_err());
    }

    // -- Force complete cycle --

    #[test]
    fn test_force_complete_cycle() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();

        h.step(&state, &action, 2.0, &ns).unwrap();
        let outcome = h.force_complete_cycle(&ns);
        assert_eq!(outcome.steps_taken, 1);
        assert!((outcome.cumulative_reward - 2.0).abs() < 1e-10);
    }

    // -- Statistics tracking --

    #[test]
    fn test_stats_after_cycle() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();

        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        assert_eq!(h.stats().total_steps, 3);
        assert_eq!(h.stats().total_cycles, 1);
        assert!(h.stats().ema_cycle_reward.is_finite());
    }

    #[test]
    fn test_stats_best_worst_tracked() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        // Cycle 1: reward = 3.0
        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        // Cycle 2: reward = 6.0
        h.select_goal(&state).unwrap();
        let action2 = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action2, 2.0, &ns).unwrap();
        }

        assert!(h.stats().best_cycle_reward >= 6.0 - 1e-10);
        assert!(h.stats().worst_cycle_reward <= 3.0 + 1e-10);
    }

    #[test]
    fn test_most_selected_goal() {
        let stats = HierarchicalRlStats {
            goal_counts: vec![1, 5, 2, 0],
            ..HierarchicalRlStats::new(4)
        };
        assert_eq!(stats.most_selected_goal(), 1);
    }

    #[test]
    fn test_best_goal() {
        let stats = HierarchicalRlStats {
            manager_q_values: vec![0.5, 1.2, 0.8, 0.3],
            ..HierarchicalRlStats::new(4)
        };
        assert_eq!(stats.best_goal(), 1);
    }

    #[test]
    fn test_avg_cycle_reward_zero_cycles() {
        let stats = HierarchicalRlStats::new(4);
        assert_eq!(stats.avg_cycle_reward(), 0.0);
    }

    // -- Q-value updates --

    #[test]
    fn test_q_values_update() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let idx = h.active_goal().index;
        let q_before = h.q_values()[idx];

        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        let q_after = h.q_values()[idx];
        // Q should have moved toward cumulative reward of 3.0
        assert!((q_after - q_before).abs() > 1e-10);
    }

    // -- Epsilon decay --

    #[test]
    fn test_epsilon_decays() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let initial_eps = h.epsilon();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        assert!(h.epsilon() < initial_eps);
    }

    #[test]
    fn test_epsilon_bounded_above_min() {
        let mut cfg = small_config();
        cfg.epsilon_min = 0.05;
        cfg.epsilon_start = 0.06;
        cfg.epsilon_decay = 0.5; // Aggressive decay
        let mut h = HierarchicalRl::with_config(cfg).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        for _ in 0..20 {
            h.select_goal(&state).unwrap();
            let action = h.select_action(&state).unwrap();
            for _ in 0..3 {
                h.step(&state, &action, 1.0, &ns).unwrap();
            }
        }

        assert!(h.epsilon() >= 0.05 - 1e-10);
    }

    #[test]
    fn test_is_exploration_converged() {
        let mut cfg = small_config();
        cfg.epsilon_start = 0.05;
        cfg.epsilon_min = 0.05;
        let h = HierarchicalRl::with_config(cfg).unwrap();
        assert!(h.is_exploration_converged());
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_on_first_cycle() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        assert!(h.smoothed_cycle_reward().is_finite());
        assert!(h.smoothed_intrinsic_reward().is_finite());
    }

    #[test]
    fn test_ema_blends_on_subsequent_cycles() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        // Cycle 1
        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 10.0, &ns).unwrap();
        }
        let ema1 = h.smoothed_cycle_reward();

        // Cycle 2 with lower reward
        h.select_goal(&state).unwrap();
        let a2 = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a2, 0.0, &ns).unwrap();
        }
        let ema2 = h.smoothed_cycle_reward();

        assert!(ema2 < ema1);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_outcomes_stored() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 1.0, &ns).unwrap();
        }

        assert_eq!(h.recent_outcomes().len(), 1);
    }

    #[test]
    fn test_recent_outcomes_windowed() {
        let mut cfg = small_config();
        cfg.window_size = 3;
        let mut h = HierarchicalRl::with_config(cfg).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        for _ in 0..10 {
            h.select_goal(&state).unwrap();
            let a = h.select_action(&state).unwrap();
            for _ in 0..3 {
                h.step(&state, &a, 1.0, &ns).unwrap();
            }
        }

        assert!(h.recent_outcomes().len() <= 3);
    }

    #[test]
    fn test_windowed_avg_reward() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 2.0, &ns).unwrap();
        }

        // Cumulative reward per cycle = 6.0
        assert!((h.windowed_avg_reward() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_avg_reward_empty() {
        let h = HierarchicalRl::with_config(small_config()).unwrap();
        assert_eq!(h.windowed_avg_reward(), 0.0);
    }

    #[test]
    fn test_windowed_achievement_rate() {
        let h = HierarchicalRl::with_config(small_config()).unwrap();
        assert_eq!(h.windowed_achievement_rate(), 0.0);
    }

    // -- Reward improving trend --

    #[test]
    fn test_is_reward_improving_insufficient_data() {
        let h = HierarchicalRl::with_config(small_config()).unwrap();
        assert!(!h.is_reward_improving());
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 1.0, &ns).unwrap();
        }

        assert!(h.stats().total_steps > 0);
        assert!(h.stats().total_cycles > 0);
        assert!(!h.recent_outcomes().is_empty());

        h.reset();

        assert_eq!(h.stats().total_steps, 0);
        assert_eq!(h.stats().total_cycles, 0);
        assert!(h.recent_outcomes().is_empty());
        // Epsilon should be back to start
        assert!((h.epsilon() - small_config().epsilon_start).abs() < 1e-10);
        // Goal library should still be present
        assert_eq!(h.goal_library().len(), 4);
    }

    // -- Vector helpers --

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_vec_sub() {
        let a = vec![3.0, 2.0];
        let b = vec![1.0, 1.0];
        let r = vec_sub(&a, &b);
        assert!((r[0] - 2.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_normalize() {
        let v = vec![3.0, 4.0];
        let n = vec_normalize(&v);
        assert!((norm(&n) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_normalize_zero() {
        let v = vec![0.0, 0.0];
        let n = vec_normalize(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn test_vec_add() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let r = vec_add(&a, &b);
        assert!((r[0] - 4.0).abs() < 1e-10);
        assert!((r[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_scale() {
        let v = vec![2.0, 3.0];
        let r = vec_scale(&v, 0.5);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 1.5).abs() < 1e-10);
    }

    // -- Intrinsic reward --

    #[test]
    fn test_intrinsic_reward_in_outcome() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        let outcome = h.recent_outcomes().back().unwrap();
        assert!(outcome.avg_intrinsic_reward.is_finite());
    }

    // -- Worker intrinsic tracking per goal --

    #[test]
    fn test_worker_intrinsic_by_goal_tracked() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let idx = h.active_goal().index;
        let action = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &action, 1.0, &ns).unwrap();
        }

        // The active goal should have accumulated some intrinsic reward
        let intrinsic = h.stats().worker_intrinsic_by_goal[idx];
        assert!(intrinsic.is_finite());
    }

    // -- Multi-cycle learning --

    #[test]
    fn test_multi_cycle_learning() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        for cycle in 0..10 {
            h.select_goal(&state).unwrap();
            let a = h.select_action(&state).unwrap();
            for _ in 0..3 {
                h.step(&state, &a, (cycle + 1) as f64, &ns).unwrap();
            }
        }

        assert_eq!(h.stats().total_cycles, 10);
        assert_eq!(h.stats().total_steps, 30);
        // Q-values should have been updated
        let nonzero_q = h.q_values().iter().any(|q| q.abs() > 1e-10);
        assert!(nonzero_q);
    }

    // -- Default trait --

    #[test]
    fn test_default_trait() {
        let h = HierarchicalRl::default();
        assert_eq!(h.config().state_dim, 8);
        assert_eq!(h.config().goal_dim, 4);
    }

    // -- Reproducibility --

    #[test]
    fn test_reproducible_with_same_seed() {
        let cfg = small_config();
        let mut h1 = HierarchicalRl::with_config(cfg.clone()).unwrap();
        let mut h2 = HierarchicalRl::with_config(cfg).unwrap();
        let state = default_state(4);

        let g1 = h1.select_goal(&state).unwrap();
        let g2 = h2.select_goal(&state).unwrap();
        assert_eq!(g1.index, g2.index);
        assert_eq!(g1.components, g2.components);
    }

    // -- PRNG --

    #[test]
    fn test_rng_uniform_in_range() {
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_normal_finite() {
        let mut rng = Rng::new(77);
        for _ in 0..1000 {
            let v = rng.next_normal();
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rng_usize_in_range() {
        let mut rng = Rng::new(55);
        for _ in 0..1000 {
            let v = rng.next_usize(5);
            assert!(v < 5);
        }
    }

    // -- Goal outcome achievement --

    #[test]
    fn test_goal_outcome_achieved_flag() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 1.0, &ns).unwrap();
        }

        let outcome = h.recent_outcomes().back().unwrap();
        // achieved is a bool derived from cosine similarity — just check it exists
        let _achieved = outcome.achieved;
    }

    // -- Smoothed accessors --

    #[test]
    fn test_smoothed_goal_achievement() {
        let mut h = HierarchicalRl::with_config(small_config()).unwrap();
        let state = default_state(4);
        let ns = next_state(4);

        h.select_goal(&state).unwrap();
        let a = h.select_action(&state).unwrap();
        for _ in 0..3 {
            h.step(&state, &a, 1.0, &ns).unwrap();
        }

        let rate = h.smoothed_goal_achievement();
        assert!((0.0..=1.0).contains(&rate));
    }
}
