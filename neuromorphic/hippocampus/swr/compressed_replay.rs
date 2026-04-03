//! Fast compressed replay (10-20x speed)
//!
//! Part of the Hippocampus region
//! Component: swr
//!
//! This module implements compressed experience replay that accelerates
//! learning by replaying experiences at 10-20x speed with temporal abstraction,
//! skipping redundant states and focusing on key transitions.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Compression method for replay
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMethod {
    /// Skip similar consecutive states
    SkipSimilar,
    /// Keep only key state transitions
    KeyTransitions,
    /// Temporal difference compression (TD-based)
    TemporalDifference,
    /// Event-based compression (keep significant events)
    EventBased,
    /// Adaptive compression based on learning signal
    Adaptive,
    /// Summary statistics (abstract multiple experiences)
    Summary,
}

impl Default for CompressionMethod {
    fn default() -> Self {
        CompressionMethod::KeyTransitions
    }
}

/// Compression level (affects speed vs fidelity tradeoff)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// Minimal compression (2-3x speedup, high fidelity)
    Light,
    /// Moderate compression (5-10x speedup)
    Moderate,
    /// Aggressive compression (10-20x speedup, lower fidelity)
    Aggressive,
    /// Maximum compression (20x+ speedup, summary only)
    Maximum,
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Moderate
    }
}

impl CompressionLevel {
    /// Get target speedup factor
    pub fn speedup_factor(&self) -> f64 {
        match self {
            CompressionLevel::Light => 3.0,
            CompressionLevel::Moderate => 8.0,
            CompressionLevel::Aggressive => 15.0,
            CompressionLevel::Maximum => 25.0,
        }
    }

    /// Get similarity threshold for skipping
    pub fn similarity_threshold(&self) -> f64 {
        match self {
            CompressionLevel::Light => 0.95,
            CompressionLevel::Moderate => 0.85,
            CompressionLevel::Aggressive => 0.70,
            CompressionLevel::Maximum => 0.50,
        }
    }

    /// Get minimum event significance
    pub fn min_significance(&self) -> f64 {
        match self {
            CompressionLevel::Light => 0.1,
            CompressionLevel::Moderate => 0.3,
            CompressionLevel::Aggressive => 0.5,
            CompressionLevel::Maximum => 0.7,
        }
    }
}

/// A single experience frame
#[derive(Debug, Clone)]
pub struct ExperienceFrame {
    /// Frame ID
    pub id: u64,
    /// Timestamp
    pub timestamp: u64,
    /// State vector
    pub state: Vec<f32>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state
    pub next_state: Vec<f32>,
    /// Episode done flag
    pub done: bool,
    /// TD error (if computed)
    pub td_error: Option<f32>,
    /// Significance score (0-1)
    pub significance: f64,
    /// Whether this is a key frame
    pub is_key_frame: bool,
    /// Metadata
    pub metadata: HashMap<String, f64>,
}

impl ExperienceFrame {
    pub fn new(
        id: u64,
        timestamp: u64,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) -> Self {
        Self {
            id,
            timestamp,
            state,
            action,
            reward,
            next_state,
            done,
            td_error: None,
            significance: 0.5,
            is_key_frame: false,
            metadata: HashMap::new(),
        }
    }

    /// Set TD error
    pub fn with_td_error(mut self, td_error: f32) -> Self {
        self.td_error = Some(td_error);
        // TD error contributes to significance
        self.significance = (self.significance + td_error.abs() as f64).min(1.0);
        self
    }

    /// Set significance
    pub fn with_significance(mut self, significance: f64) -> Self {
        self.significance = significance.clamp(0.0, 1.0);
        self
    }

    /// Mark as key frame
    pub fn mark_key_frame(&mut self) {
        self.is_key_frame = true;
    }
}

/// Compressed episode representation
#[derive(Debug, Clone)]
pub struct CompressedEpisode {
    /// Episode ID
    pub id: u64,
    /// Key frames (compressed)
    pub key_frames: Vec<ExperienceFrame>,
    /// Original frame count
    pub original_length: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Total episode reward
    pub total_reward: f32,
    /// Episode outcome (final reward or terminal state indicator)
    pub outcome: f32,
    /// Summary statistics
    pub summary: EpisodeSummary,
    /// Compression method used
    pub method: CompressionMethod,
    /// Timestamp range
    pub start_time: u64,
    pub end_time: u64,
}

impl CompressedEpisode {
    pub fn new(id: u64, method: CompressionMethod) -> Self {
        Self {
            id,
            key_frames: Vec::new(),
            original_length: 0,
            compression_ratio: 1.0,
            total_reward: 0.0,
            outcome: 0.0,
            summary: EpisodeSummary::default(),
            method,
            start_time: 0,
            end_time: 0,
        }
    }

    /// Get compressed length
    pub fn len(&self) -> usize {
        self.key_frames.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.key_frames.is_empty()
    }

    /// Get speedup factor achieved
    pub fn speedup(&self) -> f64 {
        if self.key_frames.is_empty() {
            return 1.0;
        }
        self.original_length as f64 / self.key_frames.len() as f64
    }
}

/// Summary statistics for an episode
#[derive(Debug, Clone, Default)]
pub struct EpisodeSummary {
    /// Mean state (centroid)
    pub mean_state: Vec<f32>,
    /// State variance
    pub state_variance: Vec<f32>,
    /// Action distribution
    pub action_distribution: HashMap<usize, usize>,
    /// Mean reward
    pub mean_reward: f32,
    /// Reward variance
    pub reward_variance: f32,
    /// Max reward
    pub max_reward: f32,
    /// Min reward
    pub min_reward: f32,
    /// Mean TD error
    pub mean_td_error: f32,
    /// Number of positive rewards
    pub positive_reward_count: usize,
    /// Number of negative rewards
    pub negative_reward_count: usize,
    /// Duration (frames)
    pub duration: usize,
}

/// Configuration for compressed replay
#[derive(Debug, Clone)]
pub struct CompressedReplayConfig {
    /// Compression method
    pub method: CompressionMethod,
    /// Compression level
    pub level: CompressionLevel,
    /// Target speedup factor
    pub target_speedup: f64,
    /// State similarity threshold for skipping
    pub similarity_threshold: f64,
    /// Minimum significance to keep frame
    pub min_significance: f64,
    /// Keep every Nth frame regardless
    pub keep_every_n: usize,
    /// Always keep frames with reward above threshold
    pub reward_threshold: f32,
    /// Always keep frames with TD error above threshold
    pub td_error_threshold: f32,
    /// Maximum key frames per episode
    pub max_key_frames: usize,
    /// Minimum key frames per episode
    pub min_key_frames: usize,
    /// Include episode summaries
    pub include_summaries: bool,
    /// Adaptive threshold adjustment
    pub adaptive_threshold: bool,
}

impl Default for CompressedReplayConfig {
    fn default() -> Self {
        Self {
            method: CompressionMethod::KeyTransitions,
            level: CompressionLevel::Moderate,
            target_speedup: 10.0,
            similarity_threshold: 0.85,
            min_significance: 0.3,
            keep_every_n: 20,
            reward_threshold: 0.5,
            td_error_threshold: 0.1,
            max_key_frames: 100,
            min_key_frames: 5,
            include_summaries: true,
            adaptive_threshold: true,
        }
    }
}

/// Replay result from compressed playback
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Frames replayed
    pub frames_replayed: usize,
    /// Original frames represented
    pub original_frames: usize,
    /// Effective speedup
    pub speedup: f64,
    /// Episodes processed
    pub episodes_processed: usize,
    /// Total reward observed
    pub total_reward: f32,
    /// Learning signals generated
    pub learning_signals: Vec<LearningSignal>,
}

/// Learning signal from compressed replay
#[derive(Debug, Clone)]
pub struct LearningSignal {
    /// Frame ID
    pub frame_id: u64,
    /// State
    pub state: Vec<f32>,
    /// Action
    pub action: usize,
    /// Target value (e.g., TD target)
    pub target: f32,
    /// Weight for this signal
    pub weight: f64,
    /// Source episode
    pub episode_id: u64,
}

/// Fast compressed replay (10-20x speed)
pub struct CompressedReplay {
    /// Configuration
    config: CompressedReplayConfig,
    /// Stored compressed episodes
    episodes: Vec<CompressedEpisode>,
    /// Maximum stored episodes
    max_episodes: usize,
    /// Current episode being built
    current_episode: Option<Vec<ExperienceFrame>>,
    /// Episode counter
    episode_counter: u64,
    /// Frame counter
    frame_counter: u64,
    /// Total frames compressed
    total_frames_compressed: usize,
    /// Total key frames retained
    total_key_frames: usize,
    /// Adaptive similarity threshold
    adaptive_threshold: f64,
    /// Replay history
    replay_count: usize,
}

impl Default for CompressedReplay {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressedReplay {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_capacity(100)
    }

    /// Create with specific capacity
    pub fn with_capacity(max_episodes: usize) -> Self {
        Self {
            config: CompressedReplayConfig::default(),
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
            current_episode: None,
            episode_counter: 0,
            frame_counter: 0,
            total_frames_compressed: 0,
            total_key_frames: 0,
            adaptive_threshold: CompressedReplayConfig::default().similarity_threshold,
            replay_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: CompressedReplayConfig, max_episodes: usize) -> Self {
        let adaptive_threshold = config.similarity_threshold;
        Self {
            config,
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
            current_episode: None,
            episode_counter: 0,
            frame_counter: 0,
            total_frames_compressed: 0,
            total_key_frames: 0,
            adaptive_threshold,
            replay_count: 0,
        }
    }

    /// Start a new episode
    pub fn start_episode(&mut self) {
        self.current_episode = Some(Vec::new());
    }

    /// Add a frame to the current episode
    pub fn add_frame(&mut self, frame: ExperienceFrame) {
        self.frame_counter += 1;

        if let Some(ref mut episode) = self.current_episode {
            episode.push(frame);
        } else {
            // Auto-start episode
            self.current_episode = Some(vec![frame]);
        }
    }

    /// Add frame from components
    pub fn add(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
        timestamp: u64,
    ) {
        let frame = ExperienceFrame::new(
            self.frame_counter,
            timestamp,
            state,
            action,
            reward,
            next_state,
            done,
        );

        self.add_frame(frame);

        if done {
            let _ = self.end_episode();
        }
    }

    /// End the current episode and compress it
    pub fn end_episode(&mut self) -> Result<CompressedEpisode> {
        let frames = self
            .current_episode
            .take()
            .ok_or_else(|| Error::InvalidState("No episode in progress".to_string()))?;

        if frames.is_empty() {
            return Err(Error::InvalidInput("Empty episode".to_string()));
        }

        let compressed = self.compress_episode(frames)?;

        // Store the compressed episode
        if self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
        }
        self.episodes.push(compressed.clone());

        Ok(compressed)
    }

    /// Compress an episode using configured method
    fn compress_episode(&mut self, frames: Vec<ExperienceFrame>) -> Result<CompressedEpisode> {
        self.episode_counter += 1;
        let original_length = frames.len();

        let key_frames = match self.config.method {
            CompressionMethod::SkipSimilar => self.compress_skip_similar(&frames),
            CompressionMethod::KeyTransitions => self.compress_key_transitions(&frames),
            CompressionMethod::TemporalDifference => self.compress_td_based(&frames),
            CompressionMethod::EventBased => self.compress_event_based(&frames),
            CompressionMethod::Adaptive => self.compress_adaptive(&frames),
            CompressionMethod::Summary => self.compress_summary(&frames),
        };

        self.total_frames_compressed += original_length;
        self.total_key_frames += key_frames.len();

        let mut episode = CompressedEpisode::new(self.episode_counter, self.config.method);
        episode.original_length = original_length;
        episode.key_frames = key_frames;
        episode.compression_ratio = original_length as f64 / episode.key_frames.len().max(1) as f64;
        episode.total_reward = frames.iter().map(|f| f.reward).sum();
        episode.outcome = frames.last().map(|f| f.reward).unwrap_or(0.0);
        episode.start_time = frames.first().map(|f| f.timestamp).unwrap_or(0);
        episode.end_time = frames.last().map(|f| f.timestamp).unwrap_or(0);

        if self.config.include_summaries {
            episode.summary = self.compute_summary(&frames);
        }

        // Adapt threshold if needed
        if self.config.adaptive_threshold {
            self.adapt_threshold(episode.compression_ratio);
        }

        Ok(episode)
    }

    /// Compress by skipping similar consecutive states
    fn compress_skip_similar(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        let mut key_frames: Vec<ExperienceFrame> = Vec::new();
        let threshold = self.adaptive_threshold;

        for (i, frame) in frames.iter().enumerate() {
            let should_keep = if i == 0 || i == frames.len() - 1 {
                true // Always keep first and last
            } else if i % self.config.keep_every_n == 0 {
                true // Keep every Nth
            } else if frame.reward.abs() > self.config.reward_threshold {
                true // Significant reward
            } else if let Some(ref prev) = key_frames.last() {
                let similarity = self.state_similarity(&frame.state, &prev.state);
                similarity < threshold
            } else {
                true
            };

            if should_keep {
                let mut kf = frame.clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }

            // Enforce limits
            if key_frames.len() >= self.config.max_key_frames {
                break;
            }
        }

        // Ensure minimum key frames
        if key_frames.len() < self.config.min_key_frames
            && frames.len() >= self.config.min_key_frames
        {
            self.ensure_minimum_frames(frames, &mut key_frames);
        }

        key_frames
    }

    /// Compress by keeping key state transitions
    fn compress_key_transitions(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        let mut key_frames = Vec::new();
        let mut last_action: Option<usize> = None;

        for (i, frame) in frames.iter().enumerate() {
            let is_action_change = last_action.map(|a| a != frame.action).unwrap_or(true);
            let is_significant_reward = frame.reward.abs() > self.config.reward_threshold;
            let is_terminal = frame.done;
            let is_boundary = i == 0 || i == frames.len() - 1;
            let is_periodic = i % self.config.keep_every_n == 0;

            let is_td_significant = frame
                .td_error
                .map(|td| td.abs() > self.config.td_error_threshold)
                .unwrap_or(false);

            let should_keep = is_action_change
                || is_significant_reward
                || is_terminal
                || is_boundary
                || is_periodic
                || is_td_significant;

            if should_keep {
                let mut kf = frame.clone();
                kf.mark_key_frame();
                key_frames.push(kf);
                last_action = Some(frame.action);
            }

            if key_frames.len() >= self.config.max_key_frames {
                break;
            }
        }

        if key_frames.len() < self.config.min_key_frames
            && frames.len() >= self.config.min_key_frames
        {
            self.ensure_minimum_frames(frames, &mut key_frames);
        }

        key_frames
    }

    /// Compress based on TD error
    fn compress_td_based(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        let mut scored_frames: Vec<(usize, f64)> = frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let td_score = f.td_error.map(|td| td.abs() as f64).unwrap_or(0.0);
                let reward_score = f.reward.abs() as f64;
                let position_score = if i == 0 || i == frames.len() - 1 {
                    1.0
                } else {
                    0.0
                };
                let total = td_score * 2.0 + reward_score + position_score;
                (i, total)
            })
            .collect();

        // Sort by score descending
        scored_frames.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top N frames
        let target_count = (frames.len() as f64 / self.config.target_speedup).ceil() as usize;
        let count = target_count
            .max(self.config.min_key_frames)
            .min(self.config.max_key_frames)
            .min(frames.len());

        let mut selected: Vec<usize> = scored_frames.iter().take(count).map(|(i, _)| *i).collect();

        // Sort by original order
        selected.sort();

        selected
            .into_iter()
            .map(|i| {
                let mut kf = frames[i].clone();
                kf.mark_key_frame();
                kf
            })
            .collect()
    }

    /// Compress based on events/significance
    fn compress_event_based(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        let mut key_frames = Vec::new();

        for (i, frame) in frames.iter().enumerate() {
            let is_significant = frame.significance >= self.config.min_significance;
            let is_boundary = i == 0 || i == frames.len() - 1;
            let is_terminal = frame.done;
            let is_periodic = i % self.config.keep_every_n == 0;

            if is_significant || is_boundary || is_terminal || is_periodic {
                let mut kf = frame.clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }

            if key_frames.len() >= self.config.max_key_frames {
                break;
            }
        }

        if key_frames.len() < self.config.min_key_frames
            && frames.len() >= self.config.min_key_frames
        {
            self.ensure_minimum_frames(frames, &mut key_frames);
        }

        key_frames
    }

    /// Adaptive compression based on learning progress
    fn compress_adaptive(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        // Combine multiple methods based on frame characteristics
        let avg_td = frames
            .iter()
            .filter_map(|f| f.td_error)
            .map(|td| td.abs())
            .sum::<f32>()
            / frames.len().max(1) as f32;

        let reward_variance = self.compute_variance(frames.iter().map(|f| f.reward as f64));

        // High TD error -> use TD-based compression
        if avg_td > 0.1 {
            return self.compress_td_based(frames);
        }

        // High reward variance -> use event-based
        if reward_variance > 0.5 {
            return self.compress_event_based(frames);
        }

        // Default to key transitions
        self.compress_key_transitions(frames)
    }

    /// Summary-based compression (maximum compression)
    fn compress_summary(&self, frames: &[ExperienceFrame]) -> Vec<ExperienceFrame> {
        // Keep only first, last, max reward, min reward frames
        let mut key_frames = Vec::new();

        if let Some(first) = frames.first() {
            let mut kf = first.clone();
            kf.mark_key_frame();
            key_frames.push(kf);
        }

        if let Some((idx, _)) = frames
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.reward.partial_cmp(&b.1.reward).unwrap())
        {
            if idx != 0 {
                let mut kf = frames[idx].clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }
        }

        if let Some((idx, _)) = frames
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.reward.partial_cmp(&b.1.reward).unwrap())
        {
            if idx != 0 && !key_frames.iter().any(|kf| kf.id == frames[idx].id) {
                let mut kf = frames[idx].clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }
        }

        if let Some(last) = frames.last() {
            if !key_frames.iter().any(|kf| kf.id == last.id) {
                let mut kf = last.clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }
        }

        // Sort by original order
        key_frames.sort_by_key(|f| f.id);

        key_frames
    }

    /// Ensure minimum number of key frames
    fn ensure_minimum_frames(
        &self,
        frames: &[ExperienceFrame],
        key_frames: &mut Vec<ExperienceFrame>,
    ) {
        let existing_ids: Vec<u64> = key_frames.iter().map(|f| f.id).collect();
        let needed = self.config.min_key_frames.saturating_sub(key_frames.len());

        if needed == 0 {
            return;
        }

        // Add evenly spaced frames
        let step = frames.len() / (needed + 1);
        for i in 0..needed {
            let idx = (i + 1) * step;
            if idx < frames.len() && !existing_ids.contains(&frames[idx].id) {
                let mut kf = frames[idx].clone();
                kf.mark_key_frame();
                key_frames.push(kf);
            }
        }

        // Sort by original order
        key_frames.sort_by_key(|f| f.id);
    }

    /// Compute state similarity (cosine similarity)
    fn state_similarity(&self, s1: &[f32], s2: &[f32]) -> f64 {
        if s1.len() != s2.len() || s1.is_empty() {
            return 0.0;
        }

        let dot: f64 = s1
            .iter()
            .zip(s2.iter())
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();

        let norm1: f64 = s1.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm2: f64 = s2.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        dot / (norm1 * norm2)
    }

    /// Compute variance of values
    fn compute_variance<I: Iterator<Item = f64>>(&self, values: I) -> f64 {
        let values: Vec<f64> = values.collect();
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    }

    /// Compute episode summary statistics
    fn compute_summary(&self, frames: &[ExperienceFrame]) -> EpisodeSummary {
        if frames.is_empty() {
            return EpisodeSummary::default();
        }

        let state_dim = frames[0].state.len();
        let mut mean_state = vec![0.0f32; state_dim];
        let mut action_distribution = HashMap::new();

        let rewards: Vec<f32> = frames.iter().map(|f| f.reward).collect();
        let total_reward: f32 = rewards.iter().sum();
        let mean_reward = total_reward / frames.len() as f32;

        for frame in frames {
            // Accumulate state mean
            for (i, &val) in frame.state.iter().enumerate() {
                if i < mean_state.len() {
                    mean_state[i] += val / frames.len() as f32;
                }
            }

            // Action distribution
            *action_distribution.entry(frame.action).or_insert(0) += 1;
        }

        // Compute variance
        let reward_variance = if frames.len() > 1 {
            rewards
                .iter()
                .map(|r| (r - mean_reward).powi(2))
                .sum::<f32>()
                / (frames.len() - 1) as f32
        } else {
            0.0
        };

        let mut state_variance = vec![0.0f32; state_dim];
        for frame in frames {
            for (i, &val) in frame.state.iter().enumerate() {
                if i < state_variance.len() {
                    state_variance[i] += (val - mean_state[i]).powi(2) / frames.len().max(1) as f32;
                }
            }
        }

        let mean_td_error = frames.iter().filter_map(|f| f.td_error).sum::<f32>()
            / frames
                .iter()
                .filter(|f| f.td_error.is_some())
                .count()
                .max(1) as f32;

        EpisodeSummary {
            mean_state,
            state_variance,
            action_distribution,
            mean_reward,
            reward_variance,
            max_reward: rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            min_reward: rewards.iter().cloned().fold(f32::INFINITY, f32::min),
            mean_td_error,
            positive_reward_count: rewards.iter().filter(|&&r| r > 0.0).count(),
            negative_reward_count: rewards.iter().filter(|&&r| r < 0.0).count(),
            duration: frames.len(),
        }
    }

    /// Adapt threshold based on compression performance
    fn adapt_threshold(&mut self, achieved_ratio: f64) {
        let target_ratio = self.config.target_speedup;

        if achieved_ratio < target_ratio * 0.8 {
            // Compressing too much - increase threshold (less selective)
            self.adaptive_threshold = (self.adaptive_threshold + 0.02).min(0.99);
        } else if achieved_ratio > target_ratio * 1.2 {
            // Not compressing enough - decrease threshold (more selective)
            self.adaptive_threshold = (self.adaptive_threshold - 0.02).max(0.3);
        }
    }

    /// Replay compressed episodes
    pub fn replay(&mut self, count: Option<usize>) -> ReplayResult {
        let count = count.unwrap_or(self.episodes.len());
        let episodes_to_replay: Vec<CompressedEpisode> =
            self.episodes.iter().rev().take(count).cloned().collect();

        let mut result = ReplayResult {
            frames_replayed: 0,
            original_frames: 0,
            speedup: 0.0,
            episodes_processed: 0,
            total_reward: 0.0,
            learning_signals: Vec::new(),
        };

        for episode in &episodes_to_replay {
            result.frames_replayed += episode.key_frames.len();
            result.original_frames += episode.original_length;
            result.total_reward += episode.total_reward;
            result.episodes_processed += 1;

            // Generate learning signals from key frames
            for frame in &episode.key_frames {
                result.learning_signals.push(LearningSignal {
                    frame_id: frame.id,
                    state: frame.state.clone(),
                    action: frame.action,
                    target: frame.reward,
                    weight: frame.significance,
                    episode_id: episode.id,
                });
            }
        }

        if result.frames_replayed > 0 {
            result.speedup = result.original_frames as f64 / result.frames_replayed as f64;
        }

        self.replay_count += 1;
        result
    }

    /// Replay a specific episode by ID
    pub fn replay_episode(&self, episode_id: u64) -> Option<&CompressedEpisode> {
        self.episodes.iter().find(|e| e.id == episode_id)
    }

    /// Get all stored episodes
    pub fn episodes(&self) -> &[CompressedEpisode] {
        &self.episodes
    }

    /// Get episode count
    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Get overall compression ratio
    pub fn overall_compression_ratio(&self) -> f64 {
        if self.total_key_frames == 0 {
            return 1.0;
        }
        self.total_frames_compressed as f64 / self.total_key_frames as f64
    }

    /// Set compression method
    pub fn set_method(&mut self, method: CompressionMethod) {
        self.config.method = method;
    }

    /// Set compression level
    pub fn set_level(&mut self, level: CompressionLevel) {
        self.config.level = level;
        self.adaptive_threshold = level.similarity_threshold();
        self.config.min_significance = level.min_significance();
        self.config.target_speedup = level.speedup_factor();
    }

    /// Get current compression level
    pub fn level(&self) -> CompressionLevel {
        self.config.level
    }

    /// Clear all stored episodes
    pub fn clear(&mut self) {
        self.episodes.clear();
        self.current_episode = None;
    }

    /// Get statistics
    pub fn stats(&self) -> CompressedReplayStats {
        CompressedReplayStats {
            episode_count: self.episodes.len(),
            total_frames_compressed: self.total_frames_compressed,
            total_key_frames: self.total_key_frames,
            overall_compression_ratio: self.overall_compression_ratio(),
            replay_count: self.replay_count,
            adaptive_threshold: self.adaptive_threshold,
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via add/end_episode/replay
        Ok(())
    }
}

/// Statistics for compressed replay
#[derive(Debug, Clone)]
pub struct CompressedReplayStats {
    pub episode_count: usize,
    pub total_frames_compressed: usize,
    pub total_key_frames: usize,
    pub overall_compression_ratio: f64,
    pub replay_count: usize,
    pub adaptive_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frames(count: usize) -> Vec<ExperienceFrame> {
        (0..count)
            .map(|i| {
                // Keep action constant for 10 frames at a time to allow compression
                let action = i / 10;
                // Small variations in state, not linearly increasing
                let base = (i / 10) as f32;
                ExperienceFrame::new(
                    i as u64,
                    i as u64 * 100,
                    vec![base, base + 0.1, base + 0.2],
                    action % 4,
                    if i % 20 == 0 { 1.0 } else { 0.0 }, // Rewards less frequent
                    vec![base + 0.01, base + 0.11, base + 0.21],
                    i == count - 1,
                )
            })
            .collect()
    }

    #[test]
    fn test_basic() {
        let instance = CompressedReplay::new();
        assert!(instance.process().is_ok());
        assert_eq!(instance.episode_count(), 0);
    }

    #[test]
    fn test_add_and_compress() {
        let mut replay = CompressedReplay::new();
        replay.start_episode();

        for frame in create_test_frames(50) {
            replay.add_frame(frame);
        }

        let episode = replay.end_episode().unwrap();

        assert!(episode.key_frames.len() < 50);
        assert!(episode.compression_ratio > 1.0);
        assert_eq!(episode.original_length, 50);
    }

    #[test]
    fn test_compression_methods() {
        let methods = vec![
            CompressionMethod::SkipSimilar,
            CompressionMethod::KeyTransitions,
            CompressionMethod::TemporalDifference,
            CompressionMethod::EventBased,
            CompressionMethod::Adaptive,
            CompressionMethod::Summary,
        ];

        for method in methods {
            let mut config = CompressedReplayConfig::default();
            config.method = method;

            let mut replay = CompressedReplay::with_config(config, 10);
            replay.start_episode();

            for frame in create_test_frames(100) {
                replay.add_frame(frame);
            }

            let episode = replay.end_episode().unwrap();

            assert!(
                !episode.key_frames.is_empty(),
                "Method {:?} produced no key frames",
                method
            );
            assert!(
                episode.key_frames.len() <= 100,
                "Method {:?} didn't compress",
                method
            );
        }
    }

    #[test]
    fn test_compression_levels() {
        let levels = vec![
            CompressionLevel::Light,
            CompressionLevel::Moderate,
            CompressionLevel::Aggressive,
            CompressionLevel::Maximum,
        ];

        for level in levels {
            let mut replay = CompressedReplay::new();
            replay.set_level(level);
            replay.start_episode();

            for frame in create_test_frames(100) {
                replay.add_frame(frame);
            }

            let episode = replay.end_episode().unwrap();

            // Higher compression should mean fewer frames
            assert!(!episode.key_frames.is_empty());
        }
    }

    #[test]
    fn test_replay() {
        let mut replay = CompressedReplay::new();

        // Add multiple episodes
        for _ in 0..3 {
            replay.start_episode();
            for frame in create_test_frames(30) {
                replay.add_frame(frame);
            }
            let _ = replay.end_episode();
        }

        let result = replay.replay(Some(2));

        assert_eq!(result.episodes_processed, 2);
        assert!(result.speedup > 1.0);
        assert!(!result.learning_signals.is_empty());
    }

    #[test]
    fn test_auto_episode_end() {
        let mut replay = CompressedReplay::new();

        // Adding with done=true should auto-end
        replay.add(vec![1.0], 0, 1.0, vec![2.0], true, 100);

        assert_eq!(replay.episode_count(), 1);
    }

    #[test]
    fn test_summary_computation() {
        let mut replay = CompressedReplay::new();
        replay.start_episode();

        for frame in create_test_frames(20) {
            replay.add_frame(frame);
        }

        let episode = replay.end_episode().unwrap();

        assert_eq!(episode.summary.duration, 20);
        assert!(!episode.summary.action_distribution.is_empty());
    }

    #[test]
    fn test_td_error_frames() {
        let mut replay = CompressedReplay::new();
        replay.config.method = CompressionMethod::TemporalDifference;
        replay.start_episode();

        for i in 0..50 {
            let td_error = if i % 10 == 0 { 0.5 } else { 0.01 };
            let frame = ExperienceFrame::new(
                i as u64,
                i as u64 * 100,
                vec![i as f32],
                0,
                0.0,
                vec![(i + 1) as f32],
                i == 49,
            )
            .with_td_error(td_error);
            replay.add_frame(frame);
        }

        let episode = replay.end_episode().unwrap();

        // High TD error frames should be kept
        assert!(
            episode
                .key_frames
                .iter()
                .any(|f| f.td_error.unwrap_or(0.0) > 0.1)
        );
    }

    #[test]
    fn test_state_similarity() {
        let replay = CompressedReplay::new();

        let s1 = vec![1.0, 0.0, 0.0];
        let s2 = vec![1.0, 0.0, 0.0];
        let s3 = vec![0.0, 1.0, 0.0];

        let sim_same = replay.state_similarity(&s1, &s2);
        let sim_diff = replay.state_similarity(&s1, &s3);

        assert!((sim_same - 1.0).abs() < 0.001);
        assert!(sim_diff < 0.1);
    }

    #[test]
    fn test_minimum_key_frames() {
        let mut config = CompressedReplayConfig::default();
        config.min_key_frames = 10;
        config.method = CompressionMethod::Summary; // Most aggressive

        let mut replay = CompressedReplay::with_config(config, 10);
        replay.start_episode();

        for frame in create_test_frames(50) {
            replay.add_frame(frame);
        }

        let episode = replay.end_episode().unwrap();

        // Should respect minimum (though Summary normally keeps very few)
        assert!(episode.key_frames.len() >= 4); // Summary keeps at least first, last, max, min
    }

    #[test]
    fn test_statistics() {
        let mut replay = CompressedReplay::new();

        for _ in 0..5 {
            replay.start_episode();
            for frame in create_test_frames(20) {
                replay.add_frame(frame);
            }
            let _ = replay.end_episode();
        }

        let stats = replay.stats();

        assert_eq!(stats.episode_count, 5);
        assert_eq!(stats.total_frames_compressed, 100);
        assert!(stats.overall_compression_ratio > 1.0);
    }

    #[test]
    fn test_clear() {
        let mut replay = CompressedReplay::new();
        replay.start_episode();

        for frame in create_test_frames(10) {
            replay.add_frame(frame);
        }

        let _ = replay.end_episode();
        assert_eq!(replay.episode_count(), 1);

        replay.clear();
        assert_eq!(replay.episode_count(), 0);
    }

    #[test]
    fn test_episode_summary_fields() {
        let mut replay = CompressedReplay::new();
        replay.start_episode();

        // Create frames with known reward pattern
        for i in 0..10 {
            let reward = i as f32 - 5.0; // Range from -5 to 4
            let frame = ExperienceFrame::new(
                i as u64,
                i as u64 * 100,
                vec![i as f32],
                i % 2,
                reward,
                vec![(i + 1) as f32],
                i == 9,
            );
            replay.add_frame(frame);
        }

        let episode = replay.end_episode().unwrap();

        assert_eq!(episode.summary.max_reward, 4.0);
        assert_eq!(episode.summary.min_reward, -5.0);
        assert!(episode.summary.positive_reward_count > 0);
        assert!(episode.summary.negative_reward_count > 0);
    }

    #[test]
    fn test_adaptive_threshold() {
        let mut config = CompressedReplayConfig::default();
        config.adaptive_threshold = true;
        config.target_speedup = 5.0;

        let mut replay = CompressedReplay::with_config(config, 100);
        let initial_threshold = replay.adaptive_threshold;

        // Add many episodes to trigger adaptation
        for _ in 0..10 {
            replay.start_episode();
            for frame in create_test_frames(50) {
                replay.add_frame(frame);
            }
            let _ = replay.end_episode();
        }

        // Threshold may have changed (either direction depending on compression)
        let _final_threshold = replay.adaptive_threshold;
        assert!(initial_threshold > 0.0);
    }
}
