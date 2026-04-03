//! BERT/FinBERT Sentiment Embeddings via Candle
//!
//! Provides structured sentiment analysis of financial news and text data
//! using pre-trained BERT-family models (FinBERT, DistilBERT) running
//! natively in Rust via the Candle ML framework.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   BERT Sentiment Pipeline                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
//! │  │Raw Text  │──▶│Tokenizer │──▶│  BERT    │──▶│Sentiment│ │
//! │  │(News,    │   │(WordPiece│   │ Forward  │   │  Head   │ │
//! │  │ tweets)  │   │ encoder) │   │  Pass    │   │(Softmax)│ │
//! │  └──────────┘   └──────────┘   └──────────┘   └────┬────┘ │
//! │                                                     │      │
//! │                                    ┌────────────────┘      │
//! │                                    ▼                       │
//! │                           ┌─────────────────┐              │
//! │                           │ SentimentResult │              │
//! │                           │ • score [-1, 1] │              │
//! │                           │ • confidence    │              │
//! │                           │ • embedding vec │              │
//! │                           └────────┬────────┘              │
//! │                                    │                       │
//! │                                    ▼                       │
//! │                           ┌─────────────────┐              │
//! │                           │ SentimentFusion │              │
//! │                           │ (Thalamus)      │              │
//! │                           └─────────────────┘              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration Points
//!
//! - **Input from**: `thalamus::sources::NewsSource`, `thalamus::sources::CryptoPanicClient`,
//!   `thalamus::sources::NewsApiClient` — raw text from news feeds
//! - **Output to**: `thalamus::fusion::SentimentFusion` — structured sentiment scores,
//!   `hippocampus::memory` — sentiment embeddings for regime-aware memory retrieval
//! - **Model weights**: Loaded from safetensors files (FinBERT fine-tuned on financial text)
//!
//! # Model Support
//!
//! - **FinBERT** (`ProsusAI/finbert`): Fine-tuned on financial news, 3-class
//!   (positive/negative/neutral). Best for market-relevant sentiment.
//! - **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`): Lighter
//!   2-class model. Faster inference, acceptable for high-frequency use.
//! - **Custom**: Any BERT-architecture model loadable from safetensors.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_neuromorphic::thalamus::sources::bert_sentiment::*;
//!
//! let config = BertSentimentConfig::finbert();
//! let mut analyzer = BertSentimentAnalyzer::with_config(config)?;
//!
//! // Analyse a single headline
//! let result = analyzer.analyze("Fed raises interest rates by 50bps")?;
//! println!("Sentiment: {:.2}, Confidence: {:.2}", result.score, result.confidence);
//!
//! // Batch analysis
//! let texts = vec![
//!     "Bitcoin surges past $100k on ETF inflows",
//!     "Regulators warn of systemic crypto risk",
//! ];
//! let results = analyzer.analyze_batch(&texts)?;
//! ```

use crate::common::{Error, Result};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Dropout, Embedding, LayerNorm, Linear, VarBuilder, VarMap, layer_norm, linear};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Pre-trained model variant to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BertModelVariant {
    /// ProsusAI FinBERT — 3-class financial sentiment (positive/negative/neutral).
    /// Best accuracy for financial news but heavier (~440MB).
    FinBert,

    /// DistilBERT SST-2 — 2-class general sentiment (positive/negative).
    /// ~2× faster inference, ~260MB. Good enough for high-frequency pipeline.
    DistilBert,

    /// Custom model specified by directory path containing:
    /// - `config.json` (model config)
    /// - `model.safetensors` (weights)
    /// - `tokenizer.json` (tokenizer config)
    /// - `vocab.txt` (vocabulary)
    Custom {
        /// Path to the model directory.
        model_dir: String,
        /// Number of output classes.
        num_labels: usize,
        /// Mapping from label index to human-readable label.
        label_map: Vec<String>,
    },
}

impl BertModelVariant {
    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        match self {
            Self::FinBert => 3,
            Self::DistilBert => 2,
            Self::Custom { num_labels, .. } => *num_labels,
        }
    }

    /// Human-readable label for a class index.
    pub fn label(&self, index: usize) -> &str {
        match self {
            Self::FinBert => match index {
                0 => "positive",
                1 => "negative",
                2 => "neutral",
                _ => "unknown",
            },
            Self::DistilBert => match index {
                0 => "negative",
                1 => "positive",
                _ => "unknown",
            },
            Self::Custom { label_map, .. } => label_map
                .get(index)
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        }
    }

    /// Short identifier string for this variant.
    pub fn id(&self) -> &str {
        match self {
            Self::FinBert => "finbert",
            Self::DistilBert => "distilbert",
            Self::Custom { .. } => "custom",
        }
    }
}

impl std::fmt::Display for BertModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FinBert => write!(f, "FinBERT (3-class financial sentiment)"),
            Self::DistilBert => write!(f, "DistilBERT (2-class general sentiment)"),
            Self::Custom {
                model_dir,
                num_labels,
                ..
            } => {
                write!(f, "Custom BERT ({num_labels}-class, dir={model_dir})")
            }
        }
    }
}

/// Configuration for the BERT sentiment analyzer.
#[derive(Debug, Clone)]
pub struct BertSentimentConfig {
    /// Model variant to use.
    pub model_variant: BertModelVariant,
    /// Path to the model directory (overrides variant default).
    pub model_dir: Option<String>,
    /// Maximum sequence length (tokens). BERT supports up to 512.
    pub max_seq_length: usize,
    /// Whether to extract [CLS] embeddings alongside logits.
    pub extract_embeddings: bool,
    /// Embedding dimension for extracted vectors (must match hidden_size).
    pub embedding_dim: usize,
    /// Inference device.
    pub device: InferenceDevice,
    /// Maximum batch size for a single forward pass.
    pub max_batch_size: usize,
    /// Whether to cache tokenization results.
    pub cache_tokenization: bool,
    /// EMA decay factor for score smoothing.
    pub ema_decay: f64,
    /// Rolling window size for statistics.
    pub window_size: usize,
    /// Minimum confidence threshold for valid predictions.
    pub min_confidence: f64,
}

/// Device for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDevice {
    /// CPU inference.
    Cpu,
    /// CUDA GPU inference.
    Cuda(usize),
    /// Apple Metal GPU inference.
    Metal,
}

impl Default for BertSentimentConfig {
    fn default() -> Self {
        Self::finbert()
    }
}

impl BertSentimentConfig {
    /// Create a FinBERT configuration.
    pub fn finbert() -> Self {
        Self {
            model_variant: BertModelVariant::FinBert,
            model_dir: None,
            max_seq_length: 128,
            extract_embeddings: true,
            embedding_dim: 768,
            device: InferenceDevice::Cpu,
            max_batch_size: 32,
            cache_tokenization: true,
            ema_decay: 0.1,
            window_size: 50,
            min_confidence: 0.3,
        }
    }

    /// Create a DistilBERT configuration.
    pub fn distilbert() -> Self {
        Self {
            model_variant: BertModelVariant::DistilBert,
            model_dir: None,
            max_seq_length: 128,
            extract_embeddings: true,
            embedding_dim: 768,
            device: InferenceDevice::Cpu,
            max_batch_size: 64,
            cache_tokenization: true,
            ema_decay: 0.1,
            window_size: 50,
            min_confidence: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A single token from the tokenizer output.
#[derive(Debug, Clone)]
pub struct Token {
    /// Integer token ID in the vocabulary.
    pub id: u32,
    /// Text representation of the token.
    pub text: String,
    /// Whether this is a continuation token (starts with ##).
    pub is_continuation: bool,
}

/// Tokenized input ready for the BERT model.
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    /// Token IDs: [CLS] + tokens + [SEP] + [PAD]*
    pub input_ids: Vec<u32>,
    /// Attention mask: 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u32>,
    /// Token type IDs: 0 for first segment (always 0 for single-sentence
    /// classification).
    pub token_type_ids: Vec<u32>,
    /// Original text before tokenization.
    pub original_text: String,
    /// Number of real tokens (including [CLS] and [SEP]).
    pub num_tokens: usize,
    /// Whether the input was truncated.
    pub was_truncated: bool,
}

/// BERT WordPiece tokenizer.
///
/// Handles text preprocessing, sub-word tokenization, and special token
/// insertion ([CLS], [SEP], [PAD]).
pub struct BertTokenizer {
    /// Vocabulary: token string -> token ID.
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: token ID -> token string.
    id_to_token: Vec<String>,
    /// Maximum sequence length.
    max_seq_length: usize,
    /// Special token IDs.
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    unk_id: u32,
    /// Whether to lowercase input (true for uncased models).
    do_lower_case: bool,
}

impl BertTokenizer {
    /// Create a new tokenizer with an empty vocabulary.
    ///
    /// The tokenizer is not ready for use until `load_vocab()` or
    /// `load_from_json()` is called.
    pub fn new(max_seq_length: usize) -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            max_seq_length,
            cls_id: 101,
            sep_id: 102,
            pad_id: 0,
            unk_id: 100,
            do_lower_case: true,
        }
    }

    /// Load vocabulary from a `vocab.txt` file (one token per line).
    ///
    /// This is the standard BERT vocabulary format where each line contains
    /// a single token and the line number (0-based) is the token ID.
    pub fn load_vocab(&mut self, vocab_path: &str) -> Result<()> {
        let content = std::fs::read_to_string(vocab_path).map_err(|e| {
            Error::Configuration(format!("Failed to read vocab file '{}': {}", vocab_path, e))
        })?;

        self.vocab.clear();
        self.id_to_token.clear();

        for (id, line) in content.lines().enumerate() {
            let token = line.trim_end().to_string();
            if token.is_empty() {
                // Some vocab files have blank lines; preserve the ID slot
                self.id_to_token.push(String::new());
                continue;
            }
            self.vocab.insert(token.clone(), id as u32);
            self.id_to_token.push(token);
        }

        // Resolve special token IDs from the vocabulary
        self.cls_id = self.vocab.get("[CLS]").copied().unwrap_or(101);
        self.sep_id = self.vocab.get("[SEP]").copied().unwrap_or(102);
        self.pad_id = self.vocab.get("[PAD]").copied().unwrap_or(0);
        self.unk_id = self.vocab.get("[UNK]").copied().unwrap_or(100);

        if self.vocab.is_empty() {
            return Err(Error::Configuration(
                "Vocabulary file was empty or contained no valid tokens".into(),
            ));
        }

        Ok(())
    }

    /// Load tokenizer configuration from a HuggingFace `tokenizer.json` file.
    ///
    /// Extracts the vocabulary from the `model.vocab` field of the JSON
    /// tokenizer configuration.
    pub fn load_from_json(&mut self, tokenizer_path: &str) -> Result<()> {
        let content = std::fs::read_to_string(tokenizer_path).map_err(|e| {
            Error::Configuration(format!(
                "Failed to read tokenizer JSON '{}': {}",
                tokenizer_path, e
            ))
        })?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| Error::Configuration(format!("Failed to parse tokenizer JSON: {}", e)))?;

        // HuggingFace tokenizer.json has vocab in model.vocab as {token: id}
        let vocab_obj = json
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::Configuration("tokenizer.json missing 'model.vocab' object".into())
            })?;

        self.vocab.clear();
        self.id_to_token.clear();

        // Determine max ID to size the reverse mapping
        let max_id = vocab_obj
            .values()
            .filter_map(|v| v.as_u64())
            .max()
            .unwrap_or(0) as usize;

        self.id_to_token.resize(max_id + 1, String::new());

        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                let id = id as u32;
                self.vocab.insert(token.clone(), id);
                if (id as usize) < self.id_to_token.len() {
                    self.id_to_token[id as usize] = token.clone();
                }
            }
        }

        // Resolve special tokens
        self.cls_id = self.vocab.get("[CLS]").copied().unwrap_or(101);
        self.sep_id = self.vocab.get("[SEP]").copied().unwrap_or(102);
        self.pad_id = self.vocab.get("[PAD]").copied().unwrap_or(0);
        self.unk_id = self.vocab.get("[UNK]").copied().unwrap_or(100);

        if self.vocab.is_empty() {
            return Err(Error::Configuration(
                "tokenizer.json contained no vocabulary entries".into(),
            ));
        }

        Ok(())
    }

    /// Tokenize a single text into model input using WordPiece.
    ///
    /// Steps:
    /// 1. Normalize: lowercase (if uncased), strip accents, clean whitespace
    /// 2. Pre-tokenize: split on whitespace and punctuation
    /// 3. WordPiece: for each word, greedily match longest vocab prefix
    /// 4. Truncate to `max_seq_length - 2` (room for [CLS] and [SEP])
    /// 5. Build `input_ids`, `attention_mask`, `token_type_ids`
    pub fn tokenize(&self, text: &str) -> TokenizedInput {
        let processed = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Pre-tokenize: split on whitespace and punctuation boundaries
        let words = self.pre_tokenize(&processed);

        // WordPiece tokenize each word
        let max_tokens = self.max_seq_length.saturating_sub(2);
        let mut wp_ids: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut was_truncated = false;

        'outer: for word in &words {
            let subtokens = self.wordpiece_tokenize(word);
            for st in subtokens {
                if wp_ids.len() >= max_tokens {
                    was_truncated = true;
                    break 'outer;
                }
                wp_ids.push(st);
            }
        }

        // Build input_ids: [CLS] + tokens + [SEP] + [PAD]*
        let mut input_ids = Vec::with_capacity(self.max_seq_length);
        input_ids.push(self.cls_id);
        input_ids.extend_from_slice(&wp_ids);
        input_ids.push(self.sep_id);

        let num_tokens = input_ids.len();

        // Pad to max_seq_length
        while input_ids.len() < self.max_seq_length {
            input_ids.push(self.pad_id);
        }

        let attention_mask: Vec<u32> = (0..self.max_seq_length)
            .map(|i| if i < num_tokens { 1 } else { 0 })
            .collect();

        let token_type_ids = vec![0u32; self.max_seq_length];

        TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            original_text: text.to_string(),
            num_tokens,
            was_truncated,
        }
    }

    /// Tokenize a batch of texts.
    pub fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        texts.iter().map(|t| self.tokenize(t)).collect()
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Check if vocabulary is loaded.
    pub fn is_loaded(&self) -> bool {
        !self.vocab.is_empty()
    }

    // -- private helpers --

    /// Pre-tokenize text: split on whitespace and punctuation boundaries.
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            } else if ch.is_ascii_punctuation() || is_cjk_char(ch) {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
                tokens.push(ch.to_string());
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// WordPiece tokenization of a single pre-tokenized word.
    ///
    /// Uses the greedy longest-match-first algorithm:
    /// - Try to match the longest prefix of the remaining string in vocab
    /// - If matched, consume that prefix and continue with `##` prefix
    /// - If no match at all for a character, emit [UNK] for the whole word
    fn wordpiece_tokenize(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return vec![];
        }

        // Fast path: check if the whole word is in vocab
        if let Some(&id) = self.vocab.get(word) {
            return vec![id];
        }

        // If vocab is empty (not loaded), fall back to UNK per word
        if self.vocab.is_empty() {
            return vec![self.unk_id];
        }

        let chars: Vec<char> = word.chars().collect();
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    format!("##{}", chars[start..end].iter().collect::<String>())
                };

                if let Some(&id) = self.vocab.get(&substr) {
                    tokens.push(id);
                    start = end;
                    found = true;
                    break;
                }
                end -= 1;
            }

            if !found {
                // Character not matchable; emit UNK for entire word
                return vec![self.unk_id];
            }
        }

        tokens
    }
}

/// Check if a character is a CJK ideograph.
fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    matches!(cp,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F
    )
}

// ---------------------------------------------------------------------------
// Model Architecture (Candle)
// ---------------------------------------------------------------------------

/// BERT model hyperparameters (loaded from config.json).
#[derive(Debug, Clone)]
pub struct BertModelConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// Intermediate (feed-forward) size.
    pub intermediate_size: usize,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Number of output labels for classification.
    pub num_labels: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f64,
    /// Hidden dropout probability.
    pub hidden_dropout_prob: f64,
    /// Attention dropout probability.
    pub attention_probs_dropout_prob: f64,
}

impl Default for BertModelConfig {
    fn default() -> Self {
        // BERT-base defaults
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            num_labels: 3,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
        }
    }
}

impl BertModelConfig {
    /// FinBERT configuration.
    pub fn finbert() -> Self {
        Self {
            num_labels: 3,
            ..Default::default()
        }
    }

    /// DistilBERT configuration (6 layers instead of 12).
    pub fn distilbert() -> Self {
        Self {
            num_hidden_layers: 6,
            num_labels: 2,
            ..Default::default()
        }
    }

    /// Load configuration from a HuggingFace `config.json` file.
    pub fn from_json(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            Error::Configuration(format!("Failed to read config.json '{}': {}", path, e))
        })?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| Error::Configuration(format!("Failed to parse config.json: {}", e)))?;

        let get_usize = |key: &str, default: usize| -> usize {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_f64 = |key: &str, default: f64| -> f64 {
            json.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };

        // Determine num_labels from id2label or num_labels field
        let num_labels = if let Some(id2label) = json.get("id2label").and_then(|v| v.as_object()) {
            id2label.len()
        } else {
            get_usize("num_labels", 3)
        };

        Ok(Self {
            vocab_size: get_usize("vocab_size", 30522),
            hidden_size: get_usize("hidden_size", 768),
            num_hidden_layers: get_usize("num_hidden_layers", 12),
            num_attention_heads: get_usize("num_attention_heads", 12),
            intermediate_size: get_usize("intermediate_size", 3072),
            max_position_embeddings: get_usize("max_position_embeddings", 512),
            num_labels,
            layer_norm_eps: get_f64("layer_norm_eps", 1e-12),
            hidden_dropout_prob: get_f64("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob: get_f64("attention_probs_dropout_prob", 0.1),
        })
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// -- Candle sub-modules --

/// BERT Embeddings: token + position + token_type + LayerNorm + Dropout.
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertEmbeddings {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            2, // BERT always has 2 token types
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();

        // Position IDs: [0, 1, ..., seq_len-1]
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?;

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        let type_emb = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = word_emb.broadcast_add(&pos_emb)?.broadcast_add(&type_emb)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, train)
    }
}

/// BERT Self-Attention: Q/K/V projections, scaled dot-product attention,
/// output projection.
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    dropout: Dropout,
}

impl BertSelfAttention {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden = config.hidden_size;
        let query = linear(hidden, hidden, vb.pp("query"))?;
        let key = linear(hidden, hidden, vb.pp("key"))?;
        let value = linear(hidden, hidden, vb.pp("value"))?;
        let dropout = Dropout::new(config.attention_probs_dropout_prob as f32);

        Ok(Self {
            query,
            key,
            value,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            scale: (config.head_dim() as f64).powf(-0.5),
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut attn_scores = (q.matmul(&k_t)? * self.scale)?;

        // Apply attention mask (additive): mask positions with -10000
        if let Some(mask) = attention_mask {
            attn_scores = attn_scores.broadcast_add(mask)?;
        }

        let attn_probs = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;
        let attn_probs = self.dropout.forward(&attn_probs, train)?;

        // Apply attention to values
        let context = attn_probs.contiguous()?.matmul(&v)?;
        let context = context.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(context)
    }
}

/// BERT Self-Output: dense projection + LayerNorm + residual.
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let hidden = self.dense.forward(hidden_states)?;
        let hidden = self.dropout.forward(&hidden, train)?;
        self.layer_norm.forward(&(hidden + input_tensor)?)
    }
}

/// BERT Attention: self-attention + output projection.
struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attention = BertSelfAttention::new(config, vb.pp("self"))?;
        let output = BertSelfOutput::new(config, vb.pp("output"))?;
        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let self_output = self
            .self_attention
            .forward(hidden_states, attention_mask, train)?;
        self.output.forward(&self_output, hidden_states, train)
    }
}

/// BERT Intermediate: hidden -> intermediate with GELU activation.
struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        self.dense.forward(hidden_states)?.gelu_erf()
    }
}

/// BERT Output: intermediate -> hidden with LayerNorm + residual.
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob as f32);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let hidden = self.dense.forward(hidden_states)?;
        let hidden = self.dropout.forward(&hidden, train)?;
        self.layer_norm.forward(&(hidden + input_tensor)?)
    }
}

/// A single BERT Transformer Encoder Layer.
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let attention = BertAttention::new(config, vb.pp("attention"))?;
        let intermediate = BertIntermediate::new(config, vb.pp("intermediate"))?;
        let output = BertOutput::new(config, vb.pp("output"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let attn_output = self
            .attention
            .forward(hidden_states, attention_mask, train)?;
        let intermediate_output = self.intermediate.forward(&attn_output)?;
        self.output
            .forward(&intermediate_output, &attn_output, train)
    }
}

/// BERT Encoder: stack of transformer layers.
struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = BertLayer::new(config, vb.pp(format!("layer.{}", i)))?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<Tensor> {
        let mut output = hidden_states.clone();
        for layer in &self.layers {
            output = layer.forward(&output, attention_mask, train)?;
        }
        Ok(output)
    }
}

/// BERT Pooler: takes the [CLS] token output and applies a dense + tanh.
struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    fn new(config: &BertModelConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        // Take the [CLS] token (index 0) from each sequence
        let cls_token = hidden_states.i((.., 0))?;
        self.dense.forward(&cls_token)?.tanh()
    }
}

// ---------------------------------------------------------------------------
// Top-level BERT Model
// ---------------------------------------------------------------------------

/// Raw model output from a forward pass.
pub struct ModelOutput {
    /// Classification logits: `[batch_size, num_labels]`.
    pub logits: Vec<Vec<f64>>,
    /// Optional [CLS] embeddings: `[batch_size, hidden_size]`.
    /// Only populated when `extract_embeddings` is true.
    pub embeddings: Option<Vec<Vec<f32>>>,
}

/// The full BERT model for sequence classification.
///
/// Architecture: Embeddings -> Encoder (N layers) -> Pooler -> Classifier
pub struct BertModel {
    /// Model configuration.
    config: BertModelConfig,
    /// Whether model weights are loaded.
    loaded: bool,
    /// Candle device for inference.
    candle_device: Device,
    /// Logical device setting.
    #[allow(dead_code)]
    device: InferenceDevice,
    /// VarMap holding all model parameters (kept alive for the model's
    /// lifetime).
    #[allow(dead_code)]
    var_map: Option<VarMap>,
    /// Embeddings sub-module.
    embeddings: Option<BertEmbeddings>,
    /// Encoder sub-module.
    encoder: Option<BertEncoder>,
    /// Pooler sub-module.
    pooler: Option<BertPooler>,
    /// Classification head.
    classifier: Option<Linear>,
}

impl BertModel {
    /// Create a new unloaded model with the given configuration.
    pub fn new(config: BertModelConfig, device: InferenceDevice) -> Self {
        Self {
            config,
            loaded: false,
            candle_device: Self::resolve_device(device),
            device,
            var_map: None,
            embeddings: None,
            encoder: None,
            pooler: None,
            classifier: None,
        }
    }

    /// Resolve an InferenceDevice to a Candle Device.
    fn resolve_device(device: InferenceDevice) -> Device {
        match device {
            InferenceDevice::Cpu => Device::Cpu,
            InferenceDevice::Cuda(ordinal) => {
                Device::cuda_if_available(ordinal).unwrap_or(Device::Cpu)
            }
            InferenceDevice::Metal => {
                // Candle metal support; fall back to CPU if unavailable
                Device::Cpu
            }
        }
    }

    /// Initialize the model architecture with random weights.
    ///
    /// This creates all layers and allocates parameter tensors. Useful for
    /// testing or when you plan to load weights from safetensors afterwards.
    pub fn initialize(&mut self) -> Result<()> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &self.candle_device);

        let embeddings =
            BertEmbeddings::new(&self.config, vb.pp("bert.embeddings")).map_err(|e| {
                Error::Configuration(format!("Failed to initialize BERT embeddings: {}", e))
            })?;
        let encoder = BertEncoder::new(&self.config, vb.pp("bert.encoder")).map_err(|e| {
            Error::Configuration(format!("Failed to initialize BERT encoder: {}", e))
        })?;
        let pooler = BertPooler::new(&self.config, vb.pp("bert.pooler")).map_err(|e| {
            Error::Configuration(format!("Failed to initialize BERT pooler: {}", e))
        })?;
        let classifier = linear(
            self.config.hidden_size,
            self.config.num_labels,
            vb.pp("classifier"),
        )
        .map_err(|e| Error::Configuration(format!("Failed to initialize classifier: {}", e)))?;

        self.embeddings = Some(embeddings);
        self.encoder = Some(encoder);
        self.pooler = Some(pooler);
        self.classifier = Some(classifier);
        self.var_map = Some(var_map);

        Ok(())
    }

    /// Load model weights from a safetensors file.
    ///
    /// This initializes the model architecture and loads pre-trained weights
    /// from the specified path. The safetensors file should contain weights
    /// using HuggingFace naming conventions (e.g.
    /// `bert.embeddings.word_embeddings.weight`,
    /// `bert.encoder.layer.0.attention.self.query.weight`, etc.).
    pub fn load_weights(&mut self, weights_path: &str) -> Result<()> {
        // Load safetensors data into a HashMap<String, Tensor>
        let tensors =
            candle_core::safetensors::load(weights_path, &self.candle_device).map_err(|e| {
                Error::Configuration(format!(
                    "Failed to load safetensors '{}': {}",
                    weights_path, e
                ))
            })?;

        // Build model layers directly from loaded tensors.
        // VarBuilder::from_tensors maps tensor names to layer parameters by
        // prefix, so "bert.embeddings.word_embeddings.weight" is resolved by
        // vb.pp("bert.embeddings").pp("word_embeddings").
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &self.candle_device);

        let embeddings = BertEmbeddings::new(&self.config, vb.pp("bert.embeddings"))
            .map_err(|e| Error::Configuration(format!("Failed to build BERT embeddings: {}", e)))?;
        let encoder = BertEncoder::new(&self.config, vb.pp("bert.encoder"))
            .map_err(|e| Error::Configuration(format!("Failed to build BERT encoder: {}", e)))?;
        let pooler = BertPooler::new(&self.config, vb.pp("bert.pooler"))
            .map_err(|e| Error::Configuration(format!("Failed to build BERT pooler: {}", e)))?;
        let classifier = linear(
            self.config.hidden_size,
            self.config.num_labels,
            vb.pp("classifier"),
        )
        .map_err(|e| Error::Configuration(format!("Failed to build classifier head: {}", e)))?;

        self.embeddings = Some(embeddings);
        self.encoder = Some(encoder);
        self.pooler = Some(pooler);
        self.classifier = Some(classifier);
        self.var_map = None;
        self.loaded = true;

        Ok(())
    }

    /// Run a forward pass on tokenized input.
    ///
    /// Returns `ModelOutput` with logits and optional [CLS] embeddings.
    pub fn forward(
        &self,
        input_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
        token_type_ids: &[Vec<u32>],
        extract_embeddings: bool,
    ) -> Result<ModelOutput> {
        if !self.loaded {
            return Err(Error::Configuration(
                "Model weights not loaded — call load_weights() first".into(),
            ));
        }

        let embeddings_layer = self
            .embeddings
            .as_ref()
            .ok_or_else(|| Error::Internal("Embeddings layer not initialized".into()))?;
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Internal("Encoder not initialized".into()))?;
        let pooler = self
            .pooler
            .as_ref()
            .ok_or_else(|| Error::Internal("Pooler not initialized".into()))?;
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| Error::Internal("Classifier not initialized".into()))?;

        let batch_size = input_ids.len();
        if batch_size == 0 {
            return Ok(ModelOutput {
                logits: Vec::new(),
                embeddings: None,
            });
        }

        let device = &self.candle_device;

        // Convert input vectors to Candle tensors
        let input_ids_flat: Vec<u32> = input_ids.iter().flatten().copied().collect();
        let seq_len = input_ids[0].len();
        let input_ids_tensor = Tensor::from_vec(input_ids_flat, (batch_size, seq_len), device)
            .map_err(|e| Error::Internal(format!("Failed to create input_ids tensor: {}", e)))?;

        let type_ids_flat: Vec<u32> = token_type_ids.iter().flatten().copied().collect();
        let type_ids_tensor = Tensor::from_vec(type_ids_flat, (batch_size, seq_len), device)
            .map_err(|e| {
                Error::Internal(format!("Failed to create token_type_ids tensor: {}", e))
            })?;

        // Build additive attention mask: 0 for attend, -10000 for ignore
        let mask_flat: Vec<f32> = attention_mask
            .iter()
            .flatten()
            .map(|&m| if m == 1 { 0.0f32 } else { -10000.0f32 })
            .collect();
        let attn_mask = Tensor::from_vec(mask_flat, (batch_size, seq_len), device)
            .map_err(|e| Error::Internal(format!("Failed to create attention mask: {}", e)))?;
        // Reshape to (batch, 1, 1, seq) for broadcasting over heads and query positions
        let attn_mask = attn_mask
            .unsqueeze(1)
            .and_then(|t| t.unsqueeze(1))
            .map_err(|e| Error::Internal(format!("Failed to reshape attention mask: {}", e)))?;

        // Forward pass through BERT
        let embedded = embeddings_layer
            .forward(&input_ids_tensor, &type_ids_tensor, false)
            .map_err(|e| Error::Internal(format!("Embeddings forward failed: {}", e)))?;

        let encoded = encoder
            .forward(&embedded, Some(&attn_mask), false)
            .map_err(|e| Error::Internal(format!("Encoder forward failed: {}", e)))?;

        // Pooler: [CLS] token -> dense + tanh
        let pooled = pooler
            .forward(&encoded)
            .map_err(|e| Error::Internal(format!("Pooler forward failed: {}", e)))?;

        // Classification head
        let logits_tensor = classifier
            .forward(&pooled)
            .map_err(|e| Error::Internal(format!("Classifier forward failed: {}", e)))?;

        // Convert logits to Vec<Vec<f64>>
        let logits_data: Vec<f32> = logits_tensor
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(|e| Error::Internal(format!("Failed to extract logits: {}", e)))?;

        let num_labels = self.config.num_labels;
        let logits: Vec<Vec<f64>> = logits_data
            .chunks(num_labels)
            .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
            .collect();

        // Optionally extract [CLS] embeddings
        let embeddings_out = if extract_embeddings {
            let cls_emb: Vec<f32> = pooled
                .flatten_all()
                .and_then(|t| t.to_vec1())
                .map_err(|e| Error::Internal(format!("Failed to extract CLS embeddings: {}", e)))?;
            let hidden_size = self.config.hidden_size;
            let emb_vecs: Vec<Vec<f32>> = cls_emb
                .chunks(hidden_size)
                .map(|chunk| chunk.to_vec())
                .collect();
            Some(emb_vecs)
        } else {
            None
        };

        Ok(ModelOutput {
            logits,
            embeddings: embeddings_out,
        })
    }

    /// Hot-reload model weights from a new safetensors file.
    ///
    /// This atomically swaps all layer weights without dropping the model
    /// struct.  If loading fails the model is left in an **unloaded** state
    /// (`is_loaded() == false`) so the caller can retry or fall back.
    ///
    /// # Steps
    ///
    /// 1. Load new safetensors tensors from `weights_path`.
    /// 2. Build fresh layer instances (`BertEmbeddings`, `BertEncoder`,
    ///    `BertPooler`, classifier) from the new tensors.
    /// 3. Swap old layers for the new ones (the old tensors are dropped).
    /// 4. Mark the model as loaded.
    ///
    /// Because the swap is not wrapped in a mutex this method requires
    /// `&mut self`.  For concurrent hot-reload behind an `Arc<RwLock<…>>`
    /// the caller should acquire a write-lock before calling.
    pub fn hot_reload_weights(&mut self, weights_path: &str) -> Result<()> {
        // Mark as unloaded during the swap so concurrent readers (if any
        // exist through unsafe means) see the model as unavailable.
        self.loaded = false;

        // Load new tensors
        let tensors =
            candle_core::safetensors::load(weights_path, &self.candle_device).map_err(|e| {
                Error::Configuration(format!(
                    "Hot-reload: failed to load safetensors '{}': {}",
                    weights_path, e
                ))
            })?;

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &self.candle_device);

        let embeddings =
            BertEmbeddings::new(&self.config, vb.pp("bert.embeddings")).map_err(|e| {
                Error::Configuration(format!(
                    "Hot-reload: failed to build BERT embeddings: {}",
                    e
                ))
            })?;
        let encoder = BertEncoder::new(&self.config, vb.pp("bert.encoder")).map_err(|e| {
            Error::Configuration(format!("Hot-reload: failed to build BERT encoder: {}", e))
        })?;
        let pooler = BertPooler::new(&self.config, vb.pp("bert.pooler")).map_err(|e| {
            Error::Configuration(format!("Hot-reload: failed to build BERT pooler: {}", e))
        })?;
        let classifier = linear(
            self.config.hidden_size,
            self.config.num_labels,
            vb.pp("classifier"),
        )
        .map_err(|e| {
            Error::Configuration(format!(
                "Hot-reload: failed to build classifier head: {}",
                e
            ))
        })?;

        // Atomic swap — old layers are dropped here
        self.embeddings = Some(embeddings);
        self.encoder = Some(encoder);
        self.pooler = Some(pooler);
        self.classifier = Some(classifier);
        self.var_map = None;
        self.loaded = true;

        Ok(())
    }

    /// Check if model weights are loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get the model configuration.
    pub fn config(&self) -> &BertModelConfig {
        &self.config
    }

    /// Get the approximate number of parameters in the model.
    pub fn num_parameters(&self) -> usize {
        let c = &self.config;
        let embedding_params = c.vocab_size * c.hidden_size
            + c.max_position_embeddings * c.hidden_size
            + 2 * c.hidden_size; // token_type (2 x hidden) + LayerNorm (2 x hidden)
        let attention_params = 4 * c.hidden_size * c.hidden_size + 4 * c.hidden_size; // Q, K, V, O weights + biases
        let ffn_params = c.hidden_size * c.intermediate_size
            + c.intermediate_size
            + c.intermediate_size * c.hidden_size
            + c.hidden_size; // two dense layers + biases
        let layer_norm_params = 4 * c.hidden_size; // two LayerNorms per layer
        let layer_params = attention_params + ffn_params + layer_norm_params;
        let pooler_params = c.hidden_size * c.hidden_size + c.hidden_size;
        let classifier_params = c.hidden_size * c.num_labels + c.num_labels;

        embedding_params + c.num_hidden_layers * layer_params + pooler_params + classifier_params
    }
}

// ---------------------------------------------------------------------------
// Sentiment Result Types
// ---------------------------------------------------------------------------

/// Result from analyzing a single text.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// Original input text.
    pub text: String,
    /// Scalar sentiment score in [-1, 1].
    /// -1 = maximally negative, 0 = neutral, +1 = maximally positive.
    pub score: f64,
    /// Prediction confidence in [0, 1].
    /// Defined as the maximum class probability.
    pub confidence: f64,
    /// Human-readable sentiment label.
    pub label: String,
    /// Index of the predicted class.
    pub label_index: usize,
    /// Per-class probabilities (after softmax).
    pub class_probabilities: Vec<f64>,
    /// Optional [CLS] embedding vector for downstream storage.
    pub embedding: Option<Vec<f32>>,
    /// Whether this prediction has low confidence (below threshold).
    pub low_confidence: bool,
    /// Number of tokens used.
    pub num_tokens: usize,
    /// Whether the input was truncated.
    pub was_truncated: bool,
}

/// Result from analyzing a batch of texts.
#[derive(Debug, Clone)]
pub struct BatchSentimentResult {
    /// Per-text results.
    pub results: Vec<SentimentResult>,
    /// Mean sentiment score across the batch.
    pub mean_score: f64,
    /// Median sentiment score.
    pub median_score: f64,
    /// Standard deviation of scores.
    pub std_score: f64,
    /// Fraction of texts classified as positive.
    pub positive_fraction: f64,
    /// Fraction of texts classified as negative.
    pub negative_fraction: f64,
    /// Fraction of texts classified as neutral.
    pub neutral_fraction: f64,
    /// Mean confidence across the batch.
    pub mean_confidence: f64,
    /// Count of low-confidence predictions.
    pub low_confidence_count: usize,
    /// Total number of texts in the batch.
    pub count: usize,
}

// ---------------------------------------------------------------------------
// Windowing / Stats
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct WindowRecord {
    mean_score: f64,
    mean_confidence: f64,
    positive_fraction: f64,
    count: usize,
    tick: u64,
}

/// Accumulated statistics for the BERT sentiment analyzer.
pub struct BertSentimentStats {
    /// Total texts analyzed.
    pub total_texts: u64,
    /// Total batches processed.
    pub total_batches: u64,
    /// Current EMA sentiment score.
    pub ema_score: f64,
    /// Current EMA confidence.
    pub ema_confidence: f64,
    /// Count of low-confidence predictions.
    pub low_confidence_count: u64,
    /// Count of truncated inputs.
    pub truncated_count: u64,
    /// Sum of all scores (for computing mean).
    pub sum_score: f64,
    /// Sum of all confidences.
    pub sum_confidence: f64,
    /// Peak positive score observed.
    pub peak_positive: f64,
    /// Peak negative score observed.
    pub peak_negative: f64,
    /// Total inference time in milliseconds.
    pub total_inference_ms: f64,
}

impl Default for BertSentimentStats {
    fn default() -> Self {
        Self {
            total_texts: 0,
            total_batches: 0,
            ema_score: 0.0,
            ema_confidence: 0.0,
            low_confidence_count: 0,
            truncated_count: 0,
            sum_score: 0.0,
            sum_confidence: 0.0,
            peak_positive: f64::NEG_INFINITY,
            peak_negative: f64::INFINITY,
            total_inference_ms: 0.0,
        }
    }
}

impl BertSentimentStats {
    /// Mean sentiment score across all analyzed texts.
    pub fn mean_score(&self) -> f64 {
        if self.total_texts == 0 {
            0.0
        } else {
            self.sum_score / self.total_texts as f64
        }
    }

    /// Mean confidence across all analyzed texts.
    pub fn mean_confidence(&self) -> f64 {
        if self.total_texts == 0 {
            0.0
        } else {
            self.sum_confidence / self.total_texts as f64
        }
    }

    /// Rate of low-confidence predictions.
    pub fn low_confidence_rate(&self) -> f64 {
        if self.total_texts == 0 {
            0.0
        } else {
            self.low_confidence_count as f64 / self.total_texts as f64
        }
    }

    /// Rate of truncated inputs.
    pub fn truncation_rate(&self) -> f64 {
        if self.total_texts == 0 {
            0.0
        } else {
            self.truncated_count as f64 / self.total_texts as f64
        }
    }

    /// Mean inference time per batch.
    pub fn mean_inference_ms(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_inference_ms / self.total_batches as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// High-level BERT sentiment analyzer.
///
/// Wraps tokenizer + model + statistics in a convenient interface.
pub struct BertSentimentAnalyzer {
    config: BertSentimentConfig,
    tokenizer: BertTokenizer,
    model: BertModel,

    // EMA state
    ema_score: f64,
    ema_confidence: f64,
    ema_initialized: bool,

    // Windowed history
    recent: VecDeque<WindowRecord>,

    // Statistics
    stats: BertSentimentStats,

    // Tick counter
    current_tick: u64,
}

impl BertSentimentAnalyzer {
    /// Create a new analyzer with default FinBERT configuration.
    pub fn new() -> Self {
        Self::from_config(BertSentimentConfig::default())
    }

    /// Create a new analyzer with the given configuration.
    ///
    /// Note: This does NOT load model weights. Call `load_model()` before
    /// running inference. This allows construction to be infallible for
    /// testing and configuration.
    pub fn from_config(config: BertSentimentConfig) -> Self {
        let model_config = match &config.model_variant {
            BertModelVariant::FinBert => BertModelConfig::finbert(),
            BertModelVariant::DistilBert => BertModelConfig::distilbert(),
            BertModelVariant::Custom { num_labels, .. } => BertModelConfig {
                num_labels: *num_labels,
                ..Default::default()
            },
        };

        let tokenizer = BertTokenizer::new(config.max_seq_length);
        let model = BertModel::new(model_config, config.device);

        Self {
            tokenizer,
            model,
            ema_score: 0.0,
            ema_confidence: 0.0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(config.window_size),
            stats: BertSentimentStats::default(),
            current_tick: 0,
            config,
        }
    }

    /// Fallible constructor — validates config and returns an instance or error.
    pub fn with_config(config: BertSentimentConfig) -> Result<Self> {
        Self::validate_config(&config)?;
        Ok(Self::from_config(config))
    }

    /// Validate configuration parameters.
    fn validate_config(config: &BertSentimentConfig) -> Result<()> {
        if config.max_seq_length == 0 {
            return Err(Error::Configuration("max_seq_length must be > 0".into()));
        }
        if config.max_seq_length > 512 {
            return Err(Error::Configuration(
                "max_seq_length must be <= 512 for BERT models".into(),
            ));
        }
        if config.max_batch_size == 0 {
            return Err(Error::Configuration("max_batch_size must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::Configuration("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::Configuration("window_size must be > 0".into()));
        }
        if config.min_confidence < 0.0 || config.min_confidence > 1.0 {
            return Err(Error::Configuration(
                "min_confidence must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Load the model weights and tokenizer from disk.
    ///
    /// This resolves the model directory, loads the vocabulary, model config,
    /// and safetensors weights. Must be called before `analyze()`.
    pub fn load_model(&mut self) -> Result<()> {
        let model_dir = self.resolve_model_dir()?;

        // 1. Load tokenizer vocabulary
        let vocab_path = format!("{}/vocab.txt", model_dir);
        let tokenizer_json_path = format!("{}/tokenizer.json", model_dir);

        if std::path::Path::new(&tokenizer_json_path).exists() {
            self.tokenizer.load_from_json(&tokenizer_json_path)?;
        } else if std::path::Path::new(&vocab_path).exists() {
            self.tokenizer.load_vocab(&vocab_path)?;
        } else {
            return Err(Error::Configuration(format!(
                "No vocab.txt or tokenizer.json found in '{}'",
                model_dir
            )));
        }

        // 2. Load model config (if available, otherwise use defaults)
        let config_path = format!("{}/config.json", model_dir);
        if std::path::Path::new(&config_path).exists() {
            let model_config = BertModelConfig::from_json(&config_path)?;
            self.model = BertModel::new(model_config, self.config.device);
        }

        // 3. Load model weights
        let weights_path = format!("{}/model.safetensors", model_dir);
        if !std::path::Path::new(&weights_path).exists() {
            return Err(Error::Configuration(format!(
                "model.safetensors not found in '{}'",
                model_dir
            )));
        }

        self.model.load_weights(&weights_path)?;

        Ok(())
    }

    /// Resolve the model directory from configuration.
    fn resolve_model_dir(&self) -> Result<String> {
        // Explicit model_dir in config takes priority
        if let Some(ref dir) = self.config.model_dir {
            return Ok(dir.clone());
        }

        // Check variant-specific custom dir
        if let BertModelVariant::Custom { ref model_dir, .. } = self.config.model_variant {
            return Ok(model_dir.clone());
        }

        // Default cache directory: ~/.cache/janus/models/<variant_id>/
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let cache_dir = format!(
            "{}/.cache/janus/models/{}",
            home,
            self.config.model_variant.id()
        );

        if std::path::Path::new(&cache_dir).exists() {
            Ok(cache_dir)
        } else {
            Err(Error::Configuration(format!(
                "Model directory not found at '{}'. Please download the model or set model_dir in config.",
                cache_dir
            )))
        }
    }

    /// Check if the model is loaded and ready for inference.
    pub fn is_ready(&self) -> bool {
        self.model.is_loaded() && self.tokenizer.is_loaded()
    }

    /// Analyze a single text for sentiment.
    pub fn analyze(&mut self, text: &str) -> Result<SentimentResult> {
        let batch = self.analyze_batch(&[text])?;
        batch
            .results
            .into_iter()
            .next()
            .ok_or_else(|| Error::Internal("Empty result from batch analysis".into()))
    }

    /// Analyze a batch of texts for sentiment.
    ///
    /// Performs tokenization, runs the BERT forward pass (in sub-batches
    /// if needed), applies softmax, computes sentiment scores, and updates
    /// EMA / windowed statistics.
    pub fn analyze_batch(&mut self, texts: &[&str]) -> Result<BatchSentimentResult> {
        if texts.is_empty() {
            return Ok(BatchSentimentResult {
                results: Vec::new(),
                mean_score: 0.0,
                median_score: 0.0,
                std_score: 0.0,
                positive_fraction: 0.0,
                negative_fraction: 0.0,
                neutral_fraction: 0.0,
                mean_confidence: 0.0,
                low_confidence_count: 0,
                count: 0,
            });
        }

        if !self.is_ready() {
            return Err(Error::Configuration(
                "Model not loaded — call load_model() before analyze_batch()".into(),
            ));
        }

        let start = std::time::Instant::now();

        // Tokenize all texts
        let tokenized = self.tokenizer.tokenize_batch(texts);

        // Process in sub-batches
        let mut all_results: Vec<SentimentResult> = Vec::with_capacity(texts.len());
        let variant = self.config.model_variant.clone();
        let extract_embeddings = self.config.extract_embeddings;
        let min_confidence = self.config.min_confidence;

        for chunk in tokenized.chunks(self.config.max_batch_size) {
            let input_ids: Vec<Vec<u32>> = chunk.iter().map(|t| t.input_ids.clone()).collect();
            let attention_mask: Vec<Vec<u32>> =
                chunk.iter().map(|t| t.attention_mask.clone()).collect();
            let token_type_ids: Vec<Vec<u32>> =
                chunk.iter().map(|t| t.token_type_ids.clone()).collect();

            let output = self.model.forward(
                &input_ids,
                &attention_mask,
                &token_type_ids,
                extract_embeddings,
            )?;

            for (i, logits) in output.logits.iter().enumerate() {
                let probs = Self::softmax(logits);
                let score = Self::probabilities_to_score(&probs, &variant);
                let (label_index, confidence) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, &conf)| (idx, conf))
                    .unwrap_or((0, 0.0));

                let label = variant.label(label_index).to_string();
                let low_confidence = confidence < min_confidence;

                let embedding = output
                    .embeddings
                    .as_ref()
                    .and_then(|embs| embs.get(i).cloned());

                let tok = &chunk[i];

                all_results.push(SentimentResult {
                    text: tok.original_text.clone(),
                    score,
                    confidence,
                    label,
                    label_index,
                    class_probabilities: probs,
                    embedding,
                    low_confidence,
                    num_tokens: tok.num_tokens,
                    was_truncated: tok.was_truncated,
                });
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Compute batch statistics
        let count = all_results.len();
        let scores: Vec<f64> = all_results.iter().map(|r| r.score).collect();
        let confidences: Vec<f64> = all_results.iter().map(|r| r.confidence).collect();

        let mean_score = scores.iter().sum::<f64>() / count as f64;
        let mean_confidence = confidences.iter().sum::<f64>() / count as f64;

        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_score = if count % 2 == 0 && count > 0 {
            (sorted_scores[count / 2 - 1] + sorted_scores[count / 2]) / 2.0
        } else if count > 0 {
            sorted_scores[count / 2]
        } else {
            0.0
        };

        let variance = scores
            .iter()
            .map(|&s| (s - mean_score).powi(2))
            .sum::<f64>()
            / count as f64;
        let std_score = variance.sqrt();

        let positive_count = all_results.iter().filter(|r| r.score > 0.1).count();
        let negative_count = all_results.iter().filter(|r| r.score < -0.1).count();
        let neutral_count = count - positive_count - negative_count;
        let low_confidence_count = all_results.iter().filter(|r| r.low_confidence).count();

        let positive_fraction = positive_count as f64 / count as f64;
        let negative_fraction = negative_count as f64 / count as f64;
        let neutral_fraction = neutral_count as f64 / count as f64;

        // Update EMA
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.ema_score = mean_score;
            self.ema_confidence = mean_confidence;
            self.ema_initialized = true;
        } else {
            self.ema_score = alpha * mean_score + (1.0 - alpha) * self.ema_score;
            self.ema_confidence = alpha * mean_confidence + (1.0 - alpha) * self.ema_confidence;
        }

        // Update statistics
        self.stats.total_batches += 1;
        self.stats.total_texts += count as u64;
        self.stats.ema_score = self.ema_score;
        self.stats.ema_confidence = self.ema_confidence;
        self.stats.sum_score += scores.iter().sum::<f64>();
        self.stats.sum_confidence += confidences.iter().sum::<f64>();
        self.stats.low_confidence_count += low_confidence_count as u64;
        self.stats.truncated_count += all_results.iter().filter(|r| r.was_truncated).count() as u64;
        self.stats.total_inference_ms += elapsed_ms;

        for &s in &scores {
            if s > self.stats.peak_positive {
                self.stats.peak_positive = s;
            }
            if s < self.stats.peak_negative {
                self.stats.peak_negative = s;
            }
        }

        // Update window
        self.current_tick += 1;
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(WindowRecord {
            mean_score,
            mean_confidence,
            positive_fraction,
            count,
            tick: self.current_tick,
        });

        Ok(BatchSentimentResult {
            results: all_results,
            mean_score,
            median_score,
            std_score,
            positive_fraction,
            negative_fraction,
            neutral_fraction,
            mean_confidence,
            low_confidence_count,
            count,
        })
    }

    // -----------------------------------------------------------------------
    // Utility: softmax
    // -----------------------------------------------------------------------

    /// Compute softmax over a slice of logits.
    pub fn softmax(logits: &[f64]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        if sum_exp == 0.0 {
            return vec![1.0 / logits.len() as f64; logits.len()];
        }
        exps.iter().map(|&e| e / sum_exp).collect()
    }

    /// Convert class probabilities to a scalar sentiment score in [-1, 1].
    pub fn probabilities_to_score(probs: &[f64], variant: &BertModelVariant) -> f64 {
        match variant {
            BertModelVariant::FinBert => {
                // FinBERT: [positive=0, negative=1, neutral=2]
                let p_pos = probs.first().copied().unwrap_or(0.0);
                let p_neg = probs.get(1).copied().unwrap_or(0.0);
                // Score = P(positive) - P(negative), in [-1, 1]
                (p_pos - p_neg).clamp(-1.0, 1.0)
            }
            BertModelVariant::DistilBert => {
                // DistilBERT SST-2: [negative=0, positive=1]
                let p_neg = probs.first().copied().unwrap_or(0.0);
                let p_pos = probs.get(1).copied().unwrap_or(0.0);
                (p_pos - p_neg).clamp(-1.0, 1.0)
            }
            BertModelVariant::Custom { .. } => {
                // For custom models, assume first class is positive, second is negative
                let p_pos = probs.first().copied().unwrap_or(0.0);
                let p_neg = probs.get(1).copied().unwrap_or(0.0);
                (p_pos - p_neg).clamp(-1.0, 1.0)
            }
        }
    }

    // -----------------------------------------------------------------------
    // State & statistics
    // -----------------------------------------------------------------------

    /// Get the analyzer configuration.
    pub fn config(&self) -> &BertSentimentConfig {
        &self.config
    }

    /// Get the model variant.
    pub fn model_variant(&self) -> &BertModelVariant {
        &self.config.model_variant
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &BertSentimentStats {
        &self.stats
    }

    /// Get the current EMA sentiment score.
    pub fn smoothed_score(&self) -> f64 {
        self.ema_score
    }

    /// Get the current EMA confidence.
    pub fn smoothed_confidence(&self) -> f64 {
        self.ema_confidence
    }

    /// Windowed mean sentiment score.
    pub fn windowed_mean_score(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.mean_score).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean confidence.
    pub fn windowed_mean_confidence(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.mean_confidence).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed positive fraction.
    pub fn windowed_positive_fraction(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.positive_fraction).sum();
        sum / self.recent.len() as f64
    }

    /// Check if sentiment is trending positive over the recent window.
    pub fn is_sentiment_improving(&self) -> Option<bool> {
        if self.recent.len() < 3 {
            return None;
        }
        let n = self.recent.len();
        let half = n / 2;
        let first_mean: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.mean_score)
            .sum::<f64>()
            / half as f64;
        let second_mean: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.mean_score)
            .sum::<f64>()
            / (n - half) as f64;
        Some(second_mean > first_mean)
    }

    /// Get the approximate number of model parameters.
    pub fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }

    /// Reset all state and statistics.
    pub fn reset(&mut self) {
        self.ema_score = 0.0;
        self.ema_confidence = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = BertSentimentStats::default();
        self.current_tick = 0;
    }
}

impl Default for BertSentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Variant tests --

    #[test]
    fn test_finbert_variant() {
        let v = BertModelVariant::FinBert;
        assert_eq!(v.num_labels(), 3);
        assert_eq!(v.label(0), "positive");
        assert_eq!(v.label(1), "negative");
        assert_eq!(v.label(2), "neutral");
        assert_eq!(v.label(99), "unknown");
        assert_eq!(v.id(), "finbert");
    }

    #[test]
    fn test_distilbert_variant() {
        let v = BertModelVariant::DistilBert;
        assert_eq!(v.num_labels(), 2);
        assert_eq!(v.label(0), "negative");
        assert_eq!(v.label(1), "positive");
        assert_eq!(v.id(), "distilbert");
    }

    #[test]
    fn test_custom_variant() {
        let v = BertModelVariant::Custom {
            model_dir: "/tmp/model".into(),
            num_labels: 5,
            label_map: vec![
                "very_neg".into(),
                "neg".into(),
                "neutral".into(),
                "pos".into(),
                "very_pos".into(),
            ],
        };
        assert_eq!(v.num_labels(), 5);
        assert_eq!(v.label(0), "very_neg");
        assert_eq!(v.label(4), "very_pos");
        assert_eq!(v.label(99), "unknown");
        assert_eq!(v.id(), "custom");
    }

    #[test]
    fn test_variant_display() {
        assert!(format!("{}", BertModelVariant::FinBert).contains("FinBERT"));
        assert!(format!("{}", BertModelVariant::DistilBert).contains("DistilBERT"));
        let custom = BertModelVariant::Custom {
            model_dir: "/tmp".into(),
            num_labels: 2,
            label_map: vec![],
        };
        assert!(format!("{}", custom).contains("Custom"));
    }

    // -- Config tests --

    #[test]
    fn test_default_config_is_finbert() {
        let config = BertSentimentConfig::default();
        assert_eq!(config.model_variant, BertModelVariant::FinBert);
        assert_eq!(config.max_seq_length, 128);
        assert!(config.extract_embeddings);
    }

    #[test]
    fn test_distilbert_config() {
        let config = BertSentimentConfig::distilbert();
        assert_eq!(config.model_variant, BertModelVariant::DistilBert);
        assert_eq!(config.max_batch_size, 64);
    }

    #[test]
    fn test_config_validation_valid() {
        let config = BertSentimentConfig::finbert();
        assert!(BertSentimentAnalyzer::validate_config(&config).is_ok());
    }

    #[test]
    fn test_config_validation_bad_seq_length_zero() {
        let mut config = BertSentimentConfig::finbert();
        config.max_seq_length = 0;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_seq_length_too_large() {
        let mut config = BertSentimentConfig::finbert();
        config.max_seq_length = 1024;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_batch_size() {
        let mut config = BertSentimentConfig::finbert();
        config.max_batch_size = 0;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_ema_decay() {
        let mut config = BertSentimentConfig::finbert();
        config.ema_decay = 0.0;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());

        let mut config2 = BertSentimentConfig::finbert();
        config2.ema_decay = 1.0;
        assert!(BertSentimentAnalyzer::validate_config(&config2).is_err());
    }

    #[test]
    fn test_config_validation_bad_window_size() {
        let mut config = BertSentimentConfig::finbert();
        config.window_size = 0;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());
    }

    #[test]
    fn test_config_validation_bad_min_confidence() {
        let mut config = BertSentimentConfig::finbert();
        config.min_confidence = -0.1;
        assert!(BertSentimentAnalyzer::validate_config(&config).is_err());

        let mut config2 = BertSentimentConfig::finbert();
        config2.min_confidence = 1.5;
        assert!(BertSentimentAnalyzer::validate_config(&config2).is_err());
    }

    // -- Analyzer construction --

    #[test]
    fn test_analyzer_construction() {
        let analyzer = BertSentimentAnalyzer::new();
        assert_eq!(analyzer.config().model_variant, BertModelVariant::FinBert);
        assert!(!analyzer.is_ready());
    }

    #[test]
    fn test_analyzer_not_ready_without_loading() {
        let analyzer = BertSentimentAnalyzer::new();
        assert!(!analyzer.is_ready());
    }

    // -- Softmax tests --

    #[test]
    fn test_softmax_basic() {
        let probs = BertSentimentAnalyzer::softmax(&[2.0, 1.0, 0.1]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // First element should be largest
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = BertSentimentAnalyzer::softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_uniform() {
        let probs = BertSentimentAnalyzer::softmax(&[1.0, 1.0, 1.0]);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_large_values_no_overflow() {
        let probs = BertSentimentAnalyzer::softmax(&[1000.0, 999.0, 998.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_negative_values() {
        let probs = BertSentimentAnalyzer::softmax(&[-1.0, -2.0, -3.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[0] > probs[1]);
    }

    // -- Probabilities to score --

    #[test]
    fn test_probabilities_to_score_finbert() {
        // FinBERT: [positive=0, negative=1, neutral=2]
        // Pure positive
        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[0.9, 0.05, 0.05],
            &BertModelVariant::FinBert,
        );
        assert!((score - 0.85).abs() < 1e-10);

        // Pure negative
        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[0.05, 0.9, 0.05],
            &BertModelVariant::FinBert,
        );
        assert!((score - (-0.85)).abs() < 1e-10);

        // Neutral
        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[0.1, 0.1, 0.8],
            &BertModelVariant::FinBert,
        );
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_probabilities_to_score_distilbert() {
        // DistilBERT: [negative=0, positive=1]
        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[0.1, 0.9],
            &BertModelVariant::DistilBert,
        );
        assert!((score - 0.8).abs() < 1e-10);

        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[0.9, 0.1],
            &BertModelVariant::DistilBert,
        );
        assert!((score - (-0.8)).abs() < 1e-10);
    }

    #[test]
    fn test_probabilities_to_score_clamped() {
        let score = BertSentimentAnalyzer::probabilities_to_score(
            &[1.5, -0.5, 0.0],
            &BertModelVariant::FinBert,
        );
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_probabilities_to_score_empty() {
        let score = BertSentimentAnalyzer::probabilities_to_score(&[], &BertModelVariant::FinBert);
        assert!((score - 0.0).abs() < 1e-10);
    }

    // -- Tokenizer tests --

    #[test]
    fn test_tokenizer_construction() {
        let tok = BertTokenizer::new(128);
        assert_eq!(tok.max_seq_length, 128);
        assert!(!tok.is_loaded());
        assert_eq!(tok.vocab_size(), 0);
    }

    #[test]
    fn test_tokenizer_tokenize_placeholder() {
        // Without vocab loaded, tokens should be UNK
        let tok = BertTokenizer::new(128);
        let result = tok.tokenize("hello world");
        assert_eq!(result.input_ids.len(), 128);
        assert_eq!(result.input_ids[0], tok.cls_id); // [CLS]
        assert_eq!(result.input_ids[1], tok.unk_id); // hello -> UNK
        assert_eq!(result.input_ids[2], tok.unk_id); // world -> UNK
        assert_eq!(result.input_ids[3], tok.sep_id); // [SEP]
        assert_eq!(result.num_tokens, 4);
    }

    #[test]
    fn test_tokenizer_truncation() {
        let tok = BertTokenizer::new(5); // [CLS] + 3 tokens max + [SEP]
        let result = tok.tokenize("a b c d e f");
        assert!(result.was_truncated);
        assert_eq!(result.input_ids.len(), 5);
    }

    #[test]
    fn test_tokenizer_batch() {
        let tok = BertTokenizer::new(128);
        let results = tok.tokenize_batch(&["hello", "world"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_tokenizer_empty_input() {
        let tok = BertTokenizer::new(128);
        let result = tok.tokenize("");
        assert_eq!(result.num_tokens, 2); // [CLS] + [SEP]
    }

    // -- Model config tests --

    #[test]
    fn test_model_config_finbert() {
        let config = BertModelConfig::finbert();
        assert_eq!(config.num_labels, 3);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.hidden_size, 768);
    }

    #[test]
    fn test_model_config_distilbert() {
        let config = BertModelConfig::distilbert();
        assert_eq!(config.num_labels, 2);
        assert_eq!(config.num_hidden_layers, 6);
    }

    #[test]
    fn test_model_not_loaded_initially() {
        let model = BertModel::new(BertModelConfig::default(), InferenceDevice::Cpu);
        assert!(!model.is_loaded());
    }

    #[test]
    fn test_model_num_parameters() {
        let model = BertModel::new(BertModelConfig::finbert(), InferenceDevice::Cpu);
        let params = model.num_parameters();
        // BERT-base has ~110M parameters
        assert!(params > 100_000_000);
        assert!(params < 200_000_000);
    }

    // -- Stats tests --

    #[test]
    fn test_stats_defaults() {
        let stats = BertSentimentStats::default();
        assert_eq!(stats.total_texts, 0);
        assert_eq!(stats.total_batches, 0);
        assert_eq!(stats.mean_score(), 0.0);
        assert_eq!(stats.mean_confidence(), 0.0);
    }

    #[test]
    fn test_stats_mean_score() {
        let mut stats = BertSentimentStats::default();
        stats.total_texts = 10;
        stats.sum_score = 5.0;
        assert!((stats.mean_score() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stats_low_confidence_rate() {
        let mut stats = BertSentimentStats::default();
        stats.total_texts = 100;
        stats.low_confidence_count = 25;
        assert!((stats.low_confidence_rate() - 0.25).abs() < 1e-10);
    }

    // -- Analyzer state tests --

    #[test]
    fn test_analyzer_windowed_empty() {
        let analyzer = BertSentimentAnalyzer::new();
        assert_eq!(analyzer.windowed_mean_score(), 0.0);
        assert_eq!(analyzer.windowed_mean_confidence(), 0.0);
        assert_eq!(analyzer.windowed_positive_fraction(), 0.0);
    }

    #[test]
    fn test_analyzer_is_sentiment_improving_insufficient_data() {
        let analyzer = BertSentimentAnalyzer::new();
        assert!(analyzer.is_sentiment_improving().is_none());
    }

    #[test]
    fn test_analyzer_reset() {
        let mut analyzer = BertSentimentAnalyzer::new();
        analyzer.ema_score = 0.5;
        analyzer.ema_confidence = 0.8;
        analyzer.ema_initialized = true;
        analyzer.current_tick = 42;
        analyzer.stats.total_texts = 100;

        analyzer.reset();

        assert_eq!(analyzer.ema_score, 0.0);
        assert_eq!(analyzer.ema_confidence, 0.0);
        assert!(!analyzer.ema_initialized);
        assert_eq!(analyzer.current_tick, 0);
        assert_eq!(analyzer.stats.total_texts, 0);
    }

    #[test]
    fn test_analyze_empty_batch() {
        let mut analyzer = BertSentimentAnalyzer::new();
        let result = analyzer.analyze_batch(&[]).unwrap();
        assert_eq!(result.count, 0);
        assert!(result.results.is_empty());
        assert_eq!(result.mean_score, 0.0);
    }

    #[test]
    fn test_analyze_without_model_fails() {
        let mut analyzer = BertSentimentAnalyzer::new();
        let result = analyzer.analyze_batch(&["test"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_not_implemented() {
        let mut analyzer = BertSentimentAnalyzer::new();
        // load_model will fail because no model dir exists
        assert!(analyzer.load_model().is_err());
    }

    #[test]
    fn test_num_parameters() {
        let analyzer = BertSentimentAnalyzer::new();
        assert!(analyzer.num_parameters() > 0);
    }

    #[test]
    fn test_model_variant_accessor() {
        let analyzer = BertSentimentAnalyzer::new();
        assert_eq!(*analyzer.model_variant(), BertModelVariant::FinBert);
    }

    #[test]
    fn test_tokenizer_load_vocab_not_implemented() {
        let mut tok = BertTokenizer::new(128);
        assert!(tok.load_vocab("/nonexistent/path/vocab.txt").is_err());
    }

    #[test]
    fn test_tokenizer_load_json_not_implemented() {
        let mut tok = BertTokenizer::new(128);
        assert!(tok.load_from_json("/nonexistent/tokenizer.json").is_err());
    }

    #[test]
    fn test_model_config_from_json_not_implemented() {
        assert!(BertModelConfig::from_json("/nonexistent/config.json").is_err());
    }

    #[test]
    fn test_model_load_weights_not_implemented() {
        let mut model = BertModel::new(BertModelConfig::default(), InferenceDevice::Cpu);
        assert!(
            model
                .load_weights("/nonexistent/model.safetensors")
                .is_err()
        );
    }

    #[test]
    fn test_model_forward_without_weights_fails() {
        let model = BertModel::new(BertModelConfig::default(), InferenceDevice::Cpu);
        let result = model.forward(
            &[vec![101, 2003, 102]],
            &[vec![1, 1, 1]],
            &[vec![0, 0, 0]],
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_default_trait() {
        let analyzer = BertSentimentAnalyzer::default();
        assert!(!analyzer.is_ready());
    }

    #[test]
    fn test_tokenizer_attention_mask() {
        let tok = BertTokenizer::new(10);
        let result = tok.tokenize("hello");
        // [CLS] hello [SEP] = 3 real tokens, 7 padding
        assert_eq!(result.attention_mask[0], 1);
        assert_eq!(result.attention_mask[1], 1);
        assert_eq!(result.attention_mask[2], 1);
        assert_eq!(result.attention_mask[3], 0);
        assert_eq!(result.attention_mask.len(), 10);
    }

    #[test]
    fn test_tokenizer_lowercases_input() {
        let tok = BertTokenizer::new(128);
        let result = tok.tokenize("HELLO WORLD");
        // The original text is preserved
        assert_eq!(result.original_text, "HELLO WORLD");
        // But lowercasing happened internally (we can't directly verify
        // without vocab, but the tokenizer processes it)
        assert!(!result.was_truncated);
    }

    // -- WordPiece tokenizer tests --

    #[test]
    fn test_wordpiece_with_vocab() {
        let mut tok = BertTokenizer::new(128);
        // Simulate a small vocabulary
        tok.vocab.insert("[PAD]".to_string(), 0);
        tok.vocab.insert("[UNK]".to_string(), 100);
        tok.vocab.insert("[CLS]".to_string(), 101);
        tok.vocab.insert("[SEP]".to_string(), 102);
        tok.vocab.insert("hello".to_string(), 7592);
        tok.vocab.insert("world".to_string(), 2088);
        tok.vocab.insert("un".to_string(), 4895);
        tok.vocab.insert("##aff".to_string(), 3267);
        tok.vocab.insert("##able".to_string(), 4014);
        tok.id_to_token = vec!["[PAD]".into(); 8000];
        tok.cls_id = 101;
        tok.sep_id = 102;
        tok.pad_id = 0;
        tok.unk_id = 100;

        let result = tok.tokenize("hello world");
        assert_eq!(result.input_ids[0], 101); // [CLS]
        assert_eq!(result.input_ids[1], 7592); // hello
        assert_eq!(result.input_ids[2], 2088); // world
        assert_eq!(result.input_ids[3], 102); // [SEP]
        assert_eq!(result.num_tokens, 4);
        assert!(!result.was_truncated);
    }

    #[test]
    fn test_wordpiece_subword_tokenization() {
        let mut tok = BertTokenizer::new(128);
        tok.vocab.insert("[PAD]".to_string(), 0);
        tok.vocab.insert("[UNK]".to_string(), 100);
        tok.vocab.insert("[CLS]".to_string(), 101);
        tok.vocab.insert("[SEP]".to_string(), 102);
        tok.vocab.insert("un".to_string(), 200);
        tok.vocab.insert("##aff".to_string(), 201);
        tok.vocab.insert("##able".to_string(), 202);
        tok.cls_id = 101;
        tok.sep_id = 102;
        tok.pad_id = 0;
        tok.unk_id = 100;

        let result = tok.tokenize("unaffable");
        assert_eq!(result.input_ids[0], 101); // [CLS]
        assert_eq!(result.input_ids[1], 200); // un
        assert_eq!(result.input_ids[2], 201); // ##aff
        assert_eq!(result.input_ids[3], 202); // ##able
        assert_eq!(result.input_ids[4], 102); // [SEP]
        assert_eq!(result.num_tokens, 5);
    }

    #[test]
    fn test_wordpiece_unknown_word() {
        let mut tok = BertTokenizer::new(128);
        tok.vocab.insert("[PAD]".to_string(), 0);
        tok.vocab.insert("[UNK]".to_string(), 100);
        tok.vocab.insert("[CLS]".to_string(), 101);
        tok.vocab.insert("[SEP]".to_string(), 102);
        tok.vocab.insert("hello".to_string(), 200);
        tok.cls_id = 101;
        tok.sep_id = 102;
        tok.pad_id = 0;
        tok.unk_id = 100;

        let result = tok.tokenize("hello xyz");
        assert_eq!(result.input_ids[0], 101); // [CLS]
        assert_eq!(result.input_ids[1], 200); // hello
        assert_eq!(result.input_ids[2], 100); // xyz -> [UNK]
        assert_eq!(result.input_ids[3], 102); // [SEP]
    }

    #[test]
    fn test_pre_tokenize_punctuation() {
        let tok = BertTokenizer::new(128);
        let words = tok.pre_tokenize("hello, world! test");
        assert_eq!(words, vec!["hello", ",", "world", "!", "test"]);
    }

    #[test]
    fn test_pre_tokenize_multiple_spaces() {
        let tok = BertTokenizer::new(128);
        let words = tok.pre_tokenize("hello   world");
        assert_eq!(words, vec!["hello", "world"]);
    }

    // -- Model initialization test --

    #[test]
    fn test_model_initialize_tiny() {
        // Use a tiny config so initialization is fast
        let tiny_config = BertModelConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_labels: 3,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
        };

        let mut model = BertModel::new(tiny_config, InferenceDevice::Cpu);
        model.initialize().unwrap();

        // After initialize, loaded is not set (no weights from disk)
        // But the layers are allocated
        assert!(model.embeddings.is_some());
        assert!(model.encoder.is_some());
        assert!(model.pooler.is_some());
        assert!(model.classifier.is_some());
    }

    #[test]
    fn test_model_forward_with_initialized_tiny() {
        let tiny_config = BertModelConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_labels: 3,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
        };

        let mut model = BertModel::new(tiny_config, InferenceDevice::Cpu);
        model.initialize().unwrap();
        model.loaded = true; // Override for testing

        let input_ids = vec![vec![1u32, 2, 3, 0, 0]];
        let attention_mask = vec![vec![1u32, 1, 1, 0, 0]];
        let token_type_ids = vec![vec![0u32; 5]];

        let output = model
            .forward(&input_ids, &attention_mask, &token_type_ids, true)
            .unwrap();

        assert_eq!(output.logits.len(), 1); // batch size 1
        assert_eq!(output.logits[0].len(), 3); // 3 labels
        assert!(output.embeddings.is_some());
        assert_eq!(output.embeddings.as_ref().unwrap().len(), 1);
        assert_eq!(output.embeddings.as_ref().unwrap()[0].len(), 32); // hidden_size
    }

    #[test]
    fn test_model_forward_batch_with_initialized_tiny() {
        let tiny_config = BertModelConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_labels: 2,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
        };

        let mut model = BertModel::new(tiny_config, InferenceDevice::Cpu);
        model.initialize().unwrap();
        model.loaded = true;

        let input_ids = vec![
            vec![1u32, 5, 10, 3, 0],
            vec![1u32, 7, 12, 3, 0],
            vec![1u32, 8, 15, 3, 0],
        ];
        let attention_mask = vec![
            vec![1u32, 1, 1, 1, 0],
            vec![1u32, 1, 1, 1, 0],
            vec![1u32, 1, 1, 1, 0],
        ];
        let token_type_ids = vec![vec![0u32; 5]; 3];

        let output = model
            .forward(&input_ids, &attention_mask, &token_type_ids, false)
            .unwrap();

        assert_eq!(output.logits.len(), 3); // batch size 3
        for logits in &output.logits {
            assert_eq!(logits.len(), 2); // 2 labels
        }
        assert!(output.embeddings.is_none()); // extract_embeddings = false
    }

    #[test]
    fn test_model_forward_empty_batch() {
        let tiny_config = BertModelConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            num_labels: 3,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
        };

        let mut model = BertModel::new(tiny_config, InferenceDevice::Cpu);
        model.initialize().unwrap();
        model.loaded = true;

        let output = model.forward(&[], &[], &[], false).unwrap();
        assert!(output.logits.is_empty());
        assert!(output.embeddings.is_none());
    }

    #[test]
    fn test_head_dim() {
        let config = BertModelConfig::finbert();
        assert_eq!(config.head_dim(), 64); // 768 / 12
    }

    #[test]
    fn test_stats_truncation_rate() {
        let mut stats = BertSentimentStats::default();
        stats.total_texts = 100;
        stats.truncated_count = 10;
        assert!((stats.truncation_rate() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean_inference_ms() {
        let mut stats = BertSentimentStats::default();
        stats.total_batches = 5;
        stats.total_inference_ms = 500.0;
        assert!((stats.mean_inference_ms() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_cjk_char() {
        assert!(is_cjk_char('\u{4E00}')); // CJK start
        assert!(is_cjk_char('\u{9FFF}')); // CJK end
        assert!(!is_cjk_char('A'));
        assert!(!is_cjk_char('1'));
    }

    #[test]
    fn test_resolve_device() {
        let d = BertModel::resolve_device(InferenceDevice::Cpu);
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn test_model_config_defaults() {
        let config = BertModelConfig::default();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.max_position_embeddings, 512);
    }
}
