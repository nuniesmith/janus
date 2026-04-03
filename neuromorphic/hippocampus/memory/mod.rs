//! Memory subsystem for hippocampus
//!
//! Provides vector database integration for persistent storage
//! of episodic memories, market patterns, and learned representations.
//!
//! # Architecture
//!
//! This module provides both concrete implementations and trait abstractions
//! for vector storage:
//!
//! - `VectorDB`: Real Qdrant-backed vector database
//! - `MockVectorDB`: In-memory mock for testing
//! - `VectorStorage`: Trait for abstracting storage operations
//! - `VectorStorageBackend`: Enum for runtime backend selection

pub mod mock_vector_db;
pub mod traits;
pub mod vector_db;

pub use mock_vector_db::MockVectorDB;
pub use traits::{VectorStorage, VectorStorageBackend};
pub use vector_db::{
    CollectionStats, EMBEDDING_DIM, EMBEDDINGS_COLLECTION, EPISODE_DIM, EPISODES_COLLECTION,
    MemoryEntry, MemoryType, PATTERN_DIM, PATTERNS_COLLECTION, SearchQuery, SearchResult, VectorDB,
};
