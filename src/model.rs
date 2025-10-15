#![allow(dead_code)] // Suppress dead code warnings for this shared module

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

// This struct holds the dynamically generated intents
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DynamicIntent {
    pub name: String,
    pub keywords: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl ModelConfig {
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let config_json = std::fs::read_to_string(path_ref)?;
        let config: ModelConfig = serde_json::from_str(&config_json)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let config_json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path_ref, config_json)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    pub config: ModelConfig,
    pub embedding_matrix: Array2<f32>,
}

impl EmbeddingModel {
    pub fn load_model(config_path: &str, embedding_path: &str) -> Result<Self> {
        let config = ModelConfig::load_from_file(config_path)?;
        let mut embed_file = File::open(embedding_path)?;
        let embedding_matrix = read_array2(&mut embed_file)?;
        Ok(EmbeddingModel {
            config,
            embedding_matrix,
        })
    }

    pub fn get_embeddings_for_ids(&self, token_ids: &[usize]) -> Array2<f32> {
        let mut embeddings = Array2::<f32>::zeros((token_ids.len(), self.config.hidden_size));
        for (i, &token_id) in token_ids.iter().enumerate() {
            let id_to_use = if token_id < self.config.vocab_size {
                token_id
            } else {
                1
            };
            embeddings
                .row_mut(i)
                .assign(&self.embedding_matrix.row(id_to_use));
        }
        embeddings
    }

    pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

pub fn write_array2<W: Write>(array: &Array2<f32>, writer: &mut W) -> Result<()> {
    let shape = array.shape();
    writer.write_all(&(shape[0] as u64).to_le_bytes())?;
    writer.write_all(&(shape[1] as u64).to_le_bytes())?;
    for &value in array.iter() {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_array2<R: Read>(reader: &mut R) -> Result<Array2<f32>> {
    let mut shape_buf = [0u8; 8];
    reader.read_exact(&mut shape_buf)?;
    let rows = u64::from_le_bytes(shape_buf) as usize;
    reader.read_exact(&mut shape_buf)?;
    let cols = u64::from_le_bytes(shape_buf) as usize;
    let total_elements = rows * cols;
    let mut data = vec![0f32; total_elements];
    let mut float_buf = [0u8; 4];
    for value in data.iter_mut() {
        reader.read_exact(&mut float_buf)?;
        *value = f32::from_le_bytes(float_buf);
    }
    Array2::from_shape_vec((rows, cols), data).context("Shape and data mismatch")
}
