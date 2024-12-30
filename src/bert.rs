// src/bert.rs

use ndarray::{Array1, Array2, Axis, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use rand::distributions::Uniform;
use rand::Rng;

/// Enumeration for different embedding formats
#[derive(Debug, Clone, Copy)]
pub enum EmbeddingFormat {
    Word2Vec,
    GloVe,
    FastText,
    // Extend as needed
}

/// Configuration for the BERT model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BertConfig {
    pub hidden_size: usize,             // Should match embedding dimensions (300 for GloVe 300d)
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub dropout_prob: f64,
}

impl BertConfig {
    /// Load configuration from a JSON file
    pub fn load_from_file<P: AsRef<std::path::Path>>(config_path: P) -> std::io::Result<Self> {
        let config_json = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(config)
    }
}

pub struct MultiHotTokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    unknown_token: String,
    embedding_size: usize,
}

impl MultiHotTokenizer {
    pub fn new(vocab: HashMap<String, usize>, unknown_token: &str, embedding_size: usize) -> Self {
        let reverse_vocab = vocab
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect::<HashMap<usize, String>>();
        Self {
            vocab,
            reverse_vocab,
            unknown_token: unknown_token.to_string(),
            embedding_size,
        }
    }

    /// Tokenize input text into a multi-hot vector
    pub fn tokenize_to_multihot(&self, text: &str) -> Array1<f32> {
        let mut multihot = Array1::zeros(self.vocab.len());

        for word in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(word) {
                if id < self.vocab.len() {
                    multihot[id] = 1.0;
                }
            } else {
                if let Some(&unk_id) = self.vocab.get(&self.unknown_token) {
                    if unk_id < self.vocab.len() {
                        multihot[unk_id] = 1.0;
                    }
                }
            }
        }

        multihot
    }

    /// Decode multi-hot vector back to text
    pub fn decode_multihot(&self, multihot: &Array1<f32>) -> String {
        multihot
            .indexed_iter()
            .filter_map(|(i, &v)| if v > 0.0 { self.reverse_vocab.get(&i).cloned() } else { None })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Getter for the vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

/// Tokenizer for BERT
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    reverse_vocab: HashMap<usize, String>,
    unknown_token: String,
}

impl Tokenizer {
    /// Creates a new Tokenizer instance
    pub fn new(vocab: HashMap<String, usize>, unknown_token: &str) -> Self {
        let reverse_vocab = vocab
            .iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect::<HashMap<usize, String>>();
        Self {
            vocab,
            reverse_vocab,
            unknown_token: unknown_token.to_string(),
        }
    }

    /// Tokenizes input text into token IDs with <START> and <END> tokens
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![
            *self.vocab.get("<START>").expect("Missing <START> token"),
        ];
        for word in text.split_whitespace() {
            let word_lower = word.to_lowercase();
            if let Some(&id) = self.vocab.get(&word_lower) {
                tokens.push(id);
            } else {
                tokens.push(*self.vocab.get(&self.unknown_token).unwrap_or(&1));
            }
        }
        tokens.push(*self.vocab.get("<END>").expect("Missing <END> token"));
        tokens
    }

    /// Decodes token IDs into text, removing <START> and <END> tokens
    pub fn decode_with_filter(&self, token_ids: &[usize]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| self.reverse_vocab.get(&id).cloned())
            .filter(|token| token != "<START>" && token != "<END>")
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Decodes a single token ID to text
    pub fn decode_token(&self, token_id: usize) -> String {
        self.reverse_vocab.get(&token_id).cloned().unwrap_or("<UNK>".to_string())
    }

    /// Getter for the vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }
}

/// Scaled Dot-Product Attention
fn attention(query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>) -> Array2<f32> {
    let d_k = query.shape()[1] as f32;
    let scores = query.dot(&key.t()) / d_k.sqrt();
    let max_scores = scores.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |acc, &x| f32::max(acc, x)));
    let scores = scores - &max_scores.insert_axis(Axis(1));
    let exp_scores = scores.mapv(|x| x.exp());
    let sum_exp = exp_scores.sum_axis(Axis(1)).insert_axis(Axis(1));
    let probabilities = exp_scores / sum_exp;
    probabilities.dot(value)
}

/// Multi-head attention mechanism
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MultiHeadAttention {
    pub head_dim: usize,
    pub num_heads: usize,
    // Weight matrices for query, key, value, and output
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub w_o: Array2<f32>,
}

impl MultiHeadAttention {
    /// Initializes a new MultiHeadAttention instance
    pub fn new(num_heads: usize, hidden_size: usize) -> Self {
        assert_eq!(
            hidden_size % num_heads,
            0,
            "Hidden size must be divisible by the number of heads"
        );
        let head_dim = hidden_size / num_heads;

        // Initialize weight matrices with uniform distribution
        let mut rng = rand::thread_rng();
        let range = Uniform::new(-0.05, 0.05);

        let w_q = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(&range));
        let w_k = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(&range));
        let w_v = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(&range));
        let w_o = Array2::from_shape_fn((hidden_size, hidden_size), |_| rng.sample(&range));

        MultiHeadAttention {
            head_dim,
            num_heads,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    /// Forward pass through the multi-head attention layer
    pub fn forward(&self, query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>) -> Array2<f32> {
        // Linear projections
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Split into heads
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        // Apply attention for each head and concatenate
        let mut attention_output = Array2::<f32>::zeros((query.shape()[0], self.num_heads * self.head_dim));
        for i in 0..self.num_heads {
            let q_i = q.slice(s![.., i * self.head_dim..(i + 1) * self.head_dim]).to_owned();
            let k_i = k.slice(s![.., i * self.head_dim..(i + 1) * self.head_dim]).to_owned();
            let v_i = v.slice(s![.., i * self.head_dim..(i + 1) * self.head_dim]).to_owned();
            let head_out = attention(&q_i, &k_i, &v_i);
            attention_output.slice_mut(s![.., i * self.head_dim..(i + 1) * self.head_dim])
                                .assign(&head_out);
        }

        // Final linear projection
        attention_output.dot(&self.w_o)
    }

    /// Splits the input into multiple heads
    fn split_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }
}

/// Layer Normalization
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerNorm {
    pub epsilon: f32,
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
}

impl LayerNorm {
    /// Initializes a new LayerNorm instance
    pub fn new(hidden_size: usize) -> Self {
        // Initialize gamma to ones and beta to zeros
        let gamma = Array1::from_elem(hidden_size, 1.0);
        let beta = Array1::from_elem(hidden_size, 0.0);
        LayerNorm {
            epsilon: 1e-12,
            gamma,
            beta,
        }
    }

    /// Forward pass through the layer normalization
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1));
        let normalized = (x - &mean) / &(var + self.epsilon).mapv(|x| x.sqrt());
        &normalized * &self.gamma + &self.beta
    }
}

/// Transformer Encoder Layer
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
    // Intermediate and output projections
    pub intermediate: Array2<f32>,
    pub output_proj: Array2<f32>,
}

impl TransformerLayer {
    /// Initializes a new TransformerLayer instance
    pub fn new(config: &BertConfig) -> Self {
        let attention = MultiHeadAttention::new(config.num_attention_heads, config.hidden_size);
        let layer_norm1 = LayerNorm::new(config.hidden_size);
        let layer_norm2 = LayerNorm::new(config.hidden_size);

        // Initialize intermediate and output projection matrices with uniform distribution
        let mut rng = rand::thread_rng();
        let range = Uniform::new(-0.05, 0.05);
        let intermediate = Array2::from_shape_fn((config.hidden_size, config.intermediate_size), |_| rng.sample(&range));
        let output_proj = Array2::from_shape_fn((config.intermediate_size, config.hidden_size), |_| rng.sample(&range));

        TransformerLayer {
            attention,
            layer_norm1,
            layer_norm2,
            intermediate,
            output_proj,
        }
    }

    /// Forward pass through the transformer encoder layer
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // Self-attention
        let attention_output = self.attention.forward(input, input, input);
        // Add & Norm
        let out1 = self.layer_norm1.forward(&(input + &attention_output));

        // Intermediate projection
        let intermediate = out1.dot(&self.intermediate);
        // Activation (ReLU)
        let intermediate = intermediate.mapv(|x| x.max(0.0));

        // Output projection
        let intermediate = intermediate.dot(&self.output_proj);
        // Add & Norm
        self.layer_norm2.forward(&(out1 + intermediate))
    }
}

/// BERT Model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Bert {
    pub config: BertConfig,
    pub layers: Vec<TransformerLayer>,
    pub embedding_matrix: Array2<f32>,
    pub word2vec_embeddings: HashMap<String, Array1<f32>>,
}

impl Bert {
    /// Initializes a new BERT model instance
    pub fn new(config: BertConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| TransformerLayer::new(&config))
            .collect();

        // Initialize embedding matrix with zeros; will be filled with Word2Vec vectors
        let embedding_matrix = Array2::<f32>::zeros((config.vocab_size, config.hidden_size));

        let word2vec_embeddings = HashMap::new(); // Initialize empty; will be loaded later

        Bert {
            config,
            layers,
            embedding_matrix,
            word2vec_embeddings,
        }
    }

    /// Forward pass through the BERT model using token IDs
    pub fn forward(&self, token_ids: &[usize]) -> Array2<f32> {
        // Convert token IDs to embeddings
        let mut embeddings = Array2::<f32>::zeros((token_ids.len(), self.config.hidden_size));
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id < self.config.vocab_size {
                embeddings.slice_mut(s![i, ..]).assign(&self.embedding_matrix.slice(s![token_id, ..]));
            } else {
                // Assign [UNK] embedding if id is out of bounds
                let unk_id = 1; // Assuming [UNK] is at index 1
                embeddings.slice_mut(s![i, ..]).assign(&self.embedding_matrix.slice(s![unk_id, ..]));
            }
        }

        // Pass embeddings through each transformer layer
        let mut hidden_states = embeddings;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states);
        }

        hidden_states
    }

    /// Computes cosine similarity between two vectors
    pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Loads Word2Vec embeddings into a HashMap for quick lookup
    pub fn load_word2vec<P: AsRef<std::path::Path>>(
        &mut self,
        word2vec_path: P,
        format: EmbeddingFormat,
    ) -> std::io::Result<()> {
        println!("Loading Word2Vec embeddings from {:?}", word2vec_path.as_ref());

        if !word2vec_path.as_ref().exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Word2Vec file not found at {:?}", word2vec_path.as_ref()),
            ));
        }

        let file = File::open(&word2vec_path)?;
        let reader = BufReader::new(file);
        let mut word2vec_map = HashMap::new();
        let mut line_count = 0;
        let mut skipped_vectors = 0;

        for line in reader.lines() {
            let line = line?;
            line_count += 1;

            // GloVe does not have a header line, so no need to skip
            let parts: Vec<&str> = line.split_whitespace().collect();

            match format {
                EmbeddingFormat::Word2Vec | EmbeddingFormat::GloVe | EmbeddingFormat::FastText => {
                    // For GloVe 840B 300d: 1 word + 300 dimensions = 301 parts
                    if parts.len() != self.config.hidden_size + 1 {
                        eprintln!(
                            "Warning: Line {} has {} parts instead of {}. Skipping.",
                            line_count,
                            parts.len(),
                            self.config.hidden_size + 1
                        );
                        skipped_vectors += 1;
                        continue;
                    }

                    let word = parts[0].to_string();
                    let vector_result: Result<Vec<f32>, _> = parts[1..]
                        .iter()
                        .map(|&x| x.parse::<f32>())
                        .collect();

                    match vector_result {
                        Ok(vector) => {
                            // Ensure the vector has exactly hidden_size dimensions
                            if vector.len() != self.config.hidden_size {
                                eprintln!(
                                    "Warning: Vector for word '{}' has dimension {}, expected {}. Skipping.",
                                    word,
                                    vector.len(),
                                    self.config.hidden_size
                                );
                                skipped_vectors += 1;
                                continue;
                            }
                            word2vec_map.insert(word, Array1::from(vector));
                        }
                        Err(e) => {
                            eprintln!(
                                "Error parsing vector for word '{}' on line {}: {}. Skipping.",
                                word, line_count, e
                            );
                            skipped_vectors += 1;
                            continue;
                        }
                    }
                },
                // Add more formats if needed
            }
        }

        println!(
            "Loaded {} Word2Vec vectors. Skipped {} vectors due to errors.",
            word2vec_map.len(),
            skipped_vectors
        );

        self.word2vec_embeddings = word2vec_map;

        Ok(())
    }

    /// Initializes the embedding matrix with Word2Vec vectors
    pub fn initialize_embeddings_with_word2vec(&mut self, vocab: &HashMap<String, usize>) -> std::io::Result<()> {
        // Assign Word2Vec vectors to the embedding matrix
        for (word, vector) in &self.word2vec_embeddings {
            if let Some(&id) = vocab.get(word) {
                if id < self.config.vocab_size {
                    self.embedding_matrix.slice_mut(s![id, ..]).assign(vector);
                }
            }
        }

        // Initialize OOV tokens randomly
        let mut rng = rand::thread_rng();
        let range = Uniform::new(-0.02, 0.02);
        for id in 0..self.config.vocab_size {
            let word = self.vocab_key(id, vocab);
            if !self.word2vec_embeddings.contains_key(&word) {
                let random_vec: Array1<f32> = Array1::from_shape_fn(self.config.hidden_size, |_| rng.sample(&range));
                self.embedding_matrix.slice_mut(s![id, ..]).assign(&random_vec);
            }
        }

        Ok(())
    }

    /// Helper function to retrieve a word from its ID
    fn vocab_key(&self, id: usize, vocab: &HashMap<String, usize>) -> String {
        // Reverse lookup
        for (word, &word_id) in vocab {
            if word_id == id {
                return word.clone();
            }
        }
        "[UNK]".to_string()
    }

    /// Saves the model's configuration, embedding matrix, and transformer layers to disk
    pub fn save_model<P: AsRef<std::path::Path>>(&self, config_path: P, embedding_path: P, layers_path: P) {
        // Save configuration as JSON
        let config_json =
            serde_json::to_string_pretty(&self.config).expect("Failed to serialize model configuration");
        std::fs::write(&config_path, config_json).expect("Failed to save model configuration");

        // Save embedding matrix as binary
        let mut embed_file = File::create(&embedding_path).expect("Failed to create embedding file");
        Self::write_array2(&self.embedding_matrix, &mut embed_file)
            .expect("Failed to write embedding matrix");

        // Save transformer layers
        let mut layers_file = File::create(&layers_path).expect("Failed to create layers file");
        for layer in &self.layers {
            // Serialize attention weights
            Self::write_array2(&layer.attention.w_q, &mut layers_file).expect("Failed to write w_q");
            Self::write_array2(&layer.attention.w_k, &mut layers_file).expect("Failed to write w_k");
            Self::write_array2(&layer.attention.w_v, &mut layers_file).expect("Failed to write w_v");
            Self::write_array2(&layer.attention.w_o, &mut layers_file).expect("Failed to write w_o");
            // Serialize intermediate and output projections
            Self::write_array2(&layer.intermediate, &mut layers_file).expect("Failed to write intermediate");
            Self::write_array2(&layer.output_proj, &mut layers_file).expect("Failed to write output_proj");
            // Serialize LayerNorm parameters (gamma and beta)
            Self::write_array1(&layer.layer_norm1.gamma, &mut layers_file).expect("Failed to write layer_norm1.gamma");
            Self::write_array1(&layer.layer_norm1.beta, &mut layers_file).expect("Failed to write layer_norm1.beta");
            Self::write_array1(&layer.layer_norm2.gamma, &mut layers_file).expect("Failed to write layer_norm2.gamma");
            Self::write_array1(&layer.layer_norm2.beta, &mut layers_file).expect("Failed to write layer_norm2.beta");
        }
    }

    /// Loads the model's configuration, embedding matrix, and transformer layers from disk
    pub fn load_model<P: AsRef<std::path::Path>>(
        config_path: P,
        embedding_path: P,
        layers_path: P,
        word2vec_path: P,
        format: EmbeddingFormat,
        vocab: &HashMap<String, usize>,
    ) -> std::io::Result<Self> {
        // Load configuration from JSON
        let config_json = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Load embedding matrix
        let mut embed_file = File::open(&embedding_path)?;
        let embedding_matrix = Self::read_array2(&mut embed_file)?;

        // Load transformer layers
        let mut layers_file = File::open(&layers_path)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let w_q = Self::read_array2(&mut layers_file)?;
            let w_k = Self::read_array2(&mut layers_file)?;
            let w_v = Self::read_array2(&mut layers_file)?;
            let w_o = Self::read_array2(&mut layers_file)?;
            let intermediate = Self::read_array2(&mut layers_file)?;
            let output_proj = Self::read_array2(&mut layers_file)?;
            let gamma1 = Self::read_array1(&mut layers_file)?;
            let beta1 = Self::read_array1(&mut layers_file)?;
            let gamma2 = Self::read_array1(&mut layers_file)?;
            let beta2 = Self::read_array1(&mut layers_file)?;

            let mut layer = TransformerLayer::new(&config);
            layer.attention.w_q = w_q;
            layer.attention.w_k = w_k;
            layer.attention.w_v = w_v;
            layer.attention.w_o = w_o;
            layer.intermediate = intermediate;
            layer.output_proj = output_proj;
            layer.layer_norm1.gamma = gamma1;
            layer.layer_norm1.beta = beta1;
            layer.layer_norm2.gamma = gamma2;
            layer.layer_norm2.beta = beta2;

            layers.push(layer);
        }

        // Initialize BERT model
        let mut bert = Bert {
            config,
            layers,
            embedding_matrix,
            word2vec_embeddings: HashMap::new(),
        };

        // Load Word2Vec embeddings
        bert.load_word2vec(word2vec_path, format)?;

        // Initialize embeddings with Word2Vec vectors
        bert.initialize_embeddings_with_word2vec(vocab)?;

        Ok(bert)
    }

    /// Helper function to write a 2D array to a writer in binary format
    fn write_array2<W: Write>(array: &Array2<f32>, writer: &mut W) -> std::io::Result<()> {
        let shape = array.shape();
        let num_dims = shape.len() as u64;
        writer.write_all(&num_dims.to_le_bytes())?;
        for &dim in shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }
        for &value in array.iter() {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    /// Helper function to read a 2D array from a reader in binary format
    fn read_array2<R: Read>(reader: &mut R) -> std::io::Result<Array2<f32>> {
        let mut len_buf = [0u8; 8];
        // Read the number of dimensions
        reader.read_exact(&mut len_buf)?;
        let num_dims = u64::from_le_bytes(len_buf) as usize;

        // Ensure that the number of dimensions is exactly 2
        if num_dims != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 2 dimensions for Array2",
            ));
        }

        // Read the shape dimensions
        let mut shape = [0usize; 2];
        for i in 0..2 {
            reader.read_exact(&mut len_buf)?;
            shape[i] = u64::from_le_bytes(len_buf) as usize;
        }

        // Read the data
        let total_elements = shape[0] * shape[1];
        let mut data = vec![0f32; total_elements];
        for value in data.iter_mut() {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *value = f32::from_le_bytes(buf);
        }

        Ok(Array2::from_shape_vec(shape, data).expect("Shape and data mismatch"))
    }

    /// Helper function to write a 1D array to a writer in binary format
    fn write_array1<W: Write>(array: &Array1<f32>, writer: &mut W) -> std::io::Result<()> {
        let shape = array.shape();
        let num_dims = shape.len() as u64;
        writer.write_all(&num_dims.to_le_bytes())?;
        for &dim in shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }
        for &value in array.iter() {
            writer.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }

    /// Helper function to read a 1D array from a reader in binary format
    fn read_array1<R: Read>(reader: &mut R) -> std::io::Result<Array1<f32>> {
        let mut len_buf = [0u8; 8];
        // Read the number of dimensions
        reader.read_exact(&mut len_buf)?;
        let num_dims = u64::from_le_bytes(len_buf) as usize;

        // Ensure that the number of dimensions is exactly 1
        if num_dims != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 dimension for Array1",
            ));
        }

        // Read the shape dimensions
        let mut shape = [0usize; 1];
        for i in 0..1 {
            reader.read_exact(&mut len_buf)?;
            shape[i] = u64::from_le_bytes(len_buf) as usize;
        }

        // Read the data
        let total_elements = shape[0];
        let mut data = vec![0f32; total_elements];
        for value in data.iter_mut() {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *value = f32::from_le_bytes(buf);
        }

        Ok(Array1::from_shape_vec(shape, data).expect("Shape and data mismatch"))
    }

    /// Compute softmax probabilities for each row in a 2D array
    pub fn softmax(logits: &Array2<f32>) -> Array2<f32> {
        // Initialize an empty Array2 with the same dimensions as logits
        let mut softmaxed = Array2::<f32>::zeros(logits.dim());

        // Iterate over each row to compute softmax
        for (i, row) in logits.rows().into_iter().enumerate() {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, |acc, x| f32::max(acc, x));
            let exp: Array1<f32> = row.mapv(|x| (x - max).exp());
            let sum = exp.sum();
            let softmax_values = exp / sum;
            softmaxed.row_mut(i).assign(&softmax_values);
        }

        softmaxed
    }

    /// Compute cross-entropy loss
    pub fn cross_entropy_loss(predictions: &Array2<f32>, targets: &[usize]) -> f32 {
        let mut loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            if i >= predictions.nrows() {
                break;
            }
            let prob = predictions[(i, target)];
            loss -= prob.ln();
        }
        loss / targets.len() as f32
    }

    /// Compute gradients for cross-entropy loss
    pub fn compute_gradients(predictions: &Array2<f32>, targets: &[usize], gradients: &mut Array2<f32>) {
        for (i, &target) in targets.iter().enumerate() {
            if i >= predictions.nrows() {
                continue;
            }
            for j in 0..predictions.ncols() {
                let grad = if j == target { predictions[(i, j)] - 1.0 } else { predictions[(i, j)] };
                gradients[(j, i)] += grad; // Note the transpose for embedding matrix
            }
        }
    }

    /// Generates a response for a given input text
    pub fn generate_response(&self, tokenizer: &Tokenizer, input_text: &str) -> String {
        // Step 1: Tokenize the input text
        let token_ids = tokenizer.tokenize(input_text);

        // Step 2: Perform a forward pass through the model
        let hidden_states = self.forward(&token_ids);

        // Step 3: For simplicity, take the last hidden state as the context
        let last_hidden_state = hidden_states.row(hidden_states.nrows() - 1);

        // Step 4: Compute logits for the vocabulary using a dummy linear layer
        // Note: In a real implementation, you'd have a proper generation head
        let logits = last_hidden_state.dot(&self.layers.last().unwrap().output_proj);

        // Step 5: Apply softmax to get probabilities
        let probabilities = Bert::softmax(&logits.insert_axis(Axis(0)));

        // Step 6: Select the token with the highest probability
        let predicted_id = probabilities
            .row(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(1); // Default to [UNK] if not found

        // Step 7: Decode the predicted token ID to get the response token
        tokenizer.decode_token(predicted_id)
    }
}

