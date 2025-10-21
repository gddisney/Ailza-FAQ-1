use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

#[path = "../model.rs"]
mod model;
use model::{DynamicIntent, ModelConfig, write_array2};

#[derive(Deserialize, Debug, Clone)]
struct FaqItem {
    input: String,
    response: String,
}

/// A large, but not exhaustive, list of common English stop words.
const STOP_WORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "is", "are", "do", "does", "what",
    "how", "can", "the", "a", "an", "of", "to", "in", "for", "on", "with", "or"
];

// --- Word Embedding Training & Fine-Tuning ---

/// Sigmoid activation function used for negative sampling.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Generates (center_word, context_word) pairs for Skip-gram training.
fn generate_skipgram_pairs(
    dataset: &[FaqItem],
    vocab_map: &HashMap<String, usize>,
    window_size: usize,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let texts: Vec<String> = dataset
        .iter()
        .flat_map(|item| vec![item.input.clone(), item.response.clone()])
        .collect();

    for text in texts {
        let cleaned_text = text
            .replace("<PROMPT>", "")
            .replace("<START>", "")
            .replace("<END>", "")
            .to_lowercase();
        let word_ids: Vec<usize> = cleaned_text
            .split_whitespace()
            .filter_map(|word| vocab_map.get(word))
            .cloned()
            .collect();

        for i in 0..word_ids.len() {
            let center_word_id = word_ids[i];
            let start = i.saturating_sub(window_size);
            let end = (i + window_size).min(word_ids.len() - 1);

            for j in start..=end {
                if i == j {
                    continue;
                }
                let context_word_id = word_ids[j];
                pairs.push((center_word_id, context_word_id));
            }
        }
    }
    pairs
}

/// Loads GloVe embeddings from a text file into a HashMap.
fn load_glove_file(path: &str, expected_dim: usize) -> Result<HashMap<String, Array1<f32>>> {
    log::info!("Loading GloVe embeddings from '{}'...", path);
    let file = File::open(path).with_context(|| format!("Failed to open GloVe file at '{}'", path))?;
    let reader = BufReader::new(file);
    let mut glove_map = HashMap::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != expected_dim + 1 {
            log::warn!("Skipping malformed line #{} in GloVe file", i + 1);
            continue;
        }
        let word = parts[0].to_string();
        if let Ok(vector) = parts[1..].iter().map(|&s| s.parse()).collect::<Result<Vec<f32>, _>>() {
            glove_map.insert(word, Array1::from(vector));
        } else {
             log::warn!("Skipping line #{} due to vector parsing error", i + 1);
        }
    }
    log::info!("Loaded {} vectors from GloVe file.", glove_map.len());
    Ok(glove_map)
}

/// Trains custom word embeddings using Skip-gram with Negative Sampling for speed.
fn train_word_embeddings(
    dataset: &[FaqItem],
    vocab_map: &HashMap<String, usize>,
    hidden_size: usize,
    glove_file: &str,
) -> Result<Array2<f32>> {
    log::info!("Generating Skip-gram training pairs...");
    let mut pairs = generate_skipgram_pairs(dataset, vocab_map, 2);
    log::info!("Generated {} training pairs.", pairs.len());

    let vocab_size = vocab_map.len();
    let learning_rate = 0.025;
    let epochs = 20;
    let num_negative_samples = 5;

    // --- Initialize with GloVe (Fine-tuning) ---
    log::info!("Initializing embedding matrix with GloVe vectors...");
    let glove_map = load_glove_file(glove_file, hidden_size)?;
    let mut w_in = Array2::zeros((vocab_size, hidden_size));
    let mut rng = rand::thread_rng();

    for (word, &idx) in vocab_map.iter() {
        if let Some(vector) = glove_map.get(word) {
            w_in.row_mut(idx).assign(vector);
        } else {
            let random_vec: Array1<f32> = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-0.5..0.5) / hidden_size as f32);
            w_in.row_mut(idx).assign(&random_vec);
        }
    }
    log::info!("Seeded vectors from GloVe. Starting fine-tuning with Negative Sampling...");
    
    // The output matrix is also an embedding matrix in this model
    let mut w_out = Array2::from_shape_fn((vocab_size, hidden_size), |_| 0.0f32);

    let word_ids: Vec<usize> = (0..vocab_size).collect();

    for epoch in 0..epochs {
        pairs.shuffle(&mut rng);
        let mut total_loss = 0.0;
        let mut pair_count = 0;

        for (center_word_id, context_word_id) in pairs.iter() {
            let h = w_in.row(*center_word_id).to_owned();
            let mut grad_h = Array1::zeros(hidden_size);

            // --- Positive Sample (target = 1) ---
            let target_vector = w_out.row(*context_word_id);
            let score = h.dot(&target_vector);
            let prob = sigmoid(score);
            let error = prob - 1.0; 
            
            total_loss -= (prob + 1e-9).ln();
            grad_h.scaled_add(error, &target_vector);
            w_out.row_mut(*context_word_id).scaled_add(-learning_rate * error, &h);

            // --- Negative Samples (target = 0) ---
            let negative_samples = word_ids.choose_multiple(&mut rng, num_negative_samples);
            for neg_sample_id in negative_samples {
                if *neg_sample_id == *context_word_id { continue; }

                let target_vector = w_out.row(*neg_sample_id);
                let score = h.dot(&target_vector);
                let prob = sigmoid(score);
                let error = prob - 0.0;

                total_loss -= (1.0 - prob + 1e-9).ln();
                grad_h.scaled_add(error, &target_vector);
                w_out.row_mut(*neg_sample_id).scaled_add(-learning_rate * error, &h);
            }
            
            w_in.row_mut(*center_word_id).scaled_add(-learning_rate, &grad_h);
            pair_count += 1;
        }
        log::info!("Epoch {}/{}, Average Loss: {}", epoch + 1, epochs, total_loss / pair_count as f32);
    }
    
    log::info!("Word embedding fine-tuning complete.");
    // Averaging the input and output matrices is a common technique to get the final embeddings
    Ok((w_in + w_out) / 2.0)
}

// --- Vocabulary and Intent Generation ---

/// Generates a vocabulary from the entire FAQ dataset (questions and answers).
fn generate_vocab(dataset: &[FaqItem]) -> HashMap<String, usize> {
    let mut vocab_set = HashSet::new();
    for item in dataset {
        vocab_set.extend(item.input.split_whitespace().map(|w| w.to_lowercase()));
        vocab_set.extend(item.response.split_whitespace().map(|w| w.to_lowercase()));
    }

    let mut vocab_map = HashMap::new();
    vocab_map.insert("[PAD]".to_string(), 0);
    vocab_map.insert("[UNK]".to_string(), 1);

    // Remove special tokens as they are not part of the learnable vocabulary
    vocab_set.remove("<start>");
    vocab_set.remove("<end>");
    vocab_set.remove("<prompt>");

    for token in vocab_set {
        if !vocab_map.contains_key(&token) {
            let len = vocab_map.len();
            vocab_map.insert(token, len);
        }
    }
    vocab_map
}

/// Helper for K-Means clustering to create sentence embeddings.
fn get_sentence_embedding(
    text: &str,
    vocab_map: &HashMap<String, usize>,
    embedding_matrix: &Array2<f32>,
) -> Array1<f32> {
    let unknown_token_id = *vocab_map.get("[UNK]").unwrap_or(&1);
    let token_ids: Vec<usize> = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .map(|token| *vocab_map.get(&token).unwrap_or(&unknown_token_id))
        .collect();

    if token_ids.is_empty() {
        return Array1::zeros(embedding_matrix.ncols());
    }

    let mut embeddings = Array2::<f32>::zeros((token_ids.len(), embedding_matrix.ncols()));
    for (i, &token_id) in token_ids.iter().enumerate() {
        if token_id < embedding_matrix.nrows() {
            embeddings.row_mut(i).assign(&embedding_matrix.row(token_id));
        } else {
            embeddings.row_mut(i).assign(&embedding_matrix.row(unknown_token_id));
        }
    }

    embeddings.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(embedding_matrix.ncols()))
}

/// Simple distance function for K-Means.
fn squared_euclidean_distance(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    (a - b).mapv(|x| x.powi(2)).sum()
}

/// Analyzes FAQ questions using a homebrewed K-Means algorithm to generate intents.
fn generate_intents_from_faq(
    dataset: &[FaqItem],
    vocab_map: &HashMap<String, usize>,
    embedding_matrix: &Array2<f32>,
) -> Vec<DynamicIntent> {
    log::info!("Generating sentence embeddings for all questions...");
    let question_embeddings: Vec<Array1<f32>> = dataset
        .iter()
        .map(|item| get_sentence_embedding(&item.input, vocab_map, embedding_matrix))
        .collect();

    let num_questions = question_embeddings.len();
    if num_questions == 0 { return vec![]; }
    let embedding_dim = embedding_matrix.ncols();
    let num_clusters = ((num_questions as f32).sqrt() as usize).max(1);
    log::info!("Running homebrewed K-Means with {} clusters...", num_clusters);

    let mut rng = rand::thread_rng();
    let initial_indices = rand::seq::index::sample(&mut rng, num_questions, num_clusters).into_vec();
    let mut centroids: Vec<Array1<f32>> = initial_indices
        .into_iter()
        .map(|idx| question_embeddings[idx].clone())
        .collect();

    let mut assignments = vec![0; num_questions];
    let max_iterations = 20;

    for iter in 0..max_iterations {
        let mut changed = false;
        for (i, embedding) in question_embeddings.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0;
            for (cluster_id, centroid) in centroids.iter().enumerate() {
                let dist = squared_euclidean_distance(&embedding.view(), &centroid.view());
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_id;
                }
            }
            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        let mut new_centroids = vec![Array1::zeros(embedding_dim); num_clusters];
        let mut counts = vec![0; num_clusters];
        for (i, embedding) in question_embeddings.iter().enumerate() {
            let cluster_id = assignments[i];
            new_centroids[cluster_id] += embedding;
            counts[cluster_id] += 1;
        }

        for cluster_id in 0..num_clusters {
            if counts[cluster_id] > 0 {
                centroids[cluster_id] = &new_centroids[cluster_id] / counts[cluster_id] as f32;
            }
        }
        
        if !changed { 
             log::info!("K-Means converged after {} iterations.", iter + 1);
             break;
        }
    }
    
    let mut clustered_questions: HashMap<usize, Vec<&FaqItem>> = HashMap::new();
    for (i, &cluster_id) in assignments.iter().enumerate() {
        clustered_questions.entry(cluster_id).or_default().push(&dataset[i]);
    }

    log::info!("Aggregating keywords for {} generated intents...", clustered_questions.len());
    let stop_words: HashSet<&str> = STOP_WORDS.iter().cloned().collect();
    clustered_questions.into_iter().map(|(cluster_id, items)| {
        let mut intent_keywords = HashSet::new();
        for item in items {
            let keywords = item.input.split_whitespace()
                .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
                .filter(|word| !word.is_empty() && !stop_words.contains(word.as_str()));
            intent_keywords.extend(keywords);
        }
        DynamicIntent {
            name: format!("intent_{}", cluster_id),
            keywords: intent_keywords.into_iter().collect(),
        }
    }).collect()
}


/// Main preprocessing function.
fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. Load Configuration
    let settings = config::Config::builder()
        .add_source(config::File::with_name("Config"))
        .build()?
        .try_deserialize::<HashMap<String, serde_json::Value>>()?;
    
    let data_conf = settings.get("data").context("Missing [data] config")?.clone();
    let cache_conf = settings.get("model_cache").context("Missing [model_cache] config")?.clone();
    let params_conf = settings.get("model_params").context("Missing [model_params] config")?.clone();

    let faq_file = data_conf["faq_file"].as_str().context("faq_file path missing")?;
    let glove_file = data_conf["glove_file"].as_str().context("glove_file path missing")?;
    let hidden_size = params_conf["hidden_size"].as_u64().context("hidden_size missing")? as usize;

    let cache_dir = Path::new(cache_conf["dir"].as_str().context("cache dir missing")?);
    fs::create_dir_all(cache_dir)?;

    // 2. Load FAQ Data
    let file = File::open(faq_file)?;
    let reader = BufReader::new(file);
    let dataset: Vec<FaqItem> = reader.lines().filter_map(Result::ok).filter_map(|line| serde_json::from_str(&line).ok()).collect();
    log::info!("Loaded {} FAQ items.", dataset.len());

    // 3. Generate and Save Vocabulary
    log::info!("Generating vocabulary...");
    let vocab_map = generate_vocab(&dataset);
    let vocab_path = cache_dir.join(cache_conf["vocab_file"].as_str().context("vocab_file missing")?);
    fs::write(&vocab_path, serde_json::to_string_pretty(&vocab_map)?)?;
    log::info!("Vocabulary with {} tokens saved to {:?}", vocab_map.len(), vocab_path);

    // 4. Create and Save ModelConfig
    let model_config = ModelConfig { hidden_size, vocab_size: vocab_map.len() };
    let config_path = cache_dir.join(cache_conf["config_file"].as_str().context("config_file missing")?);
    model_config.save_to_file(&config_path)?;
    log::info!("Model config saved to {:?}", config_path);

    // 5. Train/Fine-tune and Save Embedding Matrix
    log::info!("Building and fine-tuning embedding matrix...");
    let embedding_matrix = train_word_embeddings(&dataset, &vocab_map, hidden_size, glove_file)?;
    
    let embedding_path = cache_dir.join(cache_conf["embedding_file"].as_str().context("embedding_file missing")?);
    let mut file = File::create(&embedding_path)?;
    write_array2(&embedding_matrix, &mut file)?;
    log::info!("Embedding matrix with shape {:?} saved to {:?}", embedding_matrix.shape(), embedding_path);

    // 6. Generate and Save Dynamic Intents (now uses trained embeddings)
    log::info!("Generating dynamic intents from FAQ data using trained embeddings...");
    let intents = generate_intents_from_faq(&dataset, &vocab_map, &embedding_matrix);
    let intents_path = cache_dir.join("intents.json");
    fs::write(&intents_path, serde_json::to_string_pretty(&intents)?)?;
    log::info!("{} dynamic intents saved to {:?}", intents.len(), intents_path);
    
    log::info!("Preprocessing complete!");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error during preprocessing: {:?}", e);
        std::process::exit(1);
    }
}
