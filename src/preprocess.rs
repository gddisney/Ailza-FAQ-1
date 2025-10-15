use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
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

/// Analyzes the FAQ questions to automatically generate a list of intents.
fn generate_intents_from_faq(dataset: &[FaqItem]) -> Vec<DynamicIntent> {
    let stop_words: HashSet<&str> = STOP_WORDS.iter().cloned().collect();

    // Group all significant keywords by the primary keyword in a question.
    let mut intent_map: HashMap<String, HashSet<String>> = HashMap::new();

    for item in dataset {
        let question = item.input.replace("<PROMPT>", "").trim().to_lowercase();
        let keywords: Vec<String> = question.split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty() && !stop_words.contains(word.as_str()))
            .collect();

        if let Some(primary_keyword) = keywords.get(0) {
            let entry = intent_map.entry(primary_keyword.clone()).or_insert_with(HashSet::new);
            for kw in keywords {
                 entry.insert(kw);
            }
        }
    }
    
    // Convert the map into the final Vec<DynamicIntent> structure.
    intent_map.into_iter().map(|(name, keywords_set)| {
        DynamicIntent { name, keywords: keywords_set.into_iter().collect() }
    }).collect()
}


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
    vocab_map.insert("<START>".to_string(), 2);
    vocab_map.insert("<END>".to_string(), 3);

    for token in vocab_set {
        let len = vocab_map.len();
        vocab_map.entry(token).or_insert(len);
    }
    vocab_map
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

    // 3. Generate and Save Dynamic Intents
    log::info!("Generating dynamic intents from FAQ data...");
    let intents = generate_intents_from_faq(&dataset);
    let intents_path = cache_dir.join("intents.json");
    fs::write(&intents_path, serde_json::to_string_pretty(&intents)?)?;
    log::info!("{} dynamic intents saved to {:?}", intents.len(), intents_path);

    // 4. Generate and Save Vocabulary
    log::info!("Generating vocabulary...");
    let vocab_map = generate_vocab(&dataset);
    let vocab_path = cache_dir.join(cache_conf["vocab_file"].as_str().context("vocab_file missing")?);
    fs::write(&vocab_path, serde_json::to_string_pretty(&vocab_map)?)?;
    log::info!("Vocabulary with {} tokens saved to {:?}", vocab_map.len(), vocab_path);

    // 5. Create and Save ModelConfig
    let model_config = ModelConfig { hidden_size, vocab_size: vocab_map.len() };
    let config_path = cache_dir.join(cache_conf["config_file"].as_str().context("config_file missing")?);
    model_config.save_to_file(&config_path)?;
    log::info!("Model config saved to {:?}", config_path);

    // 6. Build and Save Embedding Matrix
    log::info!("Building embedding matrix...");
    let glove_map = load_glove_file(glove_file, hidden_size)?;
    let mut embedding_matrix = Array2::<f32>::zeros((model_config.vocab_size, model_config.hidden_size));
    let mut rng = rand::thread_rng();
    let range = Uniform::new(-0.05, 0.05);
    
    for (word, &idx) in &vocab_map {
        if let Some(vector) = glove_map.get(word) {
            embedding_matrix.row_mut(idx).assign(vector);
        } else {
            let random_vec: Array1<f32> = Array1::from_shape_fn(hidden_size, |_| rng.sample(&range));
            embedding_matrix.row_mut(idx).assign(&random_vec);
        }
    }
    
    let embedding_path = cache_dir.join(cache_conf["embedding_file"].as_str().context("embedding_file missing")?);
    let mut file = File::create(&embedding_path)?;
    write_array2(&embedding_matrix, &mut file)?;
    log::info!("Embedding matrix saved to {:?}", embedding_path);
    
    log::info!("Preprocessing complete!");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error during preprocessing: {:?}", e);
        std::process::exit(1);
    }
}
