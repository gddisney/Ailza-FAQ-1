use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::read_to_string;
use std::path::Path;
use std::sync::Mutex;

mod conversation;
mod model;
use conversation::{ConversationManager, RecognizedIntent};
use model::{DynamicIntent, EmbeddingModel};

// --- Data Structures ---
#[derive(Deserialize)]
struct EmotionData {
    emotions: HashMap<String, Vec<String>>,
}
#[derive(Serialize)]
struct AskResponse {
    response: String,
}
#[derive(Deserialize)]
struct AskRequest {
    user_input: String,
}
#[derive(Deserialize, Debug, Clone)]
struct FaqItem {
    input: String,
    response: String,
}

#[derive(Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    unknown_token_id: usize,
}
// ... (Tokenizer implementation is unchanged)
impl Tokenizer {
    pub fn new(vocab: HashMap<String, usize>) -> Self {
        let unknown_token_id = *vocab.get("[UNK]").unwrap_or(&1);
        Self {
            vocab,
            unknown_token_id,
        }
    }
    pub fn tokenize(&self, text: &str) -> (Vec<usize>, Vec<String>) {
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let ids = tokens
            .iter()
            .map(|token| *self.vocab.get(token).unwrap_or(&self.unknown_token_id))
            .collect();
        (ids, tokens)
    }
}

// --- Core FAQ Logic ---
struct FAQ {
    // ... (This struct is unchanged)
    answers: Vec<String>,
    question_embeddings: Array2<f32>,
    emotions_map: HashMap<String, String>,
    templates: HashMap<String, Vec<String>>,
}

impl FAQ {
    /// Finds the best answer by calculating cosine similarity.
    /// The threshold is now optional. If `None`, it returns the best match regardless of score.
    /// If `Some(t)`, it only returns a match if the score is >= t.
    fn find_best_answer(
        &self,
        input_embedding: &Array1<f32>,
        threshold: Option<f32>,
    ) -> Option<String> {
        let best_match = self
            .question_embeddings
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let score = EmbeddingModel::cosine_similarity(input_embedding, &row.to_owned());
                (i, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((best_index, best_score)) = best_match {
            match threshold {
                // If a threshold is provided, check if the best score meets it.
                Some(t) => {
                    if best_score >= t {
                        Some(self.answers[best_index].clone())
                    } else {
                        None // Best score was below the threshold
                    }
                }
                // If no threshold, we are in a lenient mode; return the best match.
                None => Some(self.answers[best_index].clone()),
            }
        } else {
            None // No questions to compare against
        }
    }

    fn detect_emotion_sentiment(&self, user_input: &str) -> Option<String> {
        let detected: HashSet<_> = user_input
            .split_whitespace()
            .filter_map(|w| self.emotions_map.get(&w.to_lowercase()))
            .collect();
        if detected.contains(&"joy".to_string()) {
            Some("positive".to_string())
        } else if detected.contains(&"sadness".to_string())
            || detected.contains(&"anger".to_string())
        {
            Some("negative".to_string())
        } else {
            None
        }
    }
    fn get_empathetic_response(&self, sentiment: &str) -> String {
        self.templates
            .get(sentiment)
            .and_then(|t| t.choose(&mut rand::thread_rng()).cloned())
            .unwrap_or_default()
    }
}

// --- Main Application System ---
pub struct FAQSystem {
    // ... (Struct definition is unchanged)
    faq: FAQ,
    model: EmbeddingModel,
    tokenizer: Tokenizer,
    conversation_manager: ConversationManager,
    idf_map: HashMap<String, f32>,
    similarity_threshold: f32,
}

impl FAQSystem {
    pub fn new(
        config: &config::Config,
        model: EmbeddingModel,
        tokenizer: Tokenizer,
        intents: Vec<DynamicIntent>,
    ) -> Result<Self> {
        let data_conf = config.get_table("data")?;
        let logic_conf = config.get_table("logic")?;
        let faq_file_path = data_conf["faq_file"].clone().into_string()?;
        let emotions_file_path = data_conf["emotions_file"].clone().into_string()?;
        let similarity_threshold = logic_conf["similarity_threshold"].clone().into_float()? as f32;

        let dataset = Self::load_faq_dataset(&faq_file_path)?;
        let (_questions, answers): (Vec<_>, Vec<_>) = dataset
            .iter()
            .map(|item| (item.input.clone(), item.response.clone()))
            .unzip();
        let questions: Vec<String> = dataset.iter().map(|item| item.input.clone()).collect();

        let idf_map = Self::calculate_idf(&questions);
        let question_embeddings =
            Self::create_batch_embeddings(&questions, &tokenizer, &model, &idf_map);
        let emotions_map = Self::load_emotions(&emotions_file_path)?;

        let templates = [
            (
                "positive",
                vec!["That's great to hear! How can I assist you further?"],
            ),
            (
                "negative",
                vec!["I'm sorry you're feeling that way. I'll do my best to help."],
            ),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
        .collect();

        let faq = FAQ {
            answers,
            question_embeddings,
            emotions_map,
            templates,
        };

        Ok(FAQSystem {
            faq,
            model,
            tokenizer,
            conversation_manager: ConversationManager::new(intents), // Initialize with dynamic intents
            idf_map,
            similarity_threshold,
        })
    }

    // ... (Helper functions like load_faq_dataset, calculate_idf, etc. are unchanged)
    fn load_faq_dataset(path: &str) -> Result<Vec<FaqItem>> {
        let file_content = read_to_string(path)?;
        Ok(file_content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect())
    }
    fn load_emotions(path: &str) -> Result<HashMap<String, String>> {
        let content = read_to_string(path)?;
        let data: EmotionData = serde_json::from_str(&content)?;
        let mut map = HashMap::new();
        for (emotion, words) in data.emotions {
            for word in words {
                map.insert(word.to_lowercase(), emotion.clone());
            }
        }
        Ok(map)
    }
    fn calculate_idf(questions: &[String]) -> HashMap<String, f32> {
        let total_docs = questions.len() as f32;
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for q in questions {
            let unique_tokens: HashSet<_> =
                q.split_whitespace().map(|w| w.to_lowercase()).collect();
            for token in unique_tokens {
                *doc_freq.entry(token).or_insert(0) += 1;
            }
        }
        doc_freq
            .into_iter()
            .map(|(token, freq)| (token, (total_docs / (1.0 + freq as f32)).log10()))
            .collect()
    }
    fn create_batch_embeddings(
        texts: &[String],
        tokenizer: &Tokenizer,
        model: &EmbeddingModel,
        idf_map: &HashMap<String, f32>,
    ) -> Array2<f32> {
        let mut embeddings = Array2::zeros((texts.len(), model.config.hidden_size));
        for (i, text) in texts.iter().enumerate() {
            let embedding = Self::get_sentence_embedding(text, tokenizer, model, idf_map);
            embeddings.row_mut(i).assign(&embedding);
        }
        embeddings
    }
    fn get_sentence_embedding(
        text: &str,
        tokenizer: &Tokenizer,
        model: &EmbeddingModel,
        idf_map: &HashMap<String, f32>,
    ) -> Array1<f32> {
        let (token_ids, tokens) = tokenizer.tokenize(text);
        if token_ids.is_empty() {
            return Array1::zeros(model.config.hidden_size);
        }
        let word_embeddings = model.get_embeddings_for_ids(&token_ids);
        let mut weighted_sum = Array1::<f32>::zeros(model.config.hidden_size);
        let mut total_weight = 0.0;
        for (i, token) in tokens.iter().enumerate() {
            let weight = *idf_map.get(token).unwrap_or(&1.0);
            weighted_sum.scaled_add(weight, &word_embeddings.row(i));
            total_weight += weight;
        }
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            word_embeddings
                .mean_axis(Axis(0))
                .unwrap_or_else(|| Array1::zeros(model.config.hidden_size))
        }
    }

    // UPDATED to use the new ConversationManager and adaptive threshold
    pub fn handle_user_input(&mut self, input: &str) -> String {
        let recognized_intent = self.conversation_manager.recognize_intent(input);
        let mut response_parts = Vec::new();

        if let Some(sentiment) = self.faq.detect_emotion_sentiment(input) {
            response_parts.push(self.faq.get_empathetic_response(&sentiment));
        }

        let search_query = match recognized_intent {
            RecognizedIntent::Clarification => {
                let last_topic = self.conversation_manager.last_topic.as_deref().unwrap_or("");
                let clarifying_text = input
                    .to_lowercase()
                    .replace("what about", "")
                    .replace("how about", "")
                    .replace("and ", "");
                format!("{} {}", last_topic, clarifying_text.trim())
            }
            _ => input.to_string(), // For Dynamic and General intents, search the original query
        };
        
        // **MODIFIED LOGIC**: Use an adaptive threshold based on the recognized intent.
        // For known topics (Dynamic/Clarification), we are more confident and can be lenient.
        // For General topics, we should be stricter to avoid irrelevant answers.
        let threshold_to_use = match recognized_intent {
            RecognizedIntent::Dynamic | RecognizedIntent::Clarification => None, // Always return the best match
            RecognizedIntent::General => Some(self.similarity_threshold), // Use strict threshold
        };

        let embedding =
            Self::get_sentence_embedding(&search_query, &self.tokenizer, &self.model, &self.idf_map);
            
        if let Some(answer) = self.faq.find_best_answer(&embedding, threshold_to_use) {
            response_parts.push(answer);
        }

        if response_parts.is_empty() {
            "I'm not sure how to answer that. Could you please rephrase your question?".to_string()
        } else {
            response_parts.join(" ")
        }
    }
}

// --- Web Server Endpoints (Unchanged) ---
#[post("/ask")]
async fn ask_endpoint(
    req: web::Json<AskRequest>,
    data: web::Data<Mutex<FAQSystem>>,
) -> impl Responder {
    let mut system = data.lock().unwrap();
    let response = system.handle_user_input(&req.user_input);
    HttpResponse::Ok().json(AskResponse { response })
}
#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("index.html"))
}

#[actix_web::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let settings = config::Config::builder()
        .add_source(config::File::with_name("Config"))
        .build()?;

    let cache_conf = settings.get_table("model_cache")?;
    let cache_dir_str = cache_conf["dir"].clone().into_string()?;
    let vocab_file_str = cache_conf["vocab_file"].clone().into_string()?;
    let config_file_str = cache_conf["config_file"].clone().into_string()?;
    let embedding_file_str = cache_conf["embedding_file"].clone().into_string()?;
    let intents_file_str = "intents.json"; // New file to load

    let cache_dir = Path::new(&cache_dir_str);
    let vocab_path = cache_dir.join(&vocab_file_str);
    let config_path = cache_dir.join(&config_file_str);
    let embedding_path = cache_dir.join(&embedding_file_str);
    let intents_path = cache_dir.join(intents_file_str);

    if !intents_path.exists() {
        log::error!("'intents.json' not found in '{:?}'.", cache_dir);
        log::error!("Please run the `preprocess` binary first to generate dynamic intents.");
        return Err(anyhow::anyhow!("Model files not found."));
    }

    // Load all necessary data
    let model =
        EmbeddingModel::load_model(config_path.to_str().unwrap(), embedding_path.to_str().unwrap())?;
    let vocab_map: HashMap<String, usize> = serde_json::from_str(&read_to_string(vocab_path)?)?;
    let tokenizer = Tokenizer::new(vocab_map);
    let intents_content = read_to_string(intents_path)?;
    let intents: Vec<DynamicIntent> = serde_json::from_str(&intents_content)?;

    log::info!(
        "Initializing conversational FAQ system with {} dynamic intents...",
        intents.len()
    );
    let system = FAQSystem::new(&settings, model, tokenizer, intents)?;
    let data = web::Data::new(Mutex::new(system));

    let server_conf = settings.get_table("server")?;
    let host = server_conf["host"].clone().into_string()?;
    let port = server_conf["port"].clone().into_int()? as u16;

    log::info!("Starting server at http://{}:{}", host, port);
    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .service(index)
            .service(ask_endpoint)
    })
    .bind((host, port))?
    .run()
    .await?;
    Ok(())
}
