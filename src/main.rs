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

// --- Data Structures (unchanged) ---
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

// --- Core FAQ Logic (unchanged) ---
struct FAQ {
    questions: Vec<String>,
    answers: Vec<String>,
    question_embeddings: Array2<f32>,
    emotions_map: HashMap<String, String>,
    templates: HashMap<String, Vec<String>>,
}

impl FAQ {
    fn find_top_answers(
        &self,
        input_embedding: &Array1<f32>,
        top_n: usize,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = self
            .question_embeddings
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let score = EmbeddingModel::cosine_similarity(input_embedding, &row.to_owned());
                (i, score)
            })
            .collect();

        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(top_n).collect()
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
    faq: FAQ,
    model: EmbeddingModel,
    tokenizer: Tokenizer,
    conversation_manager: ConversationManager,
    idf_map: HashMap<String, f32>,
    intent_embeddings: HashMap<String, Array1<f32>>,
    similarity_threshold: f32,
    // *** CHANGE 1: Add a field to store groups of identical questions ***
    question_groups: HashMap<String, Vec<usize>>,
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
        let (questions, answers): (Vec<_>, Vec<_>) = dataset
            .iter()
            .map(|item| (item.input.clone(), item.response.clone()))
            .unzip();

        // *** CHANGE 2: Group identical questions by their index during initialization ***
        let mut question_groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, question) in questions.iter().enumerate() {
            question_groups
                .entry(question.clone())
                .or_default()
                .push(i);
        }

        let idf_map = Self::calculate_idf(&questions);
        let question_embeddings =
            Self::create_batch_embeddings(&questions, &tokenizer, &model, &idf_map);
        let emotions_map = Self::load_emotions(&emotions_file_path)?;

        let mut intent_embeddings = HashMap::new();
        for intent in &intents {
            let intent_embedding =
                Self::get_sentence_embedding(&intent.name, &tokenizer, &model, &idf_map);
            intent_embeddings.insert(intent.name.clone(), intent_embedding);
        }

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
            questions,
            answers,
            question_embeddings,
            emotions_map,
            templates,
        };

        Ok(FAQSystem {
            faq,
            model,
            tokenizer,
            conversation_manager: ConversationManager::new(intents),
            idf_map,
            intent_embeddings,
            similarity_threshold,
            question_groups, // Add the new field here
        })
    }

    // ... (Helper functions like load_faq_dataset, etc. are unchanged)
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

    fn jaccard_similarity(text1: &str, text2: &str) -> f32 {
        let lower_text1 = text1.to_lowercase();
        let lower_text2 = text2.to_lowercase();
        let set1: HashSet<_> = lower_text1.split_whitespace().collect();
        let set2: HashSet<_> = lower_text2.split_whitespace().collect();
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    pub fn handle_user_input(&mut self, input: &str) -> String {
        let recognized_intent = self.conversation_manager.recognize_intent(input, &self.idf_map);
        let mut response_parts = Vec::new();

        if let Some(sentiment) = self.faq.detect_emotion_sentiment(input) {
            response_parts.push(self.faq.get_empathetic_response(&sentiment));
        }

        let mut current_topic: Option<String> = None;
        
        let search_embedding = match &recognized_intent {
            RecognizedIntent::Dynamic(topic) => {
                current_topic = Some(topic.clone());
                Self::get_sentence_embedding(input, &self.tokenizer, &self.model, &self.idf_map)
            }
            RecognizedIntent::Clarification => {
                let last_topic = self.conversation_manager.last_topic.as_deref().unwrap_or("");
                current_topic = Some(last_topic.to_string());
                let last_topic_embedding = self
                    .intent_embeddings
                    .get(last_topic)
                    .cloned()
                    .unwrap_or_else(|| Array1::zeros(self.model.config.hidden_size));

                let clarifying_text = input
                    .to_lowercase()
                    .replace("what about", "")
                    .replace("how about", "")
                    .replace("and ", "");
                let clarification_embedding = Self::get_sentence_embedding(
                    clarifying_text.trim(),
                    &self.tokenizer,
                    &self.model,
                    &self.idf_map,
                );
                
                let combined = (last_topic_embedding * 0.6) + (clarification_embedding * 0.4);
                let norm = combined.dot(&combined).sqrt();
                if norm > 0.0 { combined / norm } else { combined }
            }
            RecognizedIntent::General => {
                Self::get_sentence_embedding(input, &self.tokenizer, &self.model, &self.idf_map)
            }
        };
        
        let top_candidates = self.faq.find_top_answers(&search_embedding, 5);
        if !top_candidates.is_empty() {
            let best_reranked_match = top_candidates
                .iter()
                .map(|(idx, cosine_score)| {
                    let question_text = &self.faq.questions[*idx];
                    let jaccard_score = Self::jaccard_similarity(input, question_text);
                    let final_score = (0.7 * cosine_score) + (0.3 * jaccard_score);
                    (idx, final_score, cosine_score)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            if let Some((best_idx, _, best_cosine_score)) = best_reranked_match {
                let is_confident = match recognized_intent {
                    RecognizedIntent::General => *best_cosine_score >= self.similarity_threshold,
                    _ => true,
                };

                if is_confident {
                    // *** CHANGE 3: Implement dynamic response selection ***
                    // 1. Get the text of the best-matching question.
                    let matched_question_text = &self.faq.questions[*best_idx];

                    // 2. Find all other questions that are identical.
                    if let Some(indices) = self.question_groups.get(matched_question_text) {
                        // 3. Choose a random index from the group.
                        if let Some(chosen_index) = indices.choose(&mut rand::thread_rng()) {
                             // 4. Get the answer using the new random index.
                            let mut answer = self.faq.answers[*chosen_index].clone();
                            if let Some(topic) = current_topic {
                                answer = answer.replace("{topic}", &topic);
                            }
                            response_parts.push(answer);
                        }
                    } else {
                        // Fallback for safety, though it should not be reached.
                        let mut answer = self.faq.answers[*best_idx].clone();
                        if let Some(topic) = current_topic {
                           answer = answer.replace("{topic}", &topic);
                        }
                        response_parts.push(answer);
                    }
                }
            }
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
    // ... (main function is unchanged)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let settings = config::Config::builder()
        .add_source(config::File::with_name("Config"))
        .build()?;

    let cache_conf = settings.get_table("model_cache")?;
    let cache_dir_str = cache_conf["dir"].clone().into_string()?;
    let vocab_file_str = cache_conf["vocab_file"].clone().into_string()?;
    let config_file_str = cache_conf["config_file"].clone().into_string()?;
    let embedding_file_str = cache_conf["embedding_file"].clone().into_string()?;
    let intents_file_str = "intents.json";

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
